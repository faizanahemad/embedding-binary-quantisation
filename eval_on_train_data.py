  
import os  
import sys
import mteb
from sklearn.isotonic import spearmanr
from scipy.stats import pearsonr, spearmanr

import torch  
import torch.nn as nn  
import numpy as np  
from sentence_transformers import SentenceTransformer  
from mteb import MTEB, TaskResult  
from mteb.abstasks.AbsTask import AbsTask
from mteb.models.wrapper import Wrapper
from mteb.encoder_interface import Encoder
from config import base_model_name

from typing import List, Dict  
import pandas as pd  
from datetime import datetime  
import torch.nn.functional as F  
from transformers import AutoTokenizer, AutoModel  

# Assuming QuantizationModuleStage1 and QuantizationModuleStage2 are defined in quantization_modules.py  
from improved_quantisation_module import ImprovedQuantizationModule
from train import QuantizationModuleStage1, QuantizationModuleStage2, QuantizationModuleStage1WithScales, QuantizationModuleOneBitTwoBit
from config import save_dirs, test_modules
from tqdm import tqdm
from common import OriginalEmbeddingModel, QuantizedEmbeddingModel, get_dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 512 * 2

def check_loss(embedding_model, quantization_module, dataloader):
    average_loss = 0
    average_spearman_corr = 0
    average_pearson_corr = 0
    idx = 0
    for batch in tqdm(dataloader):  
        input_ids = batch['input_ids'].squeeze(1).to(device)  # Remove extra dimension  
        attention_mask = batch['attention_mask'].squeeze(1).to(device)  
        idx += 1
        if idx > 10:
            break

        with torch.no_grad():  
            embeddings = embedding_model(input_ids=input_ids, attention_mask=attention_mask)  
            embeddings = embeddings.last_hidden_state.mean(dim=1)  # Mean pooling  
            original_embeddings = embeddings
            if quantization_module is not None:
                embeddings = quantization_module(embeddings, binary=True)  
                
        # Normalize embeddings to unit length
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        
        # Calculate similarity matrix through dot product of normalized embeddings
        # Shape: (batch_size x batch_size)
        similarity_matrix = torch.matmul(embeddings_normalized, embeddings_normalized.t())
        
        original_embeddings_normalized = F.normalize(original_embeddings, p=2, dim=1)
        original_similarity_matrix = torch.matmul(original_embeddings_normalized, original_embeddings_normalized.t())

        # flatten the similarity matrices
        similarity_matrix = similarity_matrix.flatten()
        original_similarity_matrix = original_similarity_matrix.flatten()

        loss = F.mse_loss(similarity_matrix, original_similarity_matrix)
        # print(f"Loss: {loss.item()}")
        # calculate the spearman correlation between the original similarity matrix and the quantized similarity matrix
        spearman_corr = spearmanr(similarity_matrix.cpu().numpy(), original_similarity_matrix.cpu().numpy())
        pearson_corr = pearsonr(similarity_matrix.cpu().numpy(), original_similarity_matrix.cpu().numpy())
        # print(f"Spearman Correlation: {spearman_corr.correlation}")
        average_loss += loss.item()
        average_spearman_corr += spearman_corr.correlation
        average_pearson_corr += pearson_corr.correlation
    return {
        'loss': average_loss / min(idx, len(dataloader)),
        'spearman_correlation': average_spearman_corr / min(idx, len(dataloader)),
        'pearson_correlation': average_pearson_corr / min(idx, len(dataloader))
    }



def main():
    embedding_model = AutoModel.from_pretrained(base_model_name)  
    embedding_dim = embedding_model.config.hidden_size  # e.g., 384  
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    dataloader = get_dataloader(base_model_name, batch_size)
    
    embedding_model.to(device)
  
    task_results = {}
    embedding_dim = embedding_model.config.hidden_size  # e.g., 384
    
    # 1. Original Model
    print("  Evaluating Original Model...")
    # print(f"[DEBUG] Created model type: {type(original_model)}")  # Add this debug print
    results_original = check_loss(embedding_model, None, dataloader)
    
    task_results['Original'] = results_original

    # 2. Stage1 Untrained
    print("  Evaluating Stage1 Untrained...")
    quantization_module_stage1_zero = QuantizationModuleStage1(embedding_dim)
    quantization_module_stage1_zero.thresholds.data.fill_(0.0)
    quantization_module_stage1_zero.to(device)
    
    results_stage1_zero = check_loss(embedding_model, quantization_module_stage1_zero, dataloader)
    task_results['QuantStage1_Untrained'] = results_stage1_zero

    if 'stage1' in test_modules:
        # 3. Stage1 Trained
        print("  Evaluating Stage1 Trained...")
        quantization_module_stage1_trained = QuantizationModuleStage1(embedding_dim)
        quantization_module_stage1_trained.load_state_dict(
            torch.load(f'saved_models/{save_dirs[0]}/quantization_stage1.pth', map_location=device, weights_only=False)
        )
        # print(quantization_module_stage1_trained.thresholds)
        quantization_module_stage1_trained.to(device)
        quantization_module_stage1_trained.eval()
        
        results_stage1_trained = check_loss(embedding_model, quantization_module_stage1_trained, dataloader)

        task_results['QuantStage1_Trained'] = results_stage1_trained
    
    if 'stage1.1' in test_modules:
        # 3.5 Stage1.1 Trained
        print("  Evaluating Stage1.1 Trained...")
        quantization_module_stage1_1_trained = QuantizationModuleStage1WithScales(embedding_dim)
        quantization_module_stage1_1_trained.load_state_dict(
            torch.load(f'saved_models/{save_dirs[1]}/quantization_stage1_with_scales.pth', map_location=device, weights_only=False)
        )
        quantization_module_stage1_1_trained.to(device)
        quantization_module_stage1_1_trained.eval()
        
        results_stage1_1_trained = check_loss(embedding_model, quantization_module_stage1_1_trained, dataloader)
        task_results['QuantStage1.1_Trained'] = results_stage1_1_trained
    
    if 'stage2' in test_modules:
        # 3.5 Stage2 Untrained  
        print("  Evaluating Stage2 Untrained...")
        quantization_module_stage2_untrained = QuantizationModuleStage2(embedding_dim)
        quantization_module_stage2_untrained.thresholds_first_half.data.fill_(0.0)
        quantization_module_stage2_untrained.thresholds_second_half.data.fill_(0.0)
        quantization_module_stage2_untrained.to(device)
        quantization_module_stage2_untrained.eval()
        
        results_stage2_untrained = check_loss(embedding_model, quantization_module_stage2_untrained, dataloader)
        task_results['QuantStage2_Untrained'] = results_stage2_untrained

        # 4. Stage2 Trained
        print("  Evaluating Stage2 Trained...")
        quantization_module_stage2_trained = QuantizationModuleStage2(embedding_dim)
        quantization_module_stage2_trained.load_state_dict(
            torch.load(f'saved_models/{save_dirs[2]}/quantization_stage2.pth', map_location=device, weights_only=False)
        )
        
        quantization_module_stage2_trained.to(device)
        quantization_module_stage2_trained.eval()
        
        results_stage2_trained = check_loss(embedding_model, quantization_module_stage2_trained, dataloader)
        task_results['QuantStage2_Trained'] = results_stage2_trained
    
    if 'stage3' in test_modules:
        # 5. Stage3 Trained
        print("  Evaluating Stage3 Trained...")
        quantization_module_stage3_trained = ImprovedQuantizationModule(embedding_dim)
        quantization_module_stage3_trained.load_state_dict(
            torch.load(f'saved_models/{save_dirs[3]}/improved_quantization.pth', map_location=device, weights_only=False)
        )
        
        quantization_module_stage3_trained.to(device)
        quantization_module_stage3_trained.eval()
        
        results_stage3_trained = check_loss(embedding_model, quantization_module_stage3_trained, dataloader)
        
        task_results['QuantStage3_Trained'] = results_stage3_trained
        
    if 'OneBitTwoBit' in test_modules:
        # 6. OneBitTwoBit Trained
        print("  Evaluating OneBitTwoBit Trained...")
        quantization_module_one_bit_two_bit = QuantizationModuleOneBitTwoBit(embedding_dim)
        quantization_module_one_bit_two_bit.load_state_dict(
            torch.load(f'saved_models/{save_dirs[4]}/one_bit_two_bit_thresholds.pth', map_location=device, weights_only=False)
        )
        quantization_module_one_bit_two_bit.to(device)
        quantization_module_one_bit_two_bit.eval()
        
        results_one_bit_two_bit = check_loss(embedding_model, quantization_module_one_bit_two_bit, dataloader)
        
        task_results['OneBitTwoBit_Trained'] = results_one_bit_two_bit
        
    print(task_results)
    # task_results is a dictionary with the results for each task which themselves are dictionaries with the loss and spearman correlation
    # Lets show the results in a table
    # Convert nested dictionary to DataFrame and format for display
    # Convert nested dictionary to DataFrame
    data = []
    for model, results in task_results.items():
        if isinstance(results, dict):
            data.append({
                'Model': model,
                'Loss': results['loss'],
                'Spearman Correlation': results['spearman_correlation']
            })
    
    results_df = pd.DataFrame(data)
    
    print("\nEvaluation Results:")
    print(results_df.round(4).to_markdown(index=False))

if __name__ == '__main__':
    main()
