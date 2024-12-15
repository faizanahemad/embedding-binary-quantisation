from unittest import result
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
from transformers import AutoTokenizer, AutoModel  
from torch.utils.data import DataLoader, Dataset  
import numpy as np  
from MatryoshkaModel.matryoshka_2bit_model import train_matryoshka_model, MatryoshkaEmbeddingModel, MatryoshkaTransformer
from config import base_model_name, reg_strength, num_epochs, batch_size, train_modules, save_dirs

from dataset import CombinedSimilarityDataset
from common import SentenceTransformerEmbeddingCaller


import os  
from datetime import datetime  
from common import create_save_directory, save_quantization_module, similarity_preservation_loss, get_dataloader
from improved_quantisation_module import ImprovedQuantizationModule, train_improved_quantization

from basic_quantization_modules import QuantizationModuleStage1, QuantizationModuleStage2, train_quantization_stage1, train_quantization_stage2
from towards_better_quantisation import QuantizationModuleStage1WithScales, train_quantization_stage1_with_scales
from two_bit_one_bit_dual_quantization_module import QuantizationModuleOneBitTwoBit, train_quantization_module_one_bit_two_bit
  
def main():  
    # Load the frozen embedding model  
    embedding_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)  
    embedding_dim = embedding_model.config.hidden_size  # e.g., 384  
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
  
    # Freeze the embedding model parameters  
    for param in embedding_model.parameters():  
        param.requires_grad = False  
  
    # Prepare the dataset and dataloader  
    # texts = ["This is a sample sentence.", "Another example text.", "More data for training."]  
    # dataset = ExampleDataset(texts)  
    
    dataloader = get_dataloader(base_model_name, batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    embedding_model.to(device)
    save_dir = create_save_directory()
    
    if 'stage1' in train_modules:
        # Stage 1: Train per-dimension thresholds  
        quantization_module_stage1 = QuantizationModuleStage1(embedding_dim)  
        quantization_module_stage1.to(device)
        quantization_module_stage1 = train_quantization_stage1(embedding_model, quantization_module_stage1, dataloader, num_epochs=num_epochs)  
        
        # Save thresholds to a JSON file
        import json
        thresholds_path = os.path.join(save_dir, 'stage1_thresholds.json')
        thresholds_data = {
            'thresholds': quantization_module_stage1.thresholds.detach().cpu().numpy().tolist(),
            'original_thresholds': quantization_module_stage1.original_thresholds.detach().cpu().numpy().tolist()
        }
        with open(thresholds_path, 'w') as f:
            json.dump(thresholds_data, f, indent=4)
        print(f"Saved thresholds to {thresholds_path}")
    
  
    if 'stage2' in train_modules:
        # Stage 2: Train with combined dimensions  
        quantization_module_stage2 = QuantizationModuleStage2(embedding_dim)  
        quantization_module_stage2.to(device)
        quantization_module_stage2 = train_quantization_stage2(embedding_model, quantization_module_stage2, dataloader, num_epochs=num_epochs)  
        
        # Save thresholds to a JSON file, self.thresholds_second_half, self.thresholds_first_half
        import json
        thresholds_path = os.path.join(save_dir, 'stage2_thresholds.json')
        thresholds_data = {
            'thresholds_second_half': quantization_module_stage2.thresholds_second_half.detach().cpu().numpy().tolist(),
            'thresholds_first_half': quantization_module_stage2.thresholds_first_half.detach().cpu().numpy().tolist(),
            'original_thresholds_second_half': quantization_module_stage2.original_thresholds_second_half.detach().cpu().numpy().tolist(),
            'original_thresholds_first_half': quantization_module_stage2.original_thresholds_first_half.detach().cpu().numpy().tolist()
        }
        with open(thresholds_path, 'w') as f:
            json.dump(thresholds_data, f, indent=4)
        print(f"Saved thresholds to {thresholds_path}")
    
    if 'stage3' in train_modules:
        # Stage 3: Train with adaptive thresholds, importance scoring, and progressive dimension pruning
        quantization_module_stage3 = ImprovedQuantizationModule(embedding_dim)
        quantization_module_stage3.to(device)
        quantization_module_stage3 = train_improved_quantization(embedding_model, quantization_module_stage3, dataloader, num_epochs=num_epochs)
        
        # Save thresholds to a JSON file, self.thresholds, self.scales
        import json
        thresholds_path = os.path.join(save_dir, 'stage3_thresholds.json')
        thresholds_data = {
            'thresholds': quantization_module_stage3.thresholds.detach().cpu().numpy().tolist(),
            'scales': quantization_module_stage3.scales.detach().cpu().numpy().tolist(),
            'original_thresholds': quantization_module_stage3.original_thresholds.detach().cpu().numpy().tolist(),
            
        }
        with open(thresholds_path, 'w') as f:
            json.dump(thresholds_data, f, indent=4)
        print(f"Saved thresholds to {thresholds_path}")
        
    if 'stage1.1' in train_modules:
        # Stage 4: Train with adaptive thresholds and scaling param
        quantization_module_stage1_1 = QuantizationModuleStage1WithScales(embedding_dim)
        quantization_module_stage1_1.to(device)
        quantization_module_stage1_1 = train_quantization_stage1_with_scales(embedding_model, quantization_module_stage1_1, dataloader, num_epochs=num_epochs)
        
        # Save thresholds to a JSON file, self.thresholds, self.scales
        import json
        thresholds_path = os.path.join(save_dir, 'stage1_1_thresholds.json')
        thresholds_data = {
            'thresholds': quantization_module_stage1_1.thresholds.detach().cpu().numpy().tolist(),
            'scales': quantization_module_stage1_1.scales.detach().cpu().numpy().tolist(),
            'original_thresholds': quantization_module_stage1_1.original_thresholds.detach().cpu().numpy().tolist(),
            
        }
        with open(thresholds_path, 'w') as f:
            json.dump(thresholds_data, f, indent=4)
        print(f"Saved thresholds to {thresholds_path}")
        
    if 'OneBitTwoBit' in train_modules:
        quantization_module_one_bit_two_bit = QuantizationModuleOneBitTwoBit(embedding_dim)
        quantization_module_one_bit_two_bit.to(device)
        quantization_module_one_bit_two_bit = train_quantization_module_one_bit_two_bit(embedding_model, quantization_module_one_bit_two_bit, dataloader, num_epochs=num_epochs)
        
        # Save thresholds to a JSON file, self.thresholds, self.scales
        import json
        thresholds_path = os.path.join(save_dir, 'one_bit_two_bit_thresholds.json')
        thresholds_data = {
            'thresholds': quantization_module_one_bit_two_bit.thresholds.detach().cpu().numpy().tolist()
        }
        with open(thresholds_path, 'w') as f:
            json.dump(thresholds_data, f, indent=4)
        print(f"Saved thresholds to {thresholds_path}")
        
    if 'Matryoshka' in train_modules:
        embedding_model = SentenceTransformerEmbeddingCaller(base_model_name)
        matryoshka_model = MatryoshkaEmbeddingModel(embedding_model, dimension_levels=[embedding_dim//4, embedding_dim//2, embedding_dim], train_binary=False, train_two_bit=False, expand_two_bit_to_three_bits=False)
        matryoshka_model.to(device)
        matryoshka_model = train_matryoshka_model(matryoshka_model, dataloader, num_epochs=num_epochs)
        matryoshka_model.save(os.path.join(save_dir, 'matryoshka_model.pth'))
        
    
      
    print(f'Saving models to {save_dir}')
    if 'stage1' in train_modules:
        save_quantization_module(quantization_module_stage1, save_dir, 'quantization_stage1')  
    if 'stage2' in train_modules:
        save_quantization_module(quantization_module_stage2, save_dir, 'quantization_stage2')  
    if 'stage3' in train_modules:
        save_quantization_module(quantization_module_stage3, save_dir, 'improved_quantization')
    if 'stage1.1' in train_modules:
        save_quantization_module(quantization_module_stage1_1, save_dir, 'quantization_stage1_with_scales')
    if 'OneBitTwoBit' in train_modules:
        save_quantization_module(quantization_module_one_bit_two_bit, save_dir, 'one_bit_two_bit_thresholds')
    
        
    # Inference example  
    return
    embedding_model.eval()  
    quantization_module_stage2.eval()  
  
    # Example input  
    input_text = "New input text for inference."  
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)  
    encoded_input = tokenizer(input_text, return_tensors='pt')  
  
    with torch.no_grad():  
        embedding = embedding_model(**encoded_input.to(device))  
        embedding = embedding.last_hidden_state.mean(dim=1)  
  
        quantized_embedding = quantization_module_stage2(embedding)  
  
        # Apply hard thresholding for binary output  
        binary_embedding = (quantized_embedding > 0.5).int()  
        print("Binary Embedding:", binary_embedding)  
        
        

  
if __name__ == '__main__':  
    main()  
