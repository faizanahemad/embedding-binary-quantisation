from unittest import result
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
from transformers import AutoTokenizer, AutoModel  
from torch.utils.data import DataLoader, Dataset  
import numpy as np  
from config import base_model_name, reg_strength, num_epochs, batch_size, train_modules, save_dirs

from dataset import CombinedSimilarityDataset


import os  
from datetime import datetime  
from common import create_save_directory, save_quantization_module, similarity_preservation_loss
from improved_quantisation_module import ImprovedQuantizationModule, train_improved_quantization

from basic_quantization_modules import QuantizationModuleStage1, QuantizationModuleStage2, train_quantization_stage1, train_quantization_stage2
from towards_better_quantisation import QuantizationModuleStage1WithScales, train_quantization_stage1_with_scales
  
def main():  
    # Load the frozen embedding model  
    embedding_model = AutoModel.from_pretrained(base_model_name)  
    embedding_dim = embedding_model.config.hidden_size  # e.g., 384  
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
  
    # Freeze the embedding model parameters  
    for param in embedding_model.parameters():  
        param.requires_grad = False  
  
    # Prepare the dataset and dataloader  
    # texts = ["This is a sample sentence.", "Another example text.", "More data for training."]  
    # dataset = ExampleDataset(texts)  
    
    dataset = CombinedSimilarityDataset(tokenizer, max_length=128, max_samples_per_dataset=10000)
    # Create a sampler that keeps pairs together while shuffling between pairs
    indices = list(range(0, len(dataset), 2))  # Get indices of first element of each pair
    shuffled_pair_indices = torch.randperm(len(indices)).tolist()
    final_indices = []
    for idx in shuffled_pair_indices:
        pair_start = indices[idx]
        final_indices.extend([pair_start, pair_start + 1])  # Keep pairs together
        
    sampler = torch.utils.data.sampler.SequentialSampler(final_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    embedding_model.to(device)
    
    if 'stage1' in train_modules:
        # Stage 1: Train per-dimension thresholds  
        quantization_module_stage1 = QuantizationModuleStage1(embedding_dim)  
        quantization_module_stage1.to(device)
        quantization_module_stage1 = train_quantization_stage1(embedding_model, quantization_module_stage1, dataloader, num_epochs=num_epochs)  
    
  
    if 'stage2' in train_modules:
        # Stage 2: Train with combined dimensions  
        quantization_module_stage2 = QuantizationModuleStage2(embedding_dim)  
        quantization_module_stage2.to(device)
        quantization_module_stage2 = train_quantization_stage2(embedding_model, quantization_module_stage2, dataloader, num_epochs=num_epochs)  
    
    if 'stage3' in train_modules:
        # Stage 3: Train with adaptive thresholds, importance scoring, and progressive dimension pruning
        quantization_module_stage3 = ImprovedQuantizationModule(embedding_dim)
        quantization_module_stage3.to(device)
        quantization_module_stage3 = train_improved_quantization(embedding_model, quantization_module_stage3, dataloader, num_epochs=num_epochs)
        
    if 'stage1.1' in train_modules:
        # Stage 4: Train with adaptive thresholds and scaling param
        quantization_module_stage1_1 = QuantizationModuleStage1WithScales(embedding_dim)
        quantization_module_stage1_1.to(device)
        quantization_module_stage1_1 = train_quantization_stage1_with_scales(embedding_model, quantization_module_stage1_1, dataloader, num_epochs=num_epochs)
    
    save_dir = create_save_directory()  
    print(f'Saving models to {save_dir}')
    if 'stage1' in train_modules:
        save_quantization_module(quantization_module_stage1, save_dir, 'quantization_stage1')  
    if 'stage2' in train_modules:
        save_quantization_module(quantization_module_stage2, save_dir, 'quantization_stage2')  
    if 'stage3' in train_modules:
        save_quantization_module(quantization_module_stage3, save_dir, 'improved_quantization')
    if 'stage1.1' in train_modules:
        save_quantization_module(quantization_module_stage1_1, save_dir, 'quantization_stage1_with_scales')
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