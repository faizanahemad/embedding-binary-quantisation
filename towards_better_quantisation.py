from unittest import result
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel  
from torch.utils.data import DataLoader, Dataset  
import numpy as np  
from config import base_model_name, reg_strength, num_epochs, batch_size, lr, init_std, temperature, max_grad_norm

from dataset import CombinedSimilarityDataset


import os  
from datetime import datetime  
from common import *
from basic_quantization_modules import QuantizationModuleStage1

# Stage 1: Implement per-dimension thresholds  
  
class QuantizationModuleStage1WithScales(QuantizationModuleStage1):  
    """  
    Quantization Module for Stage 1: Per-dimension thresholds.  
  
    Attributes:  
        thresholds (nn.Parameter): Learnable thresholds of shape (embedding_dim,)  
    """  
    def __init__(self, embedding_dim, initial_thresholds=None):  
        super(QuantizationModuleStage1WithScales, self).__init__(embedding_dim)  
        if initial_thresholds is not None:  
            self.thresholds = nn.Parameter(initial_thresholds)  
        else:  
            # Initialize thresholds to zero  
            self.thresholds = nn.Parameter(torch.zeros(embedding_dim) + torch.randn(embedding_dim) * init_std)  
            
        self.scales = nn.Parameter(torch.ones(embedding_dim) + torch.randn(embedding_dim) * init_std)
        self.scales.requires_grad = True
        self.original_thresholds = self.thresholds.data.clone().detach()
        self.original_thresholds.requires_grad = False
        self.temperature = temperature
        
  
    def forward(self, embeddings, binary=False):  
        """  
        Forward pass to compute the quantized embeddings.  
  
        Args:  
            embeddings (torch.Tensor): Original embeddings of shape (batch_size, embedding_dim)  
  
        Returns:  
            quantized_embeddings (torch.Tensor): Quantized embeddings of shape (batch_size, embedding_dim)  
        """  
        
        
        scaled_embeddings = embeddings * torch.abs(self.scales)
        v = scaled_embeddings - self.thresholds
        
        quantized_embeddings = torch.sigmoid(self.temperature * v)  
  
        return quantized_embeddings if not binary else (quantized_embeddings > 0.5).float()


# Training function  
  
def train_quantization_stage1_with_scales(embedding_model, quantization_module, dataloader, num_epochs=5):  
    """  
    Train the quantization module for Stage 1.  
  
    Args:  
        embedding_model (nn.Module): Frozen embedding model  
        quantization_module (QuantizationModuleStage1): Quantization module  
        dataloader (DataLoader): DataLoader for the dataset  
        num_epochs (int): Number of training epochs  
    """  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    # Determine the device to use (GPU if available)  
    
    embedding_model.to(device)  
    quantization_module.to(device)
    
    # Initialize thresholds using sample embeddings
    with torch.no_grad():
        i = 0
        for batch in tqdm(dataloader, desc="Initializing Thresholds", total=len(dataloader)):
            input_ids = batch['input_ids'].to(device)  
            attention_mask = batch['attention_mask'].to(device)  
            embeddings = embedding_model(input_ids=input_ids, attention_mask=attention_mask)  
            embeddings = mean_pool_and_L2_normalize(embeddings, attention_mask)
            
            if i == 0:
                quantization_module.thresholds.data = 0.0001 * quantization_module.initialize_thresholds(embeddings)
                i += 1
            else:
                quantization_module.thresholds.data = 0.9999 * quantization_module.thresholds.data + \
                    0.0001 * quantization_module.initialize_thresholds(embeddings)

    print(f"[DEBUG] Thresholds after initialization: {quantization_module.thresholds}")

    # return quantization_module
    
    original_thresholds = quantization_module.thresholds.data.clone().detach()
    original_thresholds.requires_grad = False
    quantization_module.original_thresholds = original_thresholds
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, quantization_module.parameters()), lr=lr)  
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(dataloader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    embedding_model.eval()  
    quantization_module.train()  
  
    for epoch in range(num_epochs):  
        total_loss = 0.0  
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_ids = batch['input_ids'].squeeze(1).to(device)  # Remove extra dimension  
            attention_mask = batch['attention_mask'].squeeze(1).to(device)  
  
            with torch.no_grad():  
                embeddings = embedding_model(input_ids=input_ids, attention_mask=attention_mask)  
                embeddings = mean_pool_and_L2_normalize(embeddings, attention_mask)
  
            quantized_embeddings = quantization_module(embeddings)  
            
            scale_reg = torch.norm(torch.abs(quantization_module.scales) - 1, 2)
            
  
            loss = similarity_preservation_loss(embeddings, quantized_embeddings)  + reg_strength * (torch.norm(quantization_module.thresholds - original_thresholds, 2) + scale_reg) + matching_preserving_loss(embeddings, quantized_embeddings) + rank_preserving_loss(embeddings, quantized_embeddings)
  
            optimizer.zero_grad()  
            loss.backward()  
            # Add gradient clipping before optimizer step
            torch.nn.utils.clip_grad_norm_(quantization_module.parameters(), max_grad_norm)
            
            optimizer.step()  
            scheduler.step()
  
            total_loss += loss.item()  
  
        avg_loss = total_loss / len(dataloader)  
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')  
    return quantization_module
    save_dir = create_save_directory()  
    save_quantization_module(quantization_module, save_dir, 'quantization_stage1')  
    
  
