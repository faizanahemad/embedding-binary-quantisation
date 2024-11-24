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

# Stage 1: Implement per-dimension thresholds  
  
class QuantizationModuleStage1WithScales(nn.Module):  
    """  
    Quantization Module for Stage 1: Per-dimension thresholds.  
  
    Attributes:  
        thresholds (nn.Parameter): Learnable thresholds of shape (embedding_dim,)  
    """  
    def __init__(self, embedding_dim, initial_thresholds=None):  
        super(QuantizationModuleStage1WithScales, self).__init__()  
        if initial_thresholds is not None:  
            self.thresholds = nn.Parameter(initial_thresholds)  
        else:  
            # Initialize thresholds to zero  
            self.thresholds = nn.Parameter(torch.zeros(embedding_dim) + torch.randn(embedding_dim) * 0.1)  
            
        self.scales = nn.Parameter(torch.ones(embedding_dim))
        self.temperature = 10
        
  
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
    optimizer = optim.Adam(quantization_module.parameters(), lr=0.001)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    embedding_model.eval()  
    quantization_module.train()  
  
    for epoch in range(num_epochs):  
        total_loss = 0.0  
        for batch in dataloader:  
            input_ids = batch['input_ids'].squeeze(1).to(device)  # Remove extra dimension  
            attention_mask = batch['attention_mask'].squeeze(1).to(device)  
  
            with torch.no_grad():  
                embeddings = embedding_model(input_ids=input_ids, attention_mask=attention_mask)  
                embeddings = embeddings.last_hidden_state.mean(dim=1)  # Mean pooling  
  
            quantized_embeddings = quantization_module(embeddings)  
            
            scale_reg = torch.norm(torch.abs(quantization_module.scales) - 1, 2)
            
  
            loss = similarity_preservation_loss(embeddings, quantized_embeddings)  + reg_strength * (torch.norm(quantization_module.thresholds, 2) + scale_reg)
  
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
  
            total_loss += loss.item()  
  
        avg_loss = total_loss / len(dataloader)  
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')  
    return quantization_module
    save_dir = create_save_directory()  
    save_quantization_module(quantization_module, save_dir, 'quantization_stage1')  
    
  
