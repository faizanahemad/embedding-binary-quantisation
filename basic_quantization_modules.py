from unittest import result
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
from transformers import AutoTokenizer, AutoModel  
from torch.utils.data import DataLoader, Dataset  
import numpy as np  
from config import base_model_name, reg_strength, num_epochs, batch_size, lr, init_std, temperature
from tqdm import tqdm

from dataset import CombinedSimilarityDataset


import os  
from datetime import datetime  
from common import create_save_directory, save_quantization_module, similarity_preservation_loss, matching_preserving_loss, rank_preserving_loss
from improved_quantisation_module import ImprovedQuantizationModule, train_improved_quantization

# Stage 1: Implement per-dimension thresholds  
  
class QuantizationModuleStage1(nn.Module):  
    """  
    Quantization Module for Stage 1: Per-dimension thresholds.  
  
    Attributes:  
        thresholds (nn.Parameter): Learnable thresholds of shape (embedding_dim,)  
    """  
    def __init__(self, embedding_dim, initial_thresholds=None):  
        super(QuantizationModuleStage1, self).__init__()  
        if initial_thresholds is not None:  
            self.thresholds = nn.Parameter(initial_thresholds)  
        else:  
            # Initialize thresholds to zero  
            self.thresholds = nn.Parameter(torch.zeros(embedding_dim) + torch.randn(embedding_dim) * init_std)  
  
    def forward(self, embeddings, binary=False):  
        """  
        Forward pass to compute the quantized embeddings.  
  
        Args:  
            embeddings (torch.Tensor): Original embeddings of shape (batch_size, embedding_dim)  
  
        Returns:  
            quantized_embeddings (torch.Tensor): Quantized embeddings of shape (batch_size, embedding_dim)  
        """  
        # Compute pre-activations  
        v = embeddings - self.thresholds  # Broadcasting over batch size  
  
        # Apply sigmoid function as a soft thresholding  
        k = temperature  # Temperature parameter controlling the steepness  
        quantized_embeddings = torch.sigmoid(k * v)  
  
        return quantized_embeddings if not binary else (quantized_embeddings > 0.5).float()
  
# Stage 2: Extend to handle combined dimensions  

class QuantizationModuleStage2(nn.Module):
    def __init__(self, embedding_dim):
        """  
        Quantization Module for Stage 2:  
        - First K//2 dimensions: per-dimension thresholds  
        - Last K//2 dimensions: combine pairs of dimensions into one binary output  
    
        Attributes:  
            thresholds_first_half (nn.Parameter): Thresholds for first half dimensions (K//2,)  
            thresholds_second_half (nn.Parameter): Thresholds for second half dimensions (K//2, 2)  
        """  
        super(QuantizationModuleStage2, self).__init__()
        self.embedding_dim = embedding_dim
        self.half_dim = embedding_dim // 2

        # Thresholds for first half
        self.thresholds_first_half = nn.Parameter(torch.zeros(self.half_dim) + torch.randn(self.half_dim) * init_std)

        # Thresholds for second half (pairs of thresholds)
        self.thresholds_second_half = nn.Parameter(torch.zeros(self.half_dim // 2, 2) + torch.randn(self.half_dim // 2, 2) * init_std)

    def forward(self, embeddings, binary=False):
        """  
        Forward pass to compute the quantized embeddings.  
  
        Args:  
            embeddings (torch.Tensor): Original embeddings of shape (batch_size, embedding_dim)  
  
        Returns:  
            quantized_embeddings (torch.Tensor): Quantized embeddings of shape (batch_size, embedding_dim_new)  
        """ 
        # First half processing remains the same
        embeddings_first_half = embeddings[:, :self.half_dim]
        v_first = embeddings_first_half - self.thresholds_first_half
        k = temperature
        quantized_first_half = torch.sigmoid(k * v_first)

        # Second half: properly reshape for pairs
        embeddings_second_half = embeddings[:, self.half_dim:]
        # Reshape to (batch_size, num_pairs, 2)
        embeddings_pairs = embeddings_second_half.reshape(-1, self.half_dim // 2, 2)
        
        # Reshape thresholds to match embedding pairs (add batch dimension)
        thresholds_pairs = self.thresholds_second_half.unsqueeze(0)  # Shape: (1, num_pairs, 2)

        # Rest of the processing remains the same
        v_pairs = embeddings_pairs - thresholds_pairs
        s_pairs = torch.sigmoid(k * v_pairs)
        combined_outputs = 1 - (1 - s_pairs[:, :, 0]) * (1 - s_pairs[:, :, 1])

        quantized_embeddings = torch.cat([quantized_first_half, combined_outputs], dim=1)
        return quantized_embeddings if not binary else (quantized_embeddings > 0.5).float()
    
    

# Loss function  
  

# Training function  
  
def train_quantization_stage1(embedding_model, quantization_module, dataloader, num_epochs=5):  
    """  
    Train the quantization module for Stage 1.  
  
    Args:  
        embedding_model (nn.Module): Frozen embedding model  
        quantization_module (QuantizationModuleStage1): Quantization module  
        dataloader (DataLoader): DataLoader for the dataset  
        num_epochs (int): Number of training epochs  
    """  
    optimizer = optim.Adam(quantization_module.parameters(), lr=lr)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(dataloader))
  
    embedding_model.eval()  
    quantization_module.train()  
  
    for epoch in range(num_epochs):  
        total_loss = 0.0  
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):  
            input_ids = batch['input_ids'].squeeze(1).to(device)  # Remove extra dimension  
            attention_mask = batch['attention_mask'].squeeze(1).to(device)  
  
            with torch.no_grad():  
                embeddings = embedding_model(input_ids=input_ids, attention_mask=attention_mask)  
                embeddings = embeddings.last_hidden_state.mean(dim=1)  # Mean pooling  
  
            quantized_embeddings = quantization_module(embeddings)  
  
            loss = similarity_preservation_loss(embeddings, quantized_embeddings)  + reg_strength * torch.norm(quantization_module.thresholds, 2) + matching_preserving_loss(embeddings, quantized_embeddings) + rank_preserving_loss(embeddings, quantized_embeddings)
  
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
            scheduler.step()
  
            total_loss += loss.item()  
  
        avg_loss = total_loss / len(dataloader)  
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')  
    return quantization_module
    save_dir = create_save_directory()  
    save_quantization_module(quantization_module, save_dir, 'quantization_stage1')  
    
  
def train_quantization_stage2(embedding_model, quantization_module, dataloader, num_epochs=5):  
    """  
    Train the quantization module for Stage 2.  
  
    Args:  
        embedding_model (nn.Module): Frozen embedding model  
        quantization_module (QuantizationModuleStage2): Quantization module  
        dataloader (DataLoader): DataLoader for the dataset  
        num_epochs (int): Number of training epochs  
    """  
    optimizer = optim.Adam(quantization_module.parameters(), lr=lr)  
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(dataloader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_model.eval()  
    quantization_module.train()  
  
    for epoch in range(num_epochs):  
        total_loss = 0.0  
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):  
            input_ids = batch['input_ids'].squeeze(1).to(device)  
            attention_mask = batch['attention_mask'].squeeze(1).to(device)  
  
            with torch.no_grad():  
                embeddings = embedding_model(input_ids=input_ids, attention_mask=attention_mask)  
                embeddings = embeddings.last_hidden_state.mean(dim=1)  
  
            quantized_embeddings = quantization_module(embeddings)  
  
            loss = similarity_preservation_loss(embeddings, quantized_embeddings)  + reg_strength * torch.norm(quantization_module.thresholds_first_half, 2) + reg_strength * torch.norm(quantization_module.thresholds_second_half, 2) + matching_preserving_loss(embeddings, quantized_embeddings) + rank_preserving_loss(embeddings, quantized_embeddings)
  
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
            scheduler.step()
            total_loss += loss.item()  
  
        avg_loss = total_loss / len(dataloader)  
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')  
    return quantization_module
    save_dir = create_save_directory()  
    save_quantization_module(quantization_module, save_dir, 'quantization_stage2')  
