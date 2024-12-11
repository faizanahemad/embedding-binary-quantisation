import json
from unittest import result
from sympy import Permanent, trailing
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
from transformers import AutoTokenizer, AutoModel  
from torch.utils.data import DataLoader, Dataset  
import numpy as np  
from config import base_model_name, reg_strength, num_epochs, batch_size, lr, init_std, temperature, max_grad_norm
from tqdm import tqdm

from dataset import CombinedSimilarityDataset


import os  
from datetime import datetime  
from common import *

from basic_quantization_modules import QuantizationModuleStage1
class ImprovedQuantizationModule(QuantizationModuleStage1):
    """A sophisticated embedding quantization module that combines ideas from neural network quantization,
    pruning literature, and information theory to perform adaptive binary quantization and dimension pruning.
    
    Key Components:
    1. Adaptive Quantization:
        - Uses learnable per-dimension thresholds and scaling factors
        - Incorporates second-order information (Hessian approximation) for optimal quantization points
        - Employs temperature-based soft quantization during training for better gradient flow
        
    2. Intelligent Dimension Pruning:
        - Utilizes gradient and magnitude-based importance scoring
        - Implements progressive soft pruning during training
        - Maintains importance scores that capture both local and temporal information
        
    Mathematical Formulation:
    For an input embedding vector x ∈ ℝᵈ, the quantization process is:
    1. Scaling: x̂ = x ⊙ s, where s are learnable scales
    2. Threshold Adjustment: t̂ = t ⊙ √(diag(H) + ε), where H is the Hessian approximation
    3. Soft Quantization: q = σ((x̂ - t̂)/τ), where τ is temperature and σ is sigmoid
    4. Importance Weighting: y = q ⊙ σ(α/τ), where α are importance scores
    5. Binary Output: b = 1[y > 0.5] during inference
    
    Importance Score Computation:
    α = λ₁|∇x| + λ₂|x| + λ₃α_{t-1}, where:
    - |∇x| captures gradient-based importance
    - |x| captures magnitude-based importance
    - α_{t-1} is the historical importance
    
    Training Objectives:
    L = L_sim + L_reg + L_entropy, where:
    - L_sim: Similarity preservation loss between original and quantized embeddings
    - L_reg: L2 regularization weighted by importance scores
    - L_entropy: Binary entropy loss to encourage discrete outputs
    
    Theoretical Foundations:
    1. Second-order quantization from GPTQ (Frantar et al., 2023)
    2. Gradient-based pruning from "Movement Pruning" (Sanh et al., 2020)
    3. Temperature-based training from "Straight-Through Estimator" (Bengio et al., 2013)
    4. Importance scoring from "What's Hidden in a Randomly Weighted Neural Network?" (Ramanujan et al., 2020)
    
    Advantages over Simple Threshold-Based Quantization:
    1. Better Quantization Points:
        - Adapts to the local geometry of the loss surface using Hessian information
        - Learns optimal scaling factors per dimension
        - Smoother training through temperature annealing
        
    2. Informed Dimension Reduction:
        - Uses both gradient and magnitude signals for importance
        - Progressive pruning allows recovery from early mistakes
        - Maintains temporal consistency in importance scoring
        
    3. Enhanced Training Stability:
        - Soft quantization enables better gradient flow
        - Multiple loss components guide towards desired properties
        - Regularization prevents overfitting to training data
    
    Parameters:
        embedding_dim (int): Dimension of input embeddings
        
    Attributes:
        thresholds (nn.Parameter): Learnable quantization thresholds per dimension
        scales (nn.Parameter): Learnable scaling factors per dimension
        importance_scores (nn.Parameter): Dimension importance scores
        temperature (float): Annealing temperature for soft quantization/pruning
    """
    def __init__(self, embedding_dim, dimension_importance_decay=0.01):
        """
        Increase dimension_importance_decay to make earlier dimensions harder to prune and later dimensions easier to prune.
        Decrease dimension_importance_decay to make pruning more evenly distributed across dimensions.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Learnable parameters for sophisticated quantization
        self.thresholds = nn.Parameter(torch.zeros(embedding_dim) + torch.randn(embedding_dim) * init_std)
        self.scales = nn.Parameter(torch.ones(embedding_dim))
        self.register_buffer('pruning_mask', torch.ones(embedding_dim, dtype=torch.bool))  # False for pruned dims
        
        
        # Importance scores for dimension pruning
        self.importance_scores = nn.Parameter(torch.ones(embedding_dim) + torch.randn(embedding_dim) * init_std)
        
        # Create dimension-based importance bias
        # Earlier dimensions get lower thresholds (easier to keep)
        # Later dimensions get higher thresholds (easier to prune)
        position_importance = torch.exp(-dimension_importance_decay * 
                                     torch.arange(embedding_dim, dtype=torch.float32))
        self.register_buffer('position_importance', position_importance)
        
        self.register_buffer('hessian_diag', torch.ones(embedding_dim))  # Add this

        
        # Temperature parameter for soft pruning
        self.temperature = 1/temperature
        self.importance_momentum = 0.99
        
        self.original_thresholds = self.thresholds.clone().detach()
        self.original_thresholds.requires_grad = False
    
    def compute_importance_scores(self, embeddings):
        """Compute importance scores based on gradient and magnitude information"""
        # Gradient-based importance
        grad_importance = torch.abs(embeddings.grad).mean(0) if embeddings.grad is not None else torch.zeros_like(self.importance_scores)
        
        # Magnitude-based importance
        magnitude_importance = torch.abs(embeddings).mean(0)
        
        # Combine both metrics
        return grad_importance + magnitude_importance

    def forward(self, embeddings, binary=False):
        """
        Forward pass with sophisticated quantization and pruning.
        
        Args:
            embeddings (torch.Tensor): Input embeddings [batch_size, embedding_dim]
            binary (bool): Whether to return binary outputs
            training (bool): Whether in training mode
        
        Returns:
            torch.Tensor: Quantized embeddings
        """
        # Apply learned scaling
        # self.pruning_mask = self.pruning_mask.to(embeddings.device)
        # print(f'pruning mask: {self.pruning_mask.device}, scales: {self.scales.device}, thresholds: {self.thresholds.device}')
        effective_scales = self.scales * self.pruning_mask
        scaled_embeddings = embeddings * torch.abs(effective_scales)
        
        # Compute quantization thresholds using second-order information
        if self.training:
            scaled_embeddings.retain_grad()
            # Approximate Hessian diagonal
            # Wrong way to compute hessian diagonal, needs to be updated after backward pass
            # grad_sq = torch.pow(scaled_embeddings.grad, 2) if scaled_embeddings.grad is not None else torch.ones_like(scaled_embeddings)
            # hessian_diag = grad_sq.mean(0)
            
            # print(f'hessian diag: {hessian_diag.device}, values: {hessian_diag}')
            
            # Adjust thresholds based on Hessian
            adjusted_thresholds = self.thresholds * torch.sqrt(self.hessian_diag + 1e-6)
        else:
            adjusted_thresholds = self.thresholds

        # Soft quantization with temperature
        v = scaled_embeddings - adjusted_thresholds
        quantized = torch.sigmoid(v / self.temperature)
        
        # Apply dimension pruning using importance scores
        if self.training:
            # Update importance scores
            current_importance = self.compute_importance_scores(embeddings)
            self.importance_scores.data = self.importance_momentum * self.importance_scores.data + \
                                           (1 - self.importance_momentum) * current_importance
            
        # Soft masking based on importance scores
        mask = torch.sigmoid(self.importance_scores / self.temperature)
        quantized = quantized * mask.unsqueeze(0)
        
        if binary:
            return (quantized > 0.5).float()
        return quantized
    
    def update_hessian(self, scaled_embeddings):
        """Update Hessian diagonal approximation using gradients"""
        if scaled_embeddings.grad is not None:
            grad_sq = torch.pow(scaled_embeddings.grad, 2)
            new_hessian = grad_sq.mean(0)
            # Use exponential moving average for stability
            self.hessian_diag = 0.9 * self.hessian_diag + 0.1 * new_hessian.detach()

    
    
    def prune_dimensions(self, threshold=0.0):
        """
        Zero out scales for pruned dimensions instead of removing them
        Permanent prune dimensions with low importance.
        
        Prune dimensions with position-aware thresholding.
        Earlier dimensions have lower thresholds (harder to prune),
        Later dimensions have higher thresholds (easier to prune).
        """
        position_adjusted_threshold = threshold / self.position_importance
        new_mask = self.importance_scores >= position_adjusted_threshold
        new_mask = torch.ones_like(self.importance_scores, dtype=torch.bool)
        self.pruning_mask.data = new_mask
        # Zero out scales for pruned dimensions
        self.scales.data = self.scales.data * new_mask
        return new_mask
    
    
    
def train_improved_quantization(embedding_model, quantization_module, dataloader, num_epochs=5):
    """
    Train the improved quantization module with adaptive thresholds, importance scoring,
    and progressive dimension pruning.
    
    Args:
        embedding_model (nn.Module): Frozen embedding model
        quantization_module (ImprovedQuantizationModule): Quantization module
        dataloader (DataLoader): Dattrailingfor the dataset
        num_epochs (int): Number of training epochs
        
    Returns:
        quantization_module (ImprovedQuantizationModule): Trained quantization module
        training_stats (dict): Dictionary containing training statistics
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

    print(f"[DEBUG] Thresholds after initialization: \n{quantization_module.thresholds}")
    
    original_thresholds = quantization_module.thresholds.data.clone().detach()
    original_thresholds.requires_grad = False
    quantization_module.original_thresholds = original_thresholds
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, quantization_module.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(dataloader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize training statistics
    training_stats = {
        'epoch_losses': [],
        'importance_scores_history': [],
        'pruned_dimensions': [],
        'temperature_history': []
    }
    
    embedding_model.eval()
    quantization_module.train()
    
    initial_dimensions = quantization_module.embedding_dim
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_sim_loss = 0.0
        total_reg_loss = 0.0
        total_entropy_loss = 0.0
        
        for batch_idx, batch in tqdm(enumerate(dataloader), desc=f'Epoch {epoch+1}/{num_epochs}', total=len(dataloader)):
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            
            # Get embeddings with gradient tracking
            with torch.no_grad():
                model_output = embedding_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                embeddings = mean_pool_and_L2_normalize(embeddings, attention_mask)
            # Enable gradient tracking for importance scoring
            embeddings.requires_grad_(True)
            embeddings.retain_grad()  # Add this line
            
            # Forward pass through quantization module
            quantized_embeddings = quantization_module(embeddings)
            
            # Compute losses
            sim_loss = similarity_preservation_loss(embeddings, quantized_embeddings) + matching_preserving_loss(embeddings, quantized_embeddings) + rank_preserving_loss(embeddings, quantized_embeddings)
            
            # L2 regularization weighted by importance scores
            reg_loss = reg_strength * torch.sum(
                quantization_module.importance_scores * 
                (torch.norm(quantization_module.thresholds, 2) + 
                 torch.norm(torch.abs(quantization_module.scales) - 1, 2))
            )
            
            # Binary entropy loss for encouraging discrete outputs
            entropy_loss = 0.01 * -torch.mean(
                quantized_embeddings * torch.log(quantized_embeddings + 1e-6) +
                (1 - quantized_embeddings) * torch.log(1 - quantized_embeddings + 1e-6)
            )
            
            # Combine losses with adaptive weighting
            entropy_weight = min(0.1, 0.01 * epoch)  # Gradually increase entropy weight
            loss = sim_loss + reg_loss + entropy_weight * entropy_loss
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            # Update Hessian diagonal after backward pass
            quantization_module.update_hessian(embeddings)
            
            
            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(quantization_module.parameters(), max_norm=max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            # Accumulate losses
            total_loss += loss.item()
            total_sim_loss += sim_loss.item()
            total_reg_loss += reg_loss.item()
            total_entropy_loss += entropy_loss.item()
            
            # Print batch progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Batch [{batch_idx+1}/{len(dataloader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Compute average losses
        avg_loss = total_loss / len(dataloader)
        avg_sim_loss = total_sim_loss / len(dataloader)
        avg_reg_loss = total_reg_loss / len(dataloader)
        avg_entropy_loss = total_entropy_loss / len(dataloader)
        
        # Update temperature with cosine annealing
        # progress = epoch / num_epochs
        # START_TEMP = 0.1  # Starting temperature
        # END_TEMP = 0.05  # Ending temperature
        # quantization_module.temperature = max(
        #     END_TEMP,  # Minimum temperature
        #     0.5 * (1 + np.cos(np.pi * progress)) * (START_TEMP - END_TEMP) + END_TEMP  # Cosine decay from START_TEMP to END_TEMP
        # )
        
        # Prune dimensions if we're past halfway and loss is stable
        if epoch > num_epochs // 2:
            current_dims = quantization_module.pruning_mask.sum().item()
            threshold = 0.1 * (1 + epoch / num_epochs)  # Gradually increase threshold
            mask = quantization_module.prune_dimensions(threshold=threshold)
            pruned_dims = current_dims - quantization_module.pruning_mask.sum().item()
            
            print(f'Pruned {pruned_dims} dimensions. '
                  f'Current embedding size: {quantization_module.thresholds.shape[0]}')
        
        # Update training statistics
        training_stats['epoch_losses'].append({
            'total': avg_loss,
            'similarity': avg_sim_loss,
            'regularization': avg_reg_loss,
            'entropy': avg_entropy_loss
        })
        training_stats['importance_scores_history'].append(
            quantization_module.importance_scores.detach().cpu().numpy()
        )
        training_stats['pruned_dimensions'].append(
            initial_dimensions - quantization_module.thresholds.shape[0]
        )
        training_stats['temperature_history'].append(
            quantization_module.temperature
        )
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Avg Loss: {avg_loss:.4f}, '
              f'Sim Loss: {avg_sim_loss:.4f}, '
              f'Reg Loss: {avg_reg_loss:.4f}, '
              f'Entropy Loss: {avg_entropy_loss:.4f}, '
              f'Temperature: {quantization_module.temperature:.4f}')
    
    
    
    final_active_dims = quantization_module.pruning_mask.sum().item()
    print(f"\nTraining Complete!")
    print(f"Initial dimensions: {initial_dimensions}")
    print(f"Final active dimensions: {final_active_dims}")
    print(f"Pruned dimensions: {initial_dimensions - final_active_dims}")
    print(f"Pruning ratio: {(initial_dimensions - final_active_dims) / initial_dimensions:.2%}")
    
    dimension_stats = {
        'initial_dims': initial_dimensions,
        'final_dims': final_active_dims,
        'pruned_dims': initial_dimensions - final_active_dims,
        'pruning_ratio': (initial_dimensions - final_active_dims) / initial_dimensions,
        'final_pruning_mask': quantization_module.pruning_mask.cpu().numpy(),
        'final_importance_scores': quantization_module.importance_scores.detach().cpu().numpy()
    }
    training_stats['dimension_stats'] = dimension_stats
    # pretty print the dimension stats
    # Convert numpy arrays to lists for JSON serialization
    json_safe_stats = {
        k: v.tolist() if isinstance(v, np.ndarray) else v 
        for k, v in dimension_stats.items()
    }
    # print(json.dumps(json_safe_stats, indent=4))
    
    return quantization_module
  
