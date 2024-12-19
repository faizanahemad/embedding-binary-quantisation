from unittest import result
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
from transformers import AutoTokenizer, AutoModel  
from torch.utils.data import DataLoader, Dataset  
import numpy as np  
from config import base_model_name, reg_strength, num_epochs, batch_size, temperature, max_samples_per_dataset, dimension_levels

from dataset import CombinedSimilarityDataset
from typing import List
import math


import os  
from datetime import datetime

def get_dimension_levels(embedding_dim):
    return [embedding_dim//level for level in dimension_levels]

from dataclasses import dataclass, asdict
@dataclass
class ModelCardData:
    name: str
    base_model: str
    base_model_revision: str | None
    language: list[str]
    similarity_fn_name: str
    revision: str


    def model_name_as_path(self) -> str:
        return self.name

    def to_dict(self) -> dict:
        return asdict(self)
    
    
class FFNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, negative_slope=0.01):
        """
        Feed-forward neural network layer with LayerNorm, LeakyReLU, and Dropout.
        
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            dropout_rate (float): Dropout probability (default: 0.1)
            negative_slope (float): Negative slope for LeakyReLU (default: 0.01)
        """
        super(FFNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate fan_in and fan_out for Xavier/Kaiming initialization
        fan_in = input_dim
        fan_out = output_dim
        
        # Initialize weights using Kaiming initialization
        # Accounts for LeakyReLU's negative slope
        std = math.sqrt(2.0 / (fan_in * (1 + negative_slope**2)))
        nn.init.kaiming_normal_(
            self.linear.weight, 
            a=negative_slope, 
            mode='fan_in', 
            nonlinearity='leaky_relu'
        )
        
        # Initialize bias
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.linear.bias, -bound, bound)
        
        # Initialize LayerNorm parameters
        nn.init.constant_(self.layer_norm.weight, 1.0)
        nn.init.constant_(self.layer_norm.bias, 0.0)

    def forward(self, x):
        """
        Forward pass of the FFN layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        x = self.linear(x)
        x = self.layer_norm(x)  # Pre-activation normalization
        x = self.leaky_relu(x)
        x = self.dropout(x)
        return x


class SkipConnectionFFNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, negative_slope=0.01):
        """
        FFN layer with skip connection, adapting dimensions if needed.
        
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            dropout_rate (float): Dropout probability (default: 0.1)
            negative_slope (float): Negative slope for LeakyReLU (default: 0.01)
        """
        super(SkipConnectionFFNLayer, self).__init__()
        self.ffn_layer = FFNLayer(
            input_dim, 
            output_dim, 
            dropout_rate=dropout_rate, 
            negative_slope=negative_slope
        )
        
        # Create projection layer if dimensions don't match
        if input_dim != output_dim:
            self.skip_connection = nn.Linear(input_dim, output_dim)
            
            # Initialize skip connection weights
            fan_in = input_dim
            fan_out = output_dim
            
            # Use Xavier uniform initialization for the skip connection
            # as it doesn't have a non-linearity
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            nn.init.uniform_(self.skip_connection.weight, -bound, bound)
            nn.init.zeros_(self.skip_connection.bias)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x):
        """
        Forward pass with skip connection.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        return self.ffn_layer(x) + self.skip_connection(x)
    
    
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x)


class ModernFFN(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        dropout_rate: float = 0.1,
        expansion_factor: float = 4.0,
        activation: str = 'swiglu',
        use_bias: bool = False,
        multiple_of: int = 256  # Ensure hidden dim is multiple of this
    ):
        """
        Modern FFN with SwiGLU/GLU variants, RMSNorm, and skip connections.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            dropout_rate: Dropout probability
            expansion_factor: Factor to expand hidden dimension
            activation: Activation type ('swiglu', 'geglu', 'gelu', 'silu')
            use_bias: Whether to use bias in linear layers
            multiple_of: Ensure hidden dim is multiple of this value
        """
        super().__init__()
        
        # Calculate hidden dimension
        hidden_dim = int(input_dim * expansion_factor)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        # Pre-normalization
        self.norm = RMSNorm(input_dim)
        
        # Main FFN blocks
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.down_proj = nn.Linear(hidden_dim, output_dim, bias=use_bias)
        
        # Skip connection handling
        if input_dim != output_dim:
            self.skip_proj = nn.Linear(input_dim, output_dim, bias=use_bias)
        else:
            self.skip_proj = nn.Identity()
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Activation function
        self.activation_type = activation
        self.act_fn = self._get_activation(activation)
        
        self._init_weights()
    
    def _get_activation(self, activation: str):
        """Get activation function"""
        if activation == 'swiglu':
            return nn.SiLU(inplace=True)
        elif activation == 'geglu':
            return nn.GELU()
        elif activation == 'silu':
            return nn.SiLU(inplace=True)
        elif activation == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _init_weights(self):
        """Initialize weights using modern practices"""
        # Calculate fan_in and fan_out for proper scaling
        fan_in = self.up_proj.in_features
        fan_out = self.up_proj.out_features
        
        # Initialize projection matrices
        for module in [self.up_proj, self.gate_proj]:
            # Use scaled initialization for better gradient flow
            std = math.sqrt(2.0 / fan_in)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        # Special initialization for output projection
        hidden_dim = self.down_proj.in_features
        output_dim = self.down_proj.out_features
        std = math.sqrt(1.0 / hidden_dim)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=std)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)
        
        # Initialize skip projection if needed
        if isinstance(self.skip_proj, nn.Linear):
            # Use Xavier uniform for skip connection
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            nn.init.uniform_(self.skip_proj.weight, -bound, bound)
            if self.skip_proj.bias is not None:
                nn.init.zeros_(self.skip_proj.bias)
        
        # Initialize RMSNorm
        nn.init.ones_(self.norm.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated activation and skip connection.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim)
        """
        residual = self.skip_proj(x)
        
        # Pre-normalization
        x = self.norm(x)
        
        # Gated activation
        if self.activation_type in ['swiglu', 'geglu']:
            gate = self.act_fn(self.gate_proj(x))
            up = self.up_proj(x)
            x = gate * up
        else:
            x = self.up_proj(x)
            x = self.act_fn(x)
        
        # Project back to output dimension
        x = self.down_proj(x)
        
        # Dropout and skip connection
        x = self.dropout(x)
        x = x + residual
        
        return x


# Example usage:
"""
model = ModernFFN(
    input_dim=768,
    output_dim=512,
    dropout_rate=0.1,
    expansion_factor=4.0,
    activation='swiglu',
    use_bias=False
)

# Input shape: (batch_size, seq_len, input_dim)
x = torch.randn(32, 128, 768)
output = model(x)  # Shape: (32, 128, 512)
"""


def create_mlp(input_dim: int, 
               hidden_dims: List[int], 
               output_dim: int,
               dropout_rate: float = 0.1,
               negative_slope: float = 0.01,
               use_skip_connections: bool = True) -> nn.Module:
    """
    Create a multi-layer perceptron with optional skip connections.
    
    Args:
        input_dim (int): Input dimension
        hidden_dims (List[int]): List of hidden layer dimensions
        output_dim (int): Output dimension
        dropout_rate (float): Dropout probability
        negative_slope (float): Negative slope for LeakyReLU
        use_skip_connections (bool): Whether to use skip connections
        
    Returns:
        nn.Module: MLP model
    """
    layers = []
    current_dim = input_dim
    
    # Add hidden layers
    for hidden_dim in hidden_dims:
        if use_skip_connections:
            layers.append(
                SkipConnectionFFNLayer(
                    current_dim, 
                    hidden_dim,
                    dropout_rate=dropout_rate,
                    negative_slope=negative_slope
                )
            )
        else:
            layers.append(
                FFNLayer(
                    current_dim, 
                    hidden_dim,
                    dropout_rate=dropout_rate,
                    negative_slope=negative_slope
                )
            )
        current_dim = hidden_dim
    
    # Add output layer
    if use_skip_connections:
        layers.append(
            SkipConnectionFFNLayer(
                current_dim, 
                output_dim,
                dropout_rate=dropout_rate,
                negative_slope=negative_slope
            )
        )
    else:
        layers.append(
            FFNLayer(
                current_dim, 
                output_dim,
                dropout_rate=dropout_rate,
                negative_slope=negative_slope
            )
        )
    
    return nn.Sequential(*layers)


class ModernFFNWithoutSkip(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        dropout_rate: float = 0.1,
        expansion_factor: float = 4.0,
        activation: str = 'swiglu',
        use_bias: bool = False,
        multiple_of: int = 256,
        negative_slope: float = 0.01,  # Kept for backward compatibility
    ):
        """
        Modern FFN without skip connections.
        
        Args: Same as ModernFFN, with negative_slope kept for compatibility
        """
        super().__init__()
        
        # Calculate hidden dimension
        hidden_dim = int(input_dim * expansion_factor)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        # Pre-normalization
        self.norm = RMSNorm(input_dim)
        
        # Main FFN blocks
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.down_proj = nn.Linear(hidden_dim, output_dim, bias=use_bias)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.activation_type = activation
        self.act_fn = self._get_activation(activation)
        
        self._init_weights()
    
    def _get_activation(self, activation: str):
        return ModernFFN._get_activation(self, activation)
    
    def _init_weights(self):
        """Initialize weights using modern practices"""
        fan_in = self.up_proj.in_features
        fan_out = self.up_proj.out_features
        
        for module in [self.up_proj, self.gate_proj]:
            std = math.sqrt(2.0 / fan_in)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        hidden_dim = self.down_proj.in_features
        output_dim = self.down_proj.out_features
        std = math.sqrt(1.0 / hidden_dim)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=std)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)
        
        nn.init.ones_(self.norm.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        
        if self.activation_type in ['swiglu', 'geglu']:
            gate = self.act_fn(self.gate_proj(x))
            up = self.up_proj(x)
            x = gate * up
        else:
            x = self.up_proj(x)
            x = self.act_fn(x)
        
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


def create_modern_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    dropout_rate: float = 0.1,
    negative_slope: float = 0.01,  # Kept for backward compatibility
    use_skip_connections: bool = True,
    activation: str = 'swiglu',
    expansion_factor: float = 4.0,
    use_bias: bool = True,
    multiple_of: int = 256
) -> nn.Module:
    """
    Create a modern multi-layer perceptron with optional skip connections.
    
    Args:
        input_dim (int): Input dimension
        hidden_dims (List[int]): List of hidden layer dimensions
        output_dim (int): Output dimension
        dropout_rate (float): Dropout probability
        negative_slope (float): Kept for backward compatibility
        use_skip_connections (bool): Whether to use skip connections
        activation (str): Activation type ('swiglu', 'geglu', 'gelu', 'silu')
        expansion_factor (float): Hidden layer expansion factor
        use_bias (bool): Whether to use bias in linear layers
        multiple_of (int): Ensure hidden dimensions are multiples of this
        
    Returns:
        nn.Module: Modern MLP model
    """
    layers = []
    current_dim = input_dim
    
    FFNClass = ModernFFN if use_skip_connections else ModernFFNWithoutSkip
    
    # Add hidden layers
    for hidden_dim in hidden_dims:
        layers.append(
            FFNClass(
                input_dim=current_dim,
                output_dim=hidden_dim,
                dropout_rate=dropout_rate,
                expansion_factor=expansion_factor,
                activation=activation,
                use_bias=use_bias,
                multiple_of=multiple_of,
                # negative_slope=negative_slope  # Kept for compatibility
            )
        )
        current_dim = hidden_dim
    
    # Add output layer
    layers.append(
        FFNClass(
            input_dim=current_dim,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
            expansion_factor=expansion_factor,
            activation=activation,
            use_bias=use_bias,
            multiple_of=multiple_of,
            # negative_slope=negative_slope  # Kept for compatibility
        )
    )
    
    return nn.Sequential(*layers)


# Example usage:
"""
model = create_modern_mlp(
    input_dim=768,
    hidden_dims=[512, 256],
    output_dim=128,
    dropout_rate=0.1,
    use_skip_connections=True,
    activation='swiglu',
    expansion_factor=4.0,
    use_bias=False
)

# Input shape: (batch_size, seq_len, input_dim)
x = torch.randn(32, 128, 768)
output = model(x)  # Shape: (32, 128, 128)
"""




def create_save_directory(base_dir='saved_models'):
    """  
    Creates a directory named 'run_{date_time}' inside the base directory.  
  
    Args:  
        base_dir (str): Base directory where the run directory will be created.  
  
    Returns:  
        save_dir (str): The path to the created directory.  
    """  
    date_time = datetime.now().strftime('%Y%m%d_%H%M')  
    save_dir = os.path.join(base_dir, f'run_{date_time}')  
    os.makedirs(save_dir, exist_ok=True)  
    return save_dir  
  
def save_quantization_module(model, save_dir, model_name):  
    """  
    Saves the quantization module's parameters to the specified directory.  
  
    Args:  
        model (nn.Module): The quantization module to save.  
        save_dir (str): Directory where the model will be saved.  
        model_name (str): Name to use for the saved model file.  
    """  
    model_path = os.path.join(save_dir, f'{model_name}.pth')  
    torch.save(model.state_dict(), model_path)  
    print(f'Model saved to {model_path}')  
    
    
def similarity_preservation_loss(original_embeddings, quantized_embeddings):  
    """  
    Compute the loss to preserve similarity relationships.  
  
    Args:  
        original_embeddings (torch.Tensor): Original embeddings, shape (batch_size, embedding_dim)  
        quantized_embeddings (torch.Tensor): Quantized embeddings, shape (batch_size, embedding_dim_new)  
  
    Returns:  
        loss (torch.Tensor): Scalar loss value  
    """  
    # Normalize embeddings  
    original_norm = F.normalize(original_embeddings, dim=1)  
    quantized_norm = F.normalize(quantized_embeddings, dim=1)  
    # original_norm = original_embeddings
    # quantized_norm = quantized_embeddings
  
    # Compute similarity matrices  
    sim_original = torch.matmul(original_norm, original_norm.t())  # Shape: (batch_size, batch_size)  
    sim_quantized = torch.matmul(quantized_norm, quantized_norm.t())  # Shape: (batch_size, batch_size)  
  
    # Compute Mean Squared Error between similarity matrices  
    loss = F.mse_loss(sim_quantized, sim_original)  
  
    return loss  


def kl_similarity_preservation_loss(original_embeddings, quantized_embeddings, eps=1e-8, temperature=0.1):
    """
    Compute similarity preservation loss using average of two KL divergences.
    Compute similarity preservation loss using Jensen-Shannon Divergence (JSD).
    JSD is symmetric and bounded, making it a better choice than KL divergence alone.
    Uses softmax with temperature scaling for proper probability distributions.
    
    Args:
        original_embeddings (torch.Tensor): Original embeddings, shape (batch_size, embedding_dim)
        quantized_embeddings (torch.Tensor): Quantized embeddings, shape (batch_size, embedding_dim_new)
        temperature (float): Temperature for scaling similarities (lower = sharper)
        eps (float): Small constant for numerical stability
        
    Returns:
        loss (torch.Tensor): Scalar loss value representing mean symmetric KL
    """
    # Normalize embeddings
    # original_norm = F.normalize(original_embeddings, dim=1)
    # quantized_norm = F.normalize(quantized_embeddings, dim=1)
    original_norm = original_embeddings
    quantized_norm = quantized_embeddings
    
    
    # Compute similarity matrices
    sim_original = torch.matmul(original_norm, original_norm.t()) / temperature
    sim_quantized = torch.matmul(quantized_norm, quantized_norm.t()) / temperature
    
    # Convert to probabilities using softmax
    sim_original = F.softmax(sim_original, dim=-1)
    sim_quantized = F.softmax(sim_quantized, dim=-1)
    
    # Add small epsilon for numerical stability
    sim_original = sim_original.clamp(min=eps)
    sim_quantized = sim_quantized.clamp(min=eps)
    
    # Compute KL in both directions
    kl_orig_quant = F.kl_div(
        sim_quantized.log(), 
        sim_original, 
        reduction='batchmean',
        log_target=False
    )
    
    kl_quant_orig = F.kl_div(
        sim_original.log(), 
        sim_quantized, 
        reduction='batchmean',
        log_target=False
    )
    
    # Average the two KL divergences
    symmetric_kl = (kl_orig_quant + kl_quant_orig) / 2
    
    return symmetric_kl

def eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))

def matching_preserving_loss(original_embeddings, quantized_embeddings):
    return 0.0
    original_embeddings_normalized = F.normalize(original_embeddings, dim=1)
    quantized_embeddings_normalized = F.normalize(quantized_embeddings, dim=1)
    cosine_similarities = torch.matmul(original_embeddings_normalized, quantized_embeddings_normalized.t())
    rank_loss = F.mse_loss(cosine_similarities, eye_like(cosine_similarities))
    return rank_loss

def rank_preserving_loss(original_embeddings, quantized_embeddings):
    original_embeddings_normalized = F.normalize(original_embeddings, dim=1)
    quantized_embeddings_normalized = F.normalize(quantized_embeddings, dim=1)
    sim_original = torch.matmul(original_embeddings_normalized, original_embeddings_normalized.t())
    sim_quantized = torch.matmul(quantized_embeddings_normalized, quantized_embeddings_normalized.t())
    
    # Create masks for upper triangular part (excluding diagonal)
    mask = torch.triu(torch.ones_like(sim_original), diagonal=1)
    
    # Get pairs of similarities
    sim_orig_pairs1 = sim_original.unsqueeze(2)  # [batch, batch, 1]
    sim_orig_pairs2 = sim_original.unsqueeze(1)  # [batch, 1, batch]
    sim_quant_pairs1 = sim_quantized.unsqueeze(2)  # [batch, batch, 1]
    sim_quant_pairs2 = sim_quantized.unsqueeze(1)  # [batch, 1, batch]
    
    # Compare relative ordering
    orig_diff = sim_orig_pairs1 - sim_orig_pairs2  # [batch, batch, batch]
    quant_diff = sim_quant_pairs1 - sim_quant_pairs2  # [batch, batch, batch]
    
    # Use sigmoid to get soft sign of differences
    orig_sign = torch.sigmoid(orig_diff * temperature)  # Scale factor 10 makes sigmoid sharper
    quant_sign = torch.sigmoid(quant_diff * temperature)
    
    # Compute loss when relative ordering is different
    loss = F.mse_loss(quant_sign, orig_sign, reduction='none')
    
    # Apply mask to consider only unique pairs
    mask = mask.unsqueeze(2) * torch.ones_like(loss)
    loss = loss * mask
    
    # Average over all valid pairs
    loss = loss.sum() / (mask.sum() + 1e-6)
    
    return loss


def contrastive_loss(embeddings, temperature=0.05, hard_negative_ratio=0.25):
    """
    Compute InfoNCE/SimCSE style contrastive loss with temperature scaling.
    
    Args:
        embeddings (torch.Tensor): Batch of embeddings (batch_size, embedding_dim)
        temperature (float): Temperature parameter to scale similarities. 
                           Lower values make the model more sensitive to differences.
        hard_negative_ratio (float): Ratio of hard negatives to use in the loss computation.
        
    Returns:
        loss (torch.Tensor): Scalar contrastive loss value
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.t())
    
    # Create positive pair mask
    batch_size = embeddings.shape[0]
    pos_mask = torch.zeros((batch_size, batch_size), device=embeddings.device)
    
    # Mark consecutive pairs as positive: (0,1), (2,3), etc.
    for i in range(0, batch_size-1, 2):
        pos_mask[i, i+1] = 1
        pos_mask[i+1, i] = 1
    
    # Apply temperature scaling
    sim_matrix = sim_matrix / temperature
    
    # Apply LogSumExp trick for numerical stability
    sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - sim_max.detach()
    
    # Compute exp(similarity/temperature) for all pairs
    exp_sim_matrix = torch.exp(sim_matrix)
    
    # Create negative mask (everything except positives and self)
    neg_mask = 1 - pos_mask - torch.eye(batch_size, device=embeddings.device)
    
    # Compute positive similarities
    pos_sim = (exp_sim_matrix * pos_mask).sum(dim=1)
    
    # Compute denominator (sum of all negative similarities + positive similarities)
    neg_sim = (exp_sim_matrix * neg_mask).sum(dim=1)
    
    # Compute loss: -log(pos_sim / (pos_sim + neg_sim))
    loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
    
    
    # Find hardest negatives (highest similarities among negatives)
    neg_sims = exp_sim_matrix * neg_mask
    k = int(batch_size * hard_negative_ratio)
    hard_negatives, _ = torch.topk(neg_sims, k=k, dim=1)
    
    # Use only hard negatives in denominator
    neg_sim = hard_negatives.sum(dim=1)
    
    hard_negative_loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
    
    loss = loss.mean() + hard_negative_loss.mean()
    
    return loss
    
    
    
    # Compute loss: -log(pos_sim / all_sim)
    loss = -torch.log(pos_sim / all_sim + 1e-8)  # Add epsilon for numerical stability
  
    # Compute loss: -log(pos_sim / all_sim)
    loss = -torch.log(pos_sim / all_sim + 1e-8)  # Add epsilon for numerical stability

class PairwiseShuffleSampler(torch.utils.data.Sampler):
    """
    Custom sampler that keeps pairs together while shuffling between epochs.
    Pairs are assumed to be consecutive indices (0,1), (2,3), etc.
    """
    def __init__(self, data_source):
        self.data_source = data_source
        # Get indices of first element of each pair
        self.pair_first_indices = list(range(0, len(data_source), 2))
        
    def __iter__(self):
        # Shuffle the pairs (but keep elements within pairs together)
        shuffled_pair_indices = torch.randperm(len(self.pair_first_indices)).tolist()
        final_indices = []
        for idx in shuffled_pair_indices:
            pair_start = self.pair_first_indices[idx]
            final_indices.extend([pair_start, pair_start + 1])
        return iter(final_indices)
    
    def __len__(self):
        return len(self.data_source)

def get_dataloader(base_model_name, batch_size, num_workers=4, persistent_workers=True, prefetch_factor=2):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    dataset = CombinedSimilarityDataset(tokenizer, max_length=256, max_samples_per_dataset=max_samples_per_dataset)
    print(f"Train Dataset size: {len(dataset)}")
    
    # Use our custom sampler instead of SequentialSampler
    sampler = PairwiseShuffleSampler(dataset)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=num_workers, 
        persistent_workers=persistent_workers, 
        prefetch_factor=prefetch_factor
    )
    print(f"Train Dataloader size: {len(dataloader)}")
    return dataloader


def get_dataloader(base_model_name, batch_size, num_workers=4, persistent_workers=True, prefetch_factor=2):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    dataset = CombinedSimilarityDataset(tokenizer, max_length=256, max_samples_per_dataset=max_samples_per_dataset)
    print(f"Train Dataset size: {len(dataset)}")
    # Create a sampler that keeps pairs together while shuffling between pairs
    indices = list(range(0, len(dataset), 2))  # Get indices of first element of each pair
    shuffled_pair_indices = torch.randperm(len(indices)).tolist()
    final_indices = []
    for idx in shuffled_pair_indices:
        pair_start = indices[idx]
        final_indices.extend([pair_start, pair_start + 1])  # Keep pairs together
        
    sampler = torch.utils.data.sampler.SequentialSampler(final_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
    print(f"Train Dataloader size: {len(dataloader)}")
    return dataloader


from mteb.models.wrapper import Wrapper
from mteb.encoder_interface import Encoder
from sentence_transformers import SentenceTransformer  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from typing import List, Dict  


class OriginalEmbeddingModel(Wrapper, Encoder):  
    """  
    Original embedding model without any quantization.  
  
    This model uses the pre-trained embedding model directly.  
    """  
    def __init__(self, model_name: str):  
        self.model = SentenceTransformer(model_name)  
        self.model.to(device)
        self.model_card_data = {
            "name": model_name,
            "base_model": model_name,
            "base_model_revision": None,
            "language": ["en"],
            "similarity_fn_name": "cos_sim",
            "revision": "1.0.0",
        }
        self.mteb_model_meta = ModelCardData(**self.model_card_data)
        # print("[INIT] Finished creating OriginalEmbeddingModel\n")
        
    def __call__(self, *args, **kwargs):
        # print("\n[DEBUG] OriginalEmbeddingModel.__call__ was invoked")
        return self.encode(*args, **kwargs)
        
  
    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:  
        """  
        Encode sentences into embeddings.  
  
        Args:  
            sentences (List[str]): List of sentences to encode.  
  
        Returns:  
            np.ndarray: Array of embeddings.  
        """  
        # raise NotImplementedError("OriginalEmbeddingModel should not be used for encoding")
        # print("\n[DEBUG] Starting OriginalEmbeddingModel.encode()")
        # print(f"Encoding {len(sentences)} sentences with OriginalEmbeddingModel")
        # print("\n[ENCODE] Starting encode method", file=sys.stderr)  # Print to stderr
        # sys.stderr.flush()  # Force flush stderr
        # # Log to file as well
        # with open('debug.log', 'a') as f:
        #     f.write(f"\nEncoding {len(sentences)} sentences\n")
        
        embeddings = self.model.encode(  
            sentences,  
            show_progress_bar=kwargs.get('show_progress_bar', True),  
            # batch_size=kwargs.get('batch_size', 32),  
            encode_kwargs = {'batch_size': kwargs.get('batch_size', batch_size)},
            normalize_embeddings=True  
        )  
        # print("[DEBUG] Finished OriginalEmbeddingModel.encode()\n")
        # print("[ENCODE] Finished encode method\n", file=sys.stderr)
        # sys.stderr.flush()
        
        return embeddings  
    
    
class OriginalEmbeddingModelBinary(Wrapper, Encoder):  
    """  
    Original embedding model without any quantization.  
  
    This model uses the pre-trained embedding model directly.  
    """  
    def __init__(self, model_name: str):  
        self.model = SentenceTransformer(model_name)  
        self.model.to(device)
        self.model_card_data = {
            "name": model_name,
            "base_model": model_name,
            "base_model_revision": None,
            "language": ["en"],
            "similarity_fn_name": "cos_sim",
            "revision": "1.0.0",
        }
        self.mteb_model_meta = ModelCardData(**self.model_card_data)
        # print("[INIT] Finished creating OriginalEmbeddingModel\n")
        
    def __call__(self, *args, **kwargs):
        # print("\n[DEBUG] OriginalEmbeddingModel.__call__ was invoked")
        return self.encode(*args, **kwargs)
        
  
    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:  
        """  
        Encode sentences into embeddings.  
  
        Args:  
            sentences (List[str]): List of sentences to encode.  
  
        Returns:  
            np.ndarray: Array of embeddings.  
        """  
        # raise NotImplementedError("OriginalEmbeddingModel should not be used for encoding")
        # print("\n[DEBUG] Starting OriginalEmbeddingModel.encode()")
        # print(f"Encoding {len(sentences)} sentences with OriginalEmbeddingModel")
        # print("\n[ENCODE] Starting encode method", file=sys.stderr)  # Print to stderr
        # sys.stderr.flush()  # Force flush stderr
        # # Log to file as well
        # with open('debug.log', 'a') as f:
        #     f.write(f"\nEncoding {len(sentences)} sentences\n")
        
        embeddings = self.model.encode(  
            sentences,  
            show_progress_bar=kwargs.get('show_progress_bar', True),  
            # batch_size=kwargs.get('batch_size', 32),  
            encode_kwargs = {'batch_size': kwargs.get('batch_size', batch_size)},
            normalize_embeddings=True  
        )  
        # print("[DEBUG] Finished OriginalEmbeddingModel.encode()\n")
        # print("[ENCODE] Finished encode method\n", file=sys.stderr)
        # sys.stderr.flush()
        
        return (embeddings > 0).astype(np.float32)
  

class QuantizedEmbeddingModel(Wrapper, Encoder):  
    """  
    Embedding model with quantization applied.  
  
    This model applies quantization to the embeddings.  
    """  
    def __init__(self, embedding_model: SentenceTransformer, quantization_module):  
        self.embedding_model = embedding_model  
        self.model = embedding_model
        self.model.to(device)  # Move embedding model to GPU
        self.quantization_module = quantization_module
        self.quantization_module.to(device)  # Move quantization module to GPU
        self.model_card_data = {
            "name": quantization_module.__class__.__name__,
            "base_model": quantization_module.__class__.__name__,
            "base_model_revision": None,
            "language": ["en"],
            "similarity_fn_name": "cos_sim",
            "revision": "1.0.0",
        }
        self.mteb_model_meta = ModelCardData(**self.model_card_data)

  
    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:  
        """  
        Encode sentences into quantized embeddings.  
  
        Args:  
            sentences (List[str]): List of sentences to encode.  
  
        Returns:  
            np.ndarray: Array of quantized embeddings.  
        """  
        # Get embeddings from the base model  
        embeddings = self.embedding_model.encode(  
            sentences,  
            show_progress_bar=kwargs.get('show_progress_bar', False),  
            
            encode_kwargs = {'batch_size': kwargs.get('batch_size', batch_size)},
            normalize_embeddings=True  # Do not normalize before quantization  
        )  
        embeddings = torch.tensor(embeddings)  
        embeddings = embeddings.to(device)
        # print(f"[DEBUG] QuantizedEmbeddingModel.encode() - embeddings shape: {embeddings.shape}")
        
        with torch.no_grad():  
            quantized_embeddings = self.quantization_module(embeddings, binary=True).cpu().numpy()
        # print(f"[DEBUG] QuantizedEmbeddingModel.encode() - quantized_embeddings shape: {quantized_embeddings.shape}")
            
        
        return quantized_embeddings  
    
    
class OriginalEmbeddingCaller(nn.Module, Wrapper, Encoder):
    def __init__(self, model_name: str, embedding_dim: int):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        raise NotImplementedError("OriginalEmbeddingCaller should be subclassed.")
    
    def forward(self, *args, **kwargs):
        return self.encode(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
class SentenceTransformerEmbeddingCaller(OriginalEmbeddingCaller):
    def __init__(self, model_name: str):
        super().__init__(model_name, None)
        self.model = SentenceTransformer(model_name)
        try:
            self.embedding_dim = self.model.config.hidden_size
        except:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.model.to(device)
        
        self.model.eval()  
        for param in self.model.parameters():  
            param.requires_grad = False  
        
    def encode(self, sentences=None, attention_mask=None, input_ids=None, **kwargs) -> np.ndarray:
        """
        Encode sentences into embeddings.
        
        Args:
            sentences: List of strings or string or tensor of input_ids
            attention_mask: Optional attention mask tensor
            input_ids: Optional input_ids tensor (alternative to sentences)
            **kwargs: Additional arguments for sentence-transformers encode
            
        Returns:
            np.ndarray: Array of embeddings
        """
        with torch.no_grad():
            # Case 1: Handle tensor inputs (either through sentences or input_ids)
            if isinstance(sentences, torch.Tensor) or isinstance(input_ids, torch.Tensor):
                # Use whichever tensor was provided
                input_ids_tensor = sentences if isinstance(sentences, torch.Tensor) else input_ids
                
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids_tensor)
                
                # Ensure tensors are on the correct device
                input_ids_tensor = input_ids_tensor.to(device)
                attention_mask = attention_mask.to(device)
                
                # Process through the model's transformer
                outputs = self.model.forward({
                    'input_ids': input_ids_tensor, 
                    'attention_mask': attention_mask
                })
                
                # Apply mean pooling and L2 normalization
                embeddings = mean_pool_and_L2_normalize(outputs, attention_mask)
                return embeddings.cpu().numpy()
            
            # Case 2: Handle string inputs
            elif isinstance(sentences, (list, str)):
                embeddings = self.model.encode(
                    sentences,
                    show_progress_bar=kwargs.get('show_progress_bar', False),
                    encode_kwargs={'batch_size': kwargs.get('batch_size', batch_size)},
                    normalize_embeddings=True
                )
                return embeddings
            
            # Case 3: Invalid input
            else:
                raise ValueError("Input must be either a torch.Tensor, a list of strings, or a string")
            
    
def mean_pool_and_L2_normalize(model_output, attention_mask):
    """
    Apply mean pooling (if needed) and L2 normalization to model outputs.
    
    Args:
        model_output (dict or torch.Tensor): Model output dictionary or tensor
            If dict: Expected to have 'sentence_embedding' or 'token_embeddings'
            If tensor: Can be either:
                - 2D: (batch_size, embedding_dim) - no pooling needed
                - 3D: (batch_size, sequence_length, embedding_dim) - needs pooling
        attention_mask (torch.Tensor): Attention mask for padded tokens
            Shape: (batch_size, sequence_length)
            Only used if pooling is needed
        
    Returns:
        torch.Tensor: L2 normalized embeddings of shape (batch_size, embedding_dim)
    """
    # 1. Extract embeddings based on input type
    if isinstance(model_output, dict):
        if 'sentence_embedding' in model_output:
            # Already pooled embeddings, just normalize
            return F.normalize(model_output['sentence_embedding'], p=2, dim=1)
        elif 'token_embeddings' in model_output:
            embeddings = model_output['token_embeddings']
        else:
            embeddings = next(iter(model_output.values()))
    else:
        embeddings = model_output

    # 2. If already 2D (batch_size, embedding_dim), just normalize
    if len(embeddings.shape) == 2:
        return F.normalize(embeddings, p=2, dim=1)

    # 3. Only do pooling for 3D tensors (batch_size, sequence_length, embedding_dim)
    embedding_dim = embeddings.size(-1)
    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(-1, -1, embedding_dim).float()
    
    # Mean pooling with attention mask
    sum_embeddings = torch.sum(embeddings * attention_mask_expanded, 1)
    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
    pooled_embeddings = sum_embeddings / sum_mask.unsqueeze(-1)
    
    return F.normalize(pooled_embeddings, p=2, dim=1)