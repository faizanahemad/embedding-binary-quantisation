from unittest import result
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
from transformers import AutoTokenizer, AutoModel  
from torch.utils.data import DataLoader, Dataset  
import numpy as np  
from config import base_model_name, reg_strength, num_epochs, batch_size, temperature

from dataset import CombinedSimilarityDataset


import os  
from datetime import datetime  
  
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
  
    # Compute similarity matrices  
    sim_original = torch.matmul(original_norm, original_norm.t())  # Shape: (batch_size, batch_size)  
    sim_quantized = torch.matmul(quantized_norm, quantized_norm.t())  # Shape: (batch_size, batch_size)  
  
    # Compute Mean Squared Error between similarity matrices  
    loss = F.mse_loss(sim_quantized, sim_original)  
  
    return loss  

def eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))

def matching_preserving_loss(original_embeddings, quantized_embeddings):
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


def contrastive_loss(embeddings, temperature=0.05):
    """
    Compute InfoNCE/SimCSE style contrastive loss with temperature scaling.
    
    Args:
        embeddings (torch.Tensor): Batch of embeddings (batch_size, embedding_dim)
        temperature (float): Temperature parameter to scale similarities. 
                           Lower values make the model more sensitive to differences.
        
    Returns:
        loss (torch.Tensor): Scalar contrastive loss value
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute similarity matrix with temperature scaling
    sim_matrix = torch.matmul(embeddings, embeddings.t())
    
    # Create positive pair mask
    batch_size = embeddings.shape[0]
    pos_mask = torch.zeros((batch_size, batch_size), device=embeddings.device)
    
    # Mark consecutive pairs as positive: (0,1), (2,3), etc.
    for i in range(0, batch_size-1, 2):
        pos_mask[i, i+1] = 1
        pos_mask[i+1, i] = 1
    
    # Apply temperature scaling before exponentiation
    sim_matrix = sim_matrix / temperature
    
    # For each anchor, get its positive similarity and all similarities
    exp_sim_matrix = torch.exp(sim_matrix)
    
    # Zero out diagonal to exclude self-similarity
    exp_sim_matrix = exp_sim_matrix * (1 - torch.eye(batch_size, device=embeddings.device))
    
    # Compute positive similarities
    pos_sim = exp_sim_matrix * pos_mask
    
    # Sum positive similarities (should be only one per row)
    pos_sim = pos_sim.sum(dim=1)
    
    # Sum all similarities for denominator (excluding self-similarity)
    all_sim = exp_sim_matrix.sum(dim=1)
    
    # Compute loss: -log(pos_sim / all_sim)
    loss = -torch.log(pos_sim / all_sim + 1e-8)  # Add epsilon for numerical stability
    
    return loss.mean()
    
  

def get_dataloader(base_model_name, batch_size, num_workers=4, persistent_workers=True, prefetch_factor=2):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    dataset = CombinedSimilarityDataset(tokenizer, max_length=384, max_samples_per_dataset=10000)
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
            "model_name": model_name,
            "base_model": model_name,
            "base_model_revision": None,
            "language": ["en"],
            "similarity_fn_name": "cos_sim"
        }
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
            "model_name": model_name,
            "base_model": model_name,
            "base_model_revision": None,
            "language": ["en"],
            "similarity_fn_name": "cos_sim"
        }
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
            "model_name": quantization_module.__class__.__name__,
            "base_model": quantization_module.__class__.__name__,
            "base_model_revision": None,
            "language": ["en"],
            "similarity_fn_name": "cos_sim",
            "license": "apache-2.0",
            "pipeline_tag": "sentence-similarity"
        }

  
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
    
    
class OriginalEmbeddingCaller(Wrapper, Encoder):
    def __init__(self, model_name: str, embedding_dim: int):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        raise NotImplementedError("OriginalEmbeddingCaller should be subclassed.")
    
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
        
    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        with torch.no_grad(): 
            embeddings = self.model.encode(sentences, **kwargs)
        return embeddings
    
    
def mean_pool_and_L2_normalize(embeddings, attention_mask):
    # Mean pooling with attention mask to ignore padded tokens
    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.last_hidden_state.size()).float()
    sum_embeddings = torch.sum(embeddings.last_hidden_state * attention_mask_expanded, 1)
    sum_mask = attention_mask_expanded.sum(1)
    embeddings = sum_embeddings / sum_mask  # Get average ignoring padded tokens
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings