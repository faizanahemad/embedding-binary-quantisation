from ast import For
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from transformers import AutoModel  
from sentence_transformers import SentenceTransformer  
import numpy as np  
from typing import List  
from tqdm import tqdm  
from config import *  
from common import create_mlp, create_modern_mlp, kl_similarity_preservation_loss, rank_preserving_loss, similarity_preservation_loss, SentenceTransformerEmbeddingCaller, OriginalEmbeddingCaller, ModelCardData, contrastive_loss
  
class MatryoshkaEmbeddingModel(OriginalEmbeddingCaller):  
    """  
    Matryoshka Embedding Model that produces nested embeddings with optional binary and multi-bit quantization.  
      
    Attributes:  
        embedding_model_name (str): Name of the pretrained embedding model.  
        dimension_levels (List[int]): List of embedding dimensions in increasing order.  
        train_binary (bool): Flag indicating whether to train with binary (1-bit) quantization.  
        train_two_bit (bool): Flag indicating whether to train with 2-bit quantization.  
        expand_two_bit_to_three_bits (bool): Whether to expand 2-bit codes to 3-bit codes using a codebook.  
    """  
    def __init__(self,  
                 embedding_model: OriginalEmbeddingCaller,  
                 dimension_levels: List[int],  
                 train_binary: bool = False,  
                 train_two_bit: bool = False,  
                 expand_two_bit_to_three_bits: bool = False):  
        """  
        Initialize the Matryoshka Embedding Model.  
  
        Args:  
            embedding_model_name (str): Name of the pretrained embedding model.  
            dimension_levels (List[int]): List of embedding dimensions in increasing order.  
            train_binary (bool): Flag to enable binary (1-bit) quantization training.  
            train_two_bit (bool): Flag to enable 2-bit quantization training.  
            expand_two_bit_to_three_bits (bool): Whether to expand 2-bit codes to 3-bit codes.  
        """  
        super().__init__(embedding_model.model_name, embedding_model.embedding_dim)
        self.embedding_model = embedding_model
        self.dimension_levels = sorted(dimension_levels)  
        
        self.train_binary = train_binary  
        self.train_two_bit = train_two_bit  
        self.expand_two_bit_to_three_bits = expand_two_bit_to_three_bits
        self.model_card_data = {
            "name": "MatryoshkaEmbeddingModel",
            "base_model": self.embedding_model.model_name,
            "base_model_revision": None,
            "language": ["en"],
            "similarity_fn_name": "cos_sim",
            "revision": "1.0.0",
        }
        self.mteb_model_meta = ModelCardData(**self.model_card_data)
        
  
        
            
        self.embedding_dim = self.embedding_model.embedding_dim
        
        # Define the transformation network  
        self.transformer = MatryoshkaTransformer(  
            input_dim=self.embedding_dim,  
            dimension_levels=self.dimension_levels,  
            train_binary=self.train_binary,  
            train_two_bit=self.train_two_bit,  
            expand_two_bit_to_three_bits=self.expand_two_bit_to_three_bits  
        )  
        self.max_dim = self.embedding_dim
        self.baseline = False
        
    def save(self, path: str):
        torch.save(self.transformer.state_dict(), path)
        
    def load(self, path: str):
        self.transformer.load_state_dict(torch.load(path, map_location=torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')))
  
    def init_thresholds(self, sample_embeddings: torch.Tensor):  
        """  
        Initialize thresholds in quantization layers based on sample embeddings.  
  
        Args:  
            sample_embeddings (torch.Tensor): Sample embeddings used for initializing thresholds.  
        """  
        for quant_layer in self.transformer.quantization_layers.values():  
            if quant_layer.quantization_bits > 1:  
                quant_layer.initialize_thresholds(sample_embeddings)  
  
    def encode(self,  
               sentences: List[str],  
               output_dim: int = matryoshka_output_dim,  
               do_binary: bool = False,  
               do_two_bits: bool = False,  
               **kwargs) -> np.ndarray:  
        """  
        Encode sentences to obtain embeddings.  
  
        Args:  
            sentences (List[str]): List of sentences to encode.  
            output_dim (int, optional): Desired embedding dimension. If None, use max dimension.  
            do_binary (bool): Whether to output binary (1-bit) quantized embeddings.  
            do_two_bits (bool): Whether to output 2-bit quantized embeddings.  
            **kwargs: Additional arguments for the embedding model.  
  
        Returns:  
            np.ndarray: Embeddings of shape (num_sentences, output_dim)  
        """  

        if output_dim is None:  
            output_dim = self.max_dim  
        assert output_dim in self.dimension_levels, f"Output dimension must be one of {self.dimension_levels}"  
  
        if do_binary and not self.train_binary:  
            raise ValueError("Model was not trained with binary quantization. Set train_binary=True during training.")  
        if do_two_bits and not self.train_two_bit:  
            raise ValueError("Model was not trained with 2-bit quantization. Set train_two_bit=True during training.")  
  
        # Get embeddings from the frozen model  
        with torch.no_grad():  
            base_embeddings = self.embedding_model.encode(  
                sentences,  
                show_progress_bar=kwargs.get('show_progress_bar', False),  
                batch_size=kwargs.get('batch_size', 32),  
                normalize_embeddings=False  
            )  
            base_embeddings = torch.tensor(base_embeddings)  
  
        # Pass through the transformation network  
        baseline_embeddings = base_embeddings[:,:output_dim]
        if self.baseline:
            embeddings = baseline_embeddings
        else:
            self.transformer.eval()  
            with torch.no_grad():  
                matryoshka_embeddings = self.transformer(  
                    base_embeddings,  
                    apply_binary=do_binary,  
                    apply_two_bit=do_two_bits  
                )  
                embeddings = matryoshka_embeddings[output_dim]  
            assert not torch.allclose(embeddings, baseline_embeddings, rtol=1e-3, atol=1e-5), "Transformed embeddings should be different from baseline embeddings"
  
        # If binary mode, convert embeddings to binary  
        if do_binary:  
            embeddings = (embeddings > 0.5).float()  
        elif do_two_bits:  
            # Quantize embeddings to 2 bits per dimension  
            embeddings = self.transformer.quantize_two_bits(embeddings, self.expand_two_bit_to_three_bits)  
        else:  
            # Normalize embeddings  
            embeddings = F.normalize(embeddings, p=2, dim=1)  
        assert embeddings.shape[1] == matryoshka_output_dim, f"Output dimension {embeddings.shape[1]} does not match desired output dimension {matryoshka_output_dim}"

        return embeddings.cpu().numpy()  
  
class MatryoshkaTransformer(nn.Module):  
    """  
    Transformation network that produces nested embeddings using a Pyramid Network architecture,  
    with optional binary and multi-bit quantization layers.  
    """  
    def __init__(self,  
                 input_dim: int,  
                 dimension_levels: List[int],  
                 train_binary: bool = False,  
                 train_two_bit: bool = False,  
                 expand_two_bit_to_three_bits: bool = False):  
        """  
        Initialize the Transformer.  
  
        Args:  
            input_dim (int): Dimension of the input embeddings.  
            dimension_levels (List[int]): List of embedding dimensions in increasing order.  
            train_binary (bool): Whether to include binary quantization layers.  
            train_two_bit (bool): Whether to include 2-bit quantization layers.  
            expand_two_bit_to_three_bits (bool): Whether to expand 2-bit codes to 3 bits.  
        """  
        super(MatryoshkaTransformer, self).__init__()  
        self.dimension_levels = sorted(dimension_levels)  
        self.train_binary = train_binary  
        self.train_two_bit = train_two_bit  
        self.expand_two_bit_to_three_bits = expand_two_bit_to_three_bits  
  
        self.blocks = nn.ModuleList()  
        self.quantization_layers = nn.ModuleDict()  
        prev_dim = 0  
  
        assert input_dim >= self.dimension_levels[-1], "Input dimension must be greater than or equal to the largest dimension level"  
        if input_dim > self.dimension_levels[-1]:  
            self.dimension_levels.append(input_dim)  
  
        
        
        # self.base_transform = create_modern_mlp(
        #     input_dim=input_dim,
        #     hidden_dims=[input_dim * 4, input_dim * 4],
        #     output_dim=input_dim * 4,
        #     dropout_rate=0.1,
        #     negative_slope=0.01,
        #     use_skip_connections=True
        # )
        
        self.base_transform = nn.Identity()
  
        for dim in self.dimension_levels:  
            
            
            # block = create_modern_mlp(
            #     input_dim=input_dim * 4,
            #     hidden_dims=[input_dim * 4, input_dim * 2],
            #     output_dim=dim - prev_dim,
            #     dropout_rate=0.1,
            #     negative_slope=0.01,
            #     use_skip_connections=True
            # )
            
            block = nn.Sequential(nn.Linear(input_dim, input_dim*8), nn.LayerNorm(input_dim*8), nn.LeakyReLU(), nn.Linear(input_dim*8, input_dim*8), nn.LayerNorm(input_dim*8), nn.LeakyReLU(), nn.Linear(input_dim*8, dim - prev_dim))
            # block = nn.Linear(input_dim, dim - prev_dim)
            
            # init block
            nn.init.kaiming_normal_(block[0].weight)
            nn.init.kaiming_normal_(block[3].weight) 
            nn.init.kaiming_normal_(block[6].weight)
            nn.init.constant_(block[0].bias, 0)
            nn.init.constant_(block[3].bias, 0)
            nn.init.constant_(block[6].bias, 0)
            
            # init layer norm
            nn.init.constant_(block[1].weight, 1)
            nn.init.constant_(block[4].weight, 1)
            nn.init.constant_(block[1].bias, 0)
            nn.init.constant_(block[4].bias, 0)

            self.blocks.append(block)  
  
            # Add quantization layer for this dimension level  
            if self.train_binary or self.train_two_bit:  
                quant_bits = 2 if self.train_two_bit else 1  
                self.quantization_layers[str(dim)] = QuantizationLayer(  
                    dim,  
                    quantization_bits=quant_bits,  
                    expand_two_bit_to_three_bits=self.expand_two_bit_to_three_bits  
                )  
            prev_dim = dim  
  
    def forward(self,  
                x: torch.Tensor,  
                apply_binary: bool = False,  
                apply_two_bit: bool = False) -> dict:  
        """  
        Forward pass to obtain nested embeddings.  
  
        Args:  
            x (torch.Tensor): Input embeddings of shape (batch_size, input_dim)  
            apply_binary (bool): Whether to apply binary quantization layers.  
            apply_two_bit (bool): Whether to apply 2-bit quantization layers.  
  
        Returns:  
            dict: Dictionary of embeddings at each dimension level.  
        """  
        embeddings = {}  
        x = self.base_transform(x)  
        prev_embedding = None  
        all_embeddings = x  
  
        for idx, block in enumerate(self.blocks):  
            
            delta = block(x)  
            if prev_embedding is None:
                prev_embedding = delta
            else:
                prev_embedding = torch.cat([prev_embedding, delta], dim=1)  
            all_embeddings = prev_embedding  
            dim = self.dimension_levels[idx]  
  
            # Normalize embeddings before quantization  
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)  
  
            # Apply quantization layer if required  
            if self.train_binary or self.train_two_bit:  
                quant_layer = self.quantization_layers[str(dim)]  
                all_embeddings = quant_layer(  
                    all_embeddings,  
                    training=self.training,  
                    quantize_binary=apply_binary,  
                    quantize_two_bit=apply_two_bit  
                )  
            embeddings[dim] = all_embeddings  
  
        embeddings_dict = {}  
        for dim, emb in embeddings.items():  
            embeddings_dict[dim] = emb  
        return embeddings_dict  
  
    def quantize_two_bits(self, embeddings: torch.Tensor, expand_to_three_bits: bool = False) -> torch.Tensor:  
        """  
        Quantize embeddings to 2 bits per dimension.  
  
        Args:  
            embeddings (torch.Tensor): Input embeddings.  
            expand_to_three_bits (bool): Whether to expand 2-bit codes to 3 bits using a codebook.  
  
        Returns:  
            torch.Tensor: Quantized embeddings.  
        """  
        # Assuming embeddings are already processed by QuantizationLayer  
        # If expand_to_three_bits is True, use the codebook to map 2-bit codes to 3 bits  
        if expand_to_three_bits:  
            # Implement the codebook expansion logic  
            # For simplicity, assuming codebook is predefined in QuantizationLayer  
            dim = embeddings.size(1)  
            quant_layer = self.quantization_layers[str(dim)]  
            embeddings = quant_layer.expand_to_three_bits(embeddings)  
        return embeddings  
  
class QuantizationLayer(nn.Module):  
    """  
    Quantization layer supporting both 1-bit and 2-bit quantization with temperature scaling and thresholding.  
  
    Args:  
        dim (int): Dimension of the input embeddings.  
        quantization_bits (int): Number of bits for quantization (1 or 2).  
        initial_temperature (float): Initial temperature for the sigmoid function.  
        min_temperature (float): Minimum temperature to anneal to.  
        expand_two_bit_to_three_bits (bool): Whether to expand 2-bit codes to 3 bits using a codebook.  
        annealing_rate (float): Decay rate for temperature annealing.  
    """  
    def __init__(self,  
                 dim: int,  
                 quantization_bits: int = 1,  
                 initial_temperature: float = 5.0,  
                 min_temperature: float = 0.1,  
                 expand_two_bit_to_three_bits: bool = False,  
                 annealing_rate: float = 0.95):  
        super(QuantizationLayer, self).__init__()  
        self.dim = dim  
        self.quantization_bits = quantization_bits  
        self.register_buffer('temperature', torch.tensor(initial_temperature))  
        self.min_temperature = min_temperature  
        self.expand_two_bit_to_three_bits = expand_two_bit_to_three_bits  
        self.annealing_rate = annealing_rate  
        self.scale = nn.Parameter(torch.ones(dim))  
  
        # Initialize thresholds for multi-bit quantization  
        self.thresholds = nn.Parameter(torch.zeros(dim, self.num_thresholds()))  
  
        # Codebook for expanding 2-bit codes to 3 bits  
        if self.quantization_bits == 2 and self.expand_two_bit_to_three_bits:  
            self.codebook = self.create_codebook()  
  
    def num_thresholds(self) -> int:  
        """  
        Compute the number of thresholds based on quantization bits.  
  
        Returns:  
            int: Number of thresholds.  
        """  
        return 2 ** self.quantization_bits - 1  
  
    def initialize_thresholds(self, sample_embeddings: torch.Tensor):  
        """  
        Initializes thresholds based on sample embeddings for improved convergence.  
  
        Args:  
            sample_embeddings (torch.Tensor): Sample embeddings, shape (num_samples, dim).  
        """  
        quantiles = torch.linspace(0, 1, self.num_thresholds() + 2)[1:-1]  
        thresholds = []  
        for d in range(self.dim):  
            values = sample_embeddings[:, d]  
            percentiles = torch.quantile(values, quantiles)  
            thresholds.append(percentiles)  
        thresholds = torch.stack(thresholds)  
        self.thresholds.data = thresholds  
  
    def create_codebook(self) -> dict:  
        """  
        Create a codebook mapping 2-bit quantization levels to 3-bit codes.  
  
        Returns:  
            dict: Codebook dictionary.  
        """  
        codebook = {  
            0: torch.tensor([0, 0, 0], dtype=torch.float32),  
            1: torch.tensor([0, 0, 1], dtype=torch.float32),  
            2: torch.tensor([0, 1, 1], dtype=torch.float32),  
            3: torch.tensor([1, 1, 1], dtype=torch.float32)  
        }  
        return codebook  
  
    def forward(self,  
                x: torch.Tensor,  
                training: bool = True,  
                quantize_binary: bool = False,  
                quantize_two_bit: bool = False) -> torch.Tensor:  
        """  
        Forward pass through the quantization layer.  
  
        Args:  
            x (torch.Tensor): Input embeddings.  
            training (bool): Indicates whether in training mode.  
            quantize_binary (bool): Whether to quantize to 1-bit.  
            quantize_two_bit (bool): Whether to quantize to 2-bits.  
  
        Returns:  
            torch.Tensor: Quantized embeddings.  
        """  
        x = x * self.scale  # Learnable scaling  
        if self.quantization_bits == 1 and quantize_binary:  
            # 1-bit quantization logic  
            x = x - self.thresholds.squeeze(-1)  # Subtract thresholds  
            if training:  
                x = torch.sigmoid(x / self.temperature)  
                # Apply Straight-Through Estimator (STE)  
                binary_x = (x > 0.5).float()  
                x = x + (binary_x - x).detach()  
            else:  
                x = (x > 0.0).float()  
        elif self.quantization_bits == 2 and quantize_two_bit:  
            # 2-bit quantization logic  
            x = self.multi_bit_quantization(x, training)  
        return x  
  
    def multi_bit_quantization(self, embeddings: torch.Tensor, training: bool = True) -> torch.Tensor:  
        """  
        Quantize embeddings to multiple bits per dimension using multi-threshold quantization.  
  
        Args:  
            embeddings (torch.Tensor): Input embeddings.  
            training (bool): Indicates whether in training mode.  
  
        Returns:  
            torch.Tensor: Quantized embeddings.  
        """  
        thresholds = self.thresholds  # Shape: (dim, num_thresholds)  
        # Enforce ordering constraints on thresholds by sorting  
        thresholds = torch.sort(thresholds, dim=1)[0]  # Shape: (dim, num_thresholds)  
  
        # Expand dimensions for broadcasting  
        embeddings_expanded = embeddings.unsqueeze(2)  # Shape: (batch_size, dim, 1)  
        thresholds_expanded = thresholds.unsqueeze(0)  # Shape: (1, dim, num_thresholds)  
  
        # Compute logits for sigmoid  
        k = 1.0 / self.temperature    # Temperature parameter controlling the steepness  
        logits = k * (embeddings_expanded - thresholds_expanded)  
        sigma = torch.sigmoid(logits)  # Shape: (batch_size, dim, num_thresholds)  
  
        # Compute probabilities for each quantization level  
        probs = [1.0 - sigma[:, :, 0]]  
        for i in range(self.num_thresholds() - 1):  
            probs.append(sigma[:, :, i] - sigma[:, :, i + 1])  
        probs.append(sigma[:, :, -1])  
  
        # Stack probabilities  
        probabilities = torch.stack(probs, dim=2)  # Shape: (batch_size, dim, num_levels)  
  
        # Quantization levels  
        levels = torch.arange(2 ** self.quantization_bits, dtype=torch.float32, device=embeddings.device)  
  
        # Compute weighted sum  
        quantized_embeddings = torch.einsum('bdk,k->bd', probabilities, levels)  
  
        if training:  
            # Apply Straight-Through Estimator (STE)  
            quantized_levels = torch.round(quantized_embeddings)  
            quantized_embeddings = quantized_embeddings + (quantized_levels - quantized_embeddings).detach()  
        else:  
            quantized_embeddings = torch.round(quantized_embeddings)  
        return quantized_embeddings  
  
    def expand_to_three_bits(self, quantized_embeddings: torch.Tensor) -> torch.Tensor:  
        """  
        Expand 2-bit quantized embeddings to 3-bit codes using the codebook.  
  
        Args:  
            quantized_embeddings (torch.Tensor): 2-bit quantized embeddings, shape (batch_size, dim).  
  
        Returns:  
            torch.Tensor: Expanded embeddings with 3 bits per dimension, shape (batch_size, dim * 3).  
        """  
        batch_size, dim = quantized_embeddings.size()  
        code_length = len(next(iter(self.codebook.values())))  
  
        # Map quantized levels to codes  
        quantized_levels = quantized_embeddings.long().view(-1)  
        codes = torch.stack([self.codebook[level.item()] for level in quantized_levels], dim=0)  
        codes = codes.view(batch_size, dim * code_length)  
        return codes  
  
    def anneal_temperature(self, current_epoch: int, total_epochs: int):  
        """  
        Anneal the temperature parameter over epochs.  
  
        Args:  
            current_epoch (int): Current training epoch.  
            total_epochs (int): Total number of training epochs.  
        """  
        # Exponential decay of temperature  
        progress = current_epoch / total_epochs  
        # Calculate the annealing rate to reach min_temperature at the end  
        annealing_rate = (self.min_temperature / self.initial_temperature) ** (1 / total_epochs)  
        # Update temperature  
        new_temperature = max(self.min_temperature, self.initial_temperature * (annealing_rate ** current_epoch))  
        self.temperature.copy_(torch.tensor(new_temperature))  
  
def multi_scale_contrastive_loss(embeddings_dict: dict,  
                                 positive_pairs: torch.Tensor,  
                                 temperature: float = 0.07,  
                                 weights: dict = None) -> torch.Tensor:  
    """  
    Compute the multi-scale contrastive loss over different embedding dimensions.  
  
    Args:  
        embeddings_dict (dict): Dictionary of embeddings at different dimensions.  
        positive_pairs (torch.Tensor): Tensor of positive pair indices, shape (batch_size // 2, 2)  
        temperature (float): Temperature parameter for contrastive loss.  
        weights (dict, optional): Weights for each dimension.  
  
    Returns:  
        torch.Tensor: Scalar loss value.  
    """  
    total_loss = 0.0  
    # Assign higher weight to lower dimensions  
    weights = weights or {dim: 1.0 / np.sqrt(idx) for idx, dim in enumerate(sorted(embeddings_dict.keys()), 1)}  
    for dim, embeddings in embeddings_dict.items():  
        # Normalize embeddings  
        embeddings = F.normalize(embeddings, p=2, dim=1)  
        # Construct positive and negative samples  
        batch_size = embeddings.size(0)  
        z_i = embeddings[::2]  # Even indices  
        z_j = embeddings[1::2]  # Odd indices  
  
        # Compute similarities  
        representations = torch.cat([z_i, z_j], dim=0)  
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)  
  
        # Create labels for contrastive loss  
        labels = torch.arange(0, batch_size // 2, device=embeddings.device).repeat(2)  
  
        # Mask to remove self-similarity  
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=embeddings.device)  
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)  
  
        # Compute loss  
        loss = F.cross_entropy(similarity_matrix / temperature, labels, reduction='mean')  
        total_loss += weights[dim] * loss  
    return total_loss  
  
def quantization_regularization_loss(embeddings_dict: dict,  
                                     quantization_bits: int,  
                                     current_epoch: int,  
                                     total_epochs: int) -> torch.Tensor:  
    """  
    Compute regularization loss to encourage quantized outputs.  
  
    Args:  
        embeddings_dict (dict): Dictionary of embeddings at different dimensions.  
        quantization_bits (int): Number of bits used for quantization (1 or 2).  
        current_epoch (int): Current training epoch.  
        total_epochs (int): Total number of training epochs.  
  
    Returns:  
        torch.Tensor: Scalar regularization loss.  
    """  
    reg_loss = 0.0  
    progress = current_epoch / total_epochs  # Progress ratio (0 to 1)  
    weight = progress  # Increase weight over time  
    for emb in embeddings_dict.values():  
        if quantization_bits == 1:  
            # Binary quantization regularization  
            bce_loss = F.binary_cross_entropy(emb, torch.ones_like(emb) * 0.5)  
            l1_loss = torch.mean(torch.abs(emb * (1 - emb)))  
            reg_loss += weight * (bce_loss + l1_loss)  
        elif quantization_bits == 2:  
            # Multi-bit quantization regularization  
            # Encourage embeddings to be close to integer levels  
            levels = torch.round(emb)  
            mse_loss = F.mse_loss(emb, levels)  
            reg_loss += weight * mse_loss  
    return reg_loss  


def kl_divergence(embeddings_original: torch.Tensor, embeddings_new: torch.Tensor) -> torch.Tensor:
    """
    Compute symmetric KL divergence (Jensen-Shannon Divergence) between original and new embeddings.
    
    Args:
        embeddings_original (torch.Tensor): Original embeddings
        embeddings_new (torch.Tensor): New embeddings
        
    Returns:
        torch.Tensor: Mean JSD value
    """
    # Convert embeddings to probability distributions using softmax
    prob_original = F.softmax(embeddings_original, dim=-1)
    prob_new = F.softmax(embeddings_new, dim=-1)
    
    # Compute KL divergence in both directions
    # Note: kl_div expects log probabilities as first argument
    kl_div = F.kl_div(
        prob_original.log(), 
        prob_new,
        reduction='batchmean',
        log_target=False
    )
    
    kl_div_2 = F.kl_div(
        prob_new.log(),
        prob_original,
        reduction='batchmean',
        log_target=False
    )
    
    # Compute Jensen-Shannon Divergence
    jsd = 0.5 * (kl_div + kl_div_2)
    
    return jsd
  
def orthogonality_regularization(embeddings_dict: dict) -> torch.Tensor:  
    """  
    Compute orthogonality regularization to encourage uniqueness in added dimensions.  
  
    Args:  
        embeddings_dict (dict): Dictionary of embeddings at different dimensions.  
  
    Returns:  
        torch.Tensor: Scalar regularization loss.  
    """  
    reg_loss = 0.0  
    dimensions = sorted(embeddings_dict.keys())  
    for i in range(1, len(dimensions)):  
        dim_prev = dimensions[i - 1]  
        dim_curr = dimensions[i]  
        emb_prev = embeddings_dict[dim_prev]  # Shape: (batch_size, dim_prev)  
        emb_curr = embeddings_dict[dim_curr]  # Shape: (batch_size, dim_curr)  
        # Extract the new dimensions added in emb_curr  
        delta = emb_curr[:, dim_prev:]  # Shape: (batch_size, dim_curr - dim_prev)  
        # Compute the dot product between delta and emb_prev  
        dot_product = torch.matmul(delta.T, emb_prev)  # Shape: (dim_curr - dim_prev, dim_prev)  
        # Compute Frobenius norm of the dot product matrix  
        reg_loss += torch.norm(dot_product, p='fro')  
    return reg_loss  
  
def information_bottleneck_regularization(embeddings_dict: dict) -> torch.Tensor:  
    """  
    Compute information bottleneck regularization to focus critical information in lower dimensions.  
    Applies progressively stronger L1 regularization to higher dimensions to encourage sparsity.  
  
    Args:  
        embeddings_dict (dict): Dictionary of embeddings at different dimensions.  
  
    Returns:  
        torch.Tensor: Scalar regularization loss.  
    """  
    reg_loss = 0.0  
    dimensions = sorted(embeddings_dict.keys())  
    # Create increasing weights for higher dimensions  
    weights = {dim: np.sqrt(idx) for idx, dim in enumerate(dimensions, 1)}  
    for dim in dimensions[1:]:  # Skip the smallest dimension  
        emb = embeddings_dict[dim]  
        # Apply weighted L1 regularization - higher dimensions get stronger regularization  
        reg_loss += weights[dim] * torch.mean(torch.abs(emb))  
    return reg_loss  
  
def train_matryoshka_model(matryoshka_model: MatryoshkaEmbeddingModel,  
                           dataloader,  
                           num_epochs: int = 5,  
                           learning_rate: float = 1e-4,  
                           temperature: float = 0.07,  
                           reg_strength: float = 1e-3):  
    """  
    Train the Matryoshka Embedding Model.  
  
    Args:  
        matryoshka_model (MatryoshkaEmbeddingModel): The model to train.  
        dataloader (DataLoader): DataLoader for the dataset.  
        num_epochs (int): Number of training epochs.  
        learning_rate (float): Learning rate for the optimizer.  
        temperature (float): Temperature parameter for contrastive loss.  
        reg_strength (float): Regularization strength.  
    """  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    matryoshka_model.transformer.to(device)  
    matryoshka_model.embedding_model.to(device)  
    optimizer = torch.optim.Adam(matryoshka_model.transformer.parameters(), lr=learning_rate)  
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,  
                                                    epochs=num_epochs, steps_per_epoch=len(dataloader), final_div_factor=100)  
    matryoshka_model.embedding_model.eval()  
    matryoshka_model.transformer.train()  
  
    # Collect sample embeddings for threshold initialization  
    sample_embeddings_list = []  
    if matryoshka_model.train_binary or matryoshka_model.train_two_bit:
        with torch.no_grad():  
            for batch in tqdm(dataloader, desc="Collecting sample embeddings", total=len(dataloader)):  
                input_ids = batch['input_ids'].squeeze(1).to(device)  
                attention_mask = batch['attention_mask'].squeeze(1).to(device)  
                embeddings = matryoshka_model.embedding_model(  
                    input_ids=input_ids,  
                    attention_mask=attention_mask,  
                )
                # convert to tensor if not already
                embeddings = torch.tensor(embeddings)
                embeddings = F.normalize(embeddings, p=2, dim=1)  
                sample_embeddings_list.append(embeddings)  
                if len(sample_embeddings_list) >= 100:  
                    break  
            sample_embeddings = torch.cat(sample_embeddings_list, dim=0)  
    
        # Initialize thresholds in quantization layers  
        matryoshka_model.init_thresholds(sample_embeddings)  
  
    
    for epoch in range(num_epochs):  
        total_loss = 0.0  
        loss_dict = {}
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:  
            input_ids = batch['input_ids'].squeeze(1).to(device)  
            attention_mask = batch['attention_mask'].squeeze(1).to(device)  
            with torch.no_grad():  
                embeddings = matryoshka_model.embedding_model(  
                    input_ids=input_ids,  
                    attention_mask=attention_mask,  
                )
                # convert to tensor if not already
                embeddings = torch.tensor(embeddings)
                embeddings = F.normalize(embeddings, p=2, dim=1)  
            embeddings = embeddings.to(device)  
            # Forward pass through transformer  
            embeddings_dict = matryoshka_model.transformer(embeddings)  
            batch_size = embeddings.size(0)  
            positive_pairs = torch.arange(0, batch_size, device=device).view(-1, 2)  
            # Compute losses 
            # loss_contrastive = multi_scale_contrastive_loss(  
            #     embeddings_dict, positive_pairs, temperature=temperature  
            # )  
            # loss_dict['contrastive'] = loss_contrastive.item()
            loss_ortho = 0.01 * orthogonality_regularization(embeddings_dict)  
            loss_dict['ortho'] = loss_ortho.item()
            loss_info_bottleneck = 0.1 * information_bottleneck_regularization(embeddings_dict)  
            loss_dict['info_bottleneck'] = loss_info_bottleneck.item()
            loss_similarity = 0.0  
            loss_kl_similarity = 0.0
            overall_contrastive_loss = 0.0
            dimension_levels = sorted(embeddings_dict.keys())  
            overall_rank_loss = 0.0
            scale_factor = (max(dimension_levels) / min(dimension_levels)) / np.sqrt(max(dimension_levels))
            for dim, emb in embeddings_dict.items():  
                weight_small_dim = 1.0 / (dim / min(dimension_levels))  
                weight_large_dim = scale_factor * np.sqrt(dim)
                loss_similarity += weight_large_dim * similarity_preservation_loss(embeddings, emb)  
                loss_kl_similarity += weight_large_dim * kl_similarity_preservation_loss(embeddings, emb)
                contrastive_loss_per_dim = weight_large_dim * contrastive_loss(embeddings)
                overall_contrastive_loss += contrastive_loss_per_dim
                overall_rank_loss += weight_large_dim * rank_preserving_loss(embeddings, emb)
                
            highest_dim = max(dimension_levels)
            embeddings_new = embeddings_dict[highest_dim]
            embeddings_original = embeddings
            # loss_kl = kl_divergence(embeddings_original, embeddings_new)
            # loss_dict['kl'] = loss_kl.item()
            loss = loss_similarity + loss_kl_similarity + overall_rank_loss # + 0.1 * overall_contrastive_loss # + reg_strength * loss_ortho + reg_strength * loss_info_bottleneck + # + loss_kl
            loss_dict['similarity'] = loss_similarity.item()
            loss_dict['kl_similarity'] = loss_kl_similarity.item()
            loss_dict['total'] = loss.item()
            loss_dict['contrastive'] = overall_contrastive_loss.item()
            loss_dict['rank'] = overall_rank_loss.item()
            # If training for quantization, add quantization regularization loss  
            if matryoshka_model.train_binary or matryoshka_model.train_two_bit:  
                quantization_bits = 2 if matryoshka_model.train_two_bit else 1  
                loss_quantization = quantization_regularization_loss(  
                    embeddings_dict, quantization_bits, epoch, num_epochs  
                )  
                loss_dict['quantization'] = loss_quantization.item()
                loss += reg_strength * loss_quantization  
                
                # Anneal temperatures in quantization layers  
                for quant_layer in matryoshka_model.transformer.quantization_layers.values():  
                    quant_layer.anneal_temperature(epoch, num_epochs)  
            optimizer.zero_grad()  
            loss.backward()  
            # Add gradient clipping before optimizer step  
            nn.utils.clip_grad_norm_(matryoshka_model.transformer.parameters(), max_grad_norm)  
            optimizer.step()  
            scheduler.step()  
            total_loss += loss.item()  
            pbar.set_postfix(loss_dict)
        avg_loss = total_loss / len(dataloader)  
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')  
        loss_dict['avg_loss'] = avg_loss
    return matryoshka_model