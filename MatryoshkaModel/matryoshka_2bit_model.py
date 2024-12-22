from ast import For
from sympy import prem, trailing
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


class CustomizedMatryoshkaEmbeddingModel(OriginalEmbeddingCaller):
    def __init__(self, embedding_model: OriginalEmbeddingCaller, dimension_levels: List[int], two_bits: int, one_and_half_bits: int, one_bits: int, half_bits: int, expand: bool = False):
        # USe a skip connection for the half bits part to create a better embedding
        # Compare 2 bit to 3 bit to just expanding to 3 bits at the end and then using binary quantization
        pass
        
class MatryoshkaEmbeddingModel(OriginalEmbeddingCaller):  
    """  
    Matryoshka Embedding Model that produces nested embeddings with optional binary,  
    multi-bit (including 1.5-bit), and two-bit quantization.  
  
    Attributes:  
        embedding_model_name (str): Name of the pretrained embedding model.  
        dimension_levels (List[int]): List of embedding dimensions in increasing order.  
        quantization_mode (str): Quantization mode ('binary', '1.5bit', '2bit', or None).  
        expand_quantized_bits (bool): Whether to expand quantized codes using a codebook.  
    """  
    def __init__(self,  
                 embedding_model: OriginalEmbeddingCaller,  
                 dimension_levels: List[int],  
                 train_binary: bool = False,  
                 train_two_bit: bool = False, 
                 train_one_and_half_bit: bool = False,
                 expand_two_bit_to_three_bits: bool = False,
                 expand_one_and_half_bit_to_two_bits: bool = False):  
        """  
        Initialize the Matryoshka Embedding Model.  
  
        Args:  
            embedding_model (OriginalEmbeddingCaller): Pretrained embedding model.  
            dimension_levels (List[int]): List of embedding dimensions in increasing order.  
            quantization_mode (str): Quantization mode ('binary', '1.5bit', '2bit', or None).  
            expand_quantized_bits (bool): Whether to expand quantized codes using a codebook.  
        """  
        super().__init__(embedding_model.model_name, embedding_model.embedding_dim)  
        self.embedding_model = embedding_model  
        self.dimension_levels = sorted(dimension_levels)  
  
        # Set quantization mode based on training flags
        if train_two_bit:
            self.quantization_mode = '2bit'
        elif train_one_and_half_bit:
            self.quantization_mode = '1.5bit'
        elif train_binary:
            self.quantization_mode = 'binary'
        else:
            self.quantization_mode = None
        self.expand_quantized_bits = expand_two_bit_to_three_bits or expand_one_and_half_bit_to_two_bits  
        
        self.train_binary = train_binary  
        self.train_two_bit = train_two_bit  
        self.train_one_and_half_bit = train_one_and_half_bit
        self.expand_two_bit_to_three_bits = expand_two_bit_to_three_bits
        self.expand_one_and_half_bit_to_two_bits = expand_one_and_half_bit_to_two_bits
        
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
            train_one_and_half_bit=self.train_one_and_half_bit,
            expand_two_bit_to_three_bits=self.expand_two_bit_to_three_bits,
            expand_one_and_half_bit_to_two_bits=self.expand_one_and_half_bit_to_two_bits
        )  
        self.max_dim = self.embedding_dim  
        self.baseline = False  
  
    def save(self, path: str):  
        torch.save(self.transformer.state_dict(), path)  
  
    def load(self, path: str):  
        self.transformer.load_state_dict(torch.load(path, map_location=torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')))  
  
    def calculate_thresholds(self, sample_embeddings: torch.Tensor):  
        """  
        Calculate thresholds based on sample embeddings.  
        """  
        thresholds = {}  
        for idx, quant_layer in self.transformer.quantization_layers.items():  
            if quant_layer.quantization_levels > 1:  
                thresholds[idx] = quant_layer.calculate_thresholds(sample_embeddings)  
        return thresholds  
  
    def init_thresholds(self, sample_embeddings: dict):  
        """  
        Initialize thresholds in quantization layers based on sample embeddings.  
  
        Args:  
            sample_embeddings (torch.Tensor): Sample embeddings used for initializing thresholds.  
        """  
        for dim, quant_layer in self.transformer.quantization_layers.items():  
            if quant_layer.quantization_levels >= 1:  
                quant_layer.initialize_thresholds(sample_embeddings[int(dim)])  
  
    def encode(self,  
               sentences: List[str],  
               output_dim: int = matryoshka_output_dim,  
               **kwargs) -> np.ndarray:  
        """  
        Encode sentences to obtain embeddings.  
  
        Args:  
            sentences (List[str]): List of sentences to encode.  
            output_dim (int, optional): Desired embedding dimension. If None, use max dimension.  
            **kwargs: Additional arguments for the embedding model.  
  
        Returns:  
            np.ndarray: Embeddings of shape (num_sentences, output_dim)  
        """  
        if output_dim is None:  
            output_dim = self.max_dim  
        assert output_dim in self.dimension_levels, f"Output dimension must be one of {self.dimension_levels}"  
  
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
        baseline_embeddings = base_embeddings[:, :output_dim]  
        if self.baseline:  
            embeddings = baseline_embeddings  
            non_quant_embeddings = baseline_embeddings  
        else:  
            self.transformer.eval()  
            with torch.no_grad():  
                embeddings_dict, non_quant_embeddings = self.transformer(  
                    base_embeddings  
                )  
                embeddings = embeddings_dict[output_dim]  
                non_quant_embeddings = non_quant_embeddings[output_dim]  
  
        # If quantization mode is specified, adjust embeddings accordingly  
        if self.quantization_mode == 'binary' and self.baseline:  
            embeddings = (embeddings > 0.0).float()  
        elif self.quantization_mode in ['1.5bit', '2bit']:  
            # Quantize embeddings  
            embeddings = embeddings  
            # Adjust output dimension if codes are expanded  
            if self.expand_quantized_bits:  
                if self.quantization_mode == '1.5bit':  
                    expansion_factor = 2  # 2 bits per dimension after expansion  
                elif self.quantization_mode == '2bit':  
                    expansion_factor = 3  # 3 bits per dimension after expansion  
                expected_dim = output_dim * expansion_factor  
                assert embeddings.shape[1] == expected_dim, f"Output dimension {embeddings.shape[1]} does not match expected dimension {expected_dim}"  
        else:  
            # Normalize embeddings  
            embeddings = F.normalize(embeddings, p=2, dim=1)  
        return embeddings.cpu().numpy()  

 

class SliceLayer(nn.Module):
    def __init__(self, start_dim, end_dim):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        
    def forward(self, x):
        return x[:, self.start_dim:self.end_dim]
  
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
                 train_one_and_half_bit: bool = False,
                 expand_two_bit_to_three_bits: bool = False,
                 expand_one_and_half_bit_to_two_bits: bool = False):  
        """  
        Initialize the Transformer.  
  
        Args:  
            input_dim (int): Dimension of the input embeddings.  
            dimension_levels (List[int]): List of embedding dimensions in increasing order.  
            train_binary (bool): Whether to include binary quantization layers.  
            train_two_bit (bool): Whether to include 2-bit quantization layers.  
            train_one_and_half_bit (bool): Whether to include 1.5-bit quantization layers.  
            expand_two_bit_to_three_bits (bool): Whether to expand 2-bit codes to 3 bits.  
            expand_one_and_half_bit_to_two_bits (bool): Whether to expand 1.5-bit codes to 2 bits.  
        """  
        super(MatryoshkaTransformer, self).__init__()  
        self.dimension_levels = sorted(dimension_levels)  
        self.train_binary = train_binary  
        self.train_two_bit = train_two_bit  
        self.train_one_and_half_bit = train_one_and_half_bit
        self.expand_two_bit_to_three_bits = expand_two_bit_to_three_bits  
        self.expand_one_and_half_bit_to_two_bits = expand_one_and_half_bit_to_two_bits
        
        self.quantization_mode = '2bit' if self.train_two_bit else '1.5bit' if self.train_one_and_half_bit else 'binary' if self.train_binary else None
        self.expand_quantized_bits = self.expand_two_bit_to_three_bits or self.expand_one_and_half_bit_to_two_bits
  
        self.blocks = nn.ModuleList()  
        self.quantization_layers = nn.ModuleDict()  
        prev_dim = 0  
  
        assert input_dim >= self.dimension_levels[-1], "Input dimension must be greater than or equal to the largest dimension level"  
        if input_dim > self.dimension_levels[-1]:  
            self.dimension_levels.append(input_dim)  
            
        assert input_dim == self.dimension_levels[-1], "Input dimension must be equal to the largest dimension level"
  
        
        
        # self.base_transform = create_modern_mlp(
        #     input_dim=input_dim,
        #     hidden_dims=[input_dim * 4, input_dim * 4],
        #     output_dim=input_dim * 4,
        #     dropout_rate=0.1,
        #     negative_slope=0.01,
        #     use_skip_connections=True
        # )
        
        if use_rms_norm:
        
            self.base_transform = nn.Sequential(nn.Linear(input_dim, input_dim*32),
                                                # nn.LayerNorm(input_dim*32),
                                                nn.GELU(), 
                                                nn.RMSNorm(input_dim*32, eps=1e-6),
                                                # nn.Linear(input_dim*32, input_dim*32),
                                                # nn.GELU(),
                                                nn.Linear(input_dim*32, input_dim))
            
            # init base transform
            nn.init.kaiming_normal_(self.base_transform[0].weight)
            nn.init.kaiming_normal_(self.base_transform[3].weight)
            
            nn.init.constant_(self.base_transform[0].bias, 0)
            nn.init.constant_(self.base_transform[3].bias, 0)
        else:
            self.base_transform = nn.Sequential(nn.Linear(input_dim, input_dim*32),
                                                nn.GELU(), 
                                                nn.Linear(input_dim*32, input_dim))
            
            # init base transform
            nn.init.kaiming_normal_(self.base_transform[0].weight)
            nn.init.kaiming_normal_(self.base_transform[2].weight)
            
            nn.init.constant_(self.base_transform[0].bias, 0)
            nn.init.constant_(self.base_transform[2].bias, 0)
        
        
  
        for dim in self.dimension_levels:  
            
            
            
                    
            block = nn.Sequential(
                nn.Identity(),
                SliceLayer(prev_dim, dim)
            )

            self.blocks.append(block)  
  
            # Add quantization layer for this dimension level  
            if self.quantization_mode in ['binary', '1.5bit', '2bit']:  
                if self.quantization_mode == 'binary':  
                    quantization_levels = 2  
                elif self.quantization_mode == '1.5bit':  
                    quantization_levels = 3  
                elif self.quantization_mode == '2bit':  
                    quantization_levels = 4  
                self.quantization_layers[str(dim)] = QuantizationLayer(  
                    dim,  
                    quantization_levels=quantization_levels,  
                    expand_quantized_bits=self.expand_quantized_bits  
                )
            prev_dim = dim  
  
    def forward(self,  
                x: torch.Tensor,  
                apply_binary: bool = False,  
                apply_two_bit: bool = False,
                apply_one_and_half_bit: bool = False) -> dict:  
        """  
        Forward pass to obtain nested embeddings.  
  
        Args:  
            x (torch.Tensor): Input embeddings of shape (batch_size, input_dim)  
            apply_binary (bool): Whether to apply binary quantization layers.  
            apply_two_bit (bool): Whether to apply 2-bit quantization layers.  
            apply_one_and_half_bit (bool): Whether to apply 1.5-bit quantization layers.  
        Returns:  
            dict: Dictionary of embeddings at each dimension level.  
        """  
        embeddings = {}  
        non_quant_embeddings = {}
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
            non_quant_embeddings[dim] = all_embeddings
            # Apply quantization layer if required  
            if self.quantization_mode in ['binary', '1.5bit', '2bit']:  
                quant_layer = self.quantization_layers[str(dim)]  
                all_embeddings = quant_layer(  
                    all_embeddings,  
                    training=self.training,  
                )  
            elif apply_two_bit:
                raise NotImplementedError("Two bit quantization is not supported without training")
                # all_embeddings = self.quantize_two_bits(all_embeddings, self.expand_two_bit_to_three_bits)
            elif apply_binary:
                all_embeddings = (all_embeddings > 0.0).float()
            embeddings[dim] = all_embeddings  
  
        embeddings_dict = {}  
        for dim, emb in embeddings.items():  
            embeddings_dict[dim] = emb  
        return embeddings_dict, non_quant_embeddings

  
class QuantizationLayer(nn.Module):  
    """  
    Quantization layer supporting binary, 1.5-bit (3-level), and 2-bit quantization  
    with temperature scaling and thresholding.  
  
    Args:  
        dim (int): Dimension of the input embeddings.  
        quantization_levels (int): Number of quantization levels (2, 3, or 4).  
        initial_temperature (float): Initial temperature for the sigmoid function.  
        min_temperature (float): Minimum temperature to anneal to.  
        expand_quantized_bits (bool): Whether to expand quantized codes using a codebook.  
        annealing_rate (float): Decay rate for temperature annealing.  
    """  
    def __init__(self,  
                 dim: int,  
                 quantization_levels: int = 2,  
                 initial_temperature: float = 5.0,  
                 min_temperature: float = 0.1,  
                 expand_quantized_bits: bool = False,  
                 annealing_rate: float = 0.95):  
        super(QuantizationLayer, self).__init__()  
        self.dim = dim  
        self.quantization_levels = quantization_levels  # 2, 3, or 4 levels  
        self.register_buffer('temperature', torch.tensor(initial_temperature))  
        self.min_temperature = min_temperature  
        self.expand_quantized_bits = expand_quantized_bits  
        self.annealing_rate = annealing_rate  
        self.initial_temperature = initial_temperature  
  
        # Initialize thresholds for multi-level quantization  
        self.register_buffer('thresholds', torch.zeros(dim, self.num_thresholds()))  
        self.register_buffer('thresholds_0_100', torch.zeros(dim, 2))  
  
        # Codebook for expanding quantized codes  
        if self.quantization_levels == 3 and self.expand_quantized_bits:  
            self.codebook = {  
                0: torch.tensor([0, 0], dtype=torch.float32),  
                1: torch.tensor([0, 1], dtype=torch.float32),  
                2: torch.tensor([1, 1], dtype=torch.float32),  
            }  
        elif self.quantization_levels == 4 and self.expand_quantized_bits:  
            self.codebook = {  
                0: torch.tensor([0, 0, 0], dtype=torch.float32),  
                1: torch.tensor([0, 0, 1], dtype=torch.float32),  
                2: torch.tensor([0, 1, 1], dtype=torch.float32),  
                3: torch.tensor([1, 1, 1], dtype=torch.float32)  
            }  
  
    def num_thresholds(self) -> int:  
        """  
        Compute the number of thresholds based on quantization levels.  
  
        Returns:  
            int: Number of thresholds.  
        """  
        return self.quantization_levels - 1  
  
    def calculate_thresholds(self, sample_embeddings: torch.Tensor):  
        """  
        Calculate thresholds based on sample embeddings.  
        """  
        quantiles = torch.linspace(0, 1, self.num_thresholds() + 2, device=sample_embeddings.device)[1:-1]  
        full_range_quantiles = torch.tensor([0.0, 1.0], device=sample_embeddings.device)  
  
        thresholds = []  
        thresholds_0_100 = []  
        for d in range(self.dim):  
            values = sample_embeddings[:, d]  
            percentiles = torch.quantile(values, quantiles)  
            thresholds.append(percentiles)  
            thresholds_0_100.append(torch.quantile(values, full_range_quantiles))  
        thresholds = torch.stack(thresholds)  
        thresholds_0_100 = torch.stack(thresholds_0_100)  
        return thresholds, thresholds_0_100  
  
    def initialize_thresholds(self, sample_embeddings: torch.Tensor):  
        """  
        Initializes thresholds based on sample embeddings for improved convergence.  
  
        Args:  
            sample_embeddings (torch.Tensor): Sample embeddings, shape (num_samples, dim).  
        """  
        thresholds, full_range_thresholds = self.calculate_thresholds(sample_embeddings)  
        self.thresholds.data = thresholds.to(self.thresholds.device)  
        self.thresholds_0_100.data = full_range_thresholds.to(self.thresholds_0_100.device)  
        
    def update_thresholds(self, batch_embeddings: torch.Tensor, momentum: float = 0.995):
        """
        Update thresholds based on the current batch statistics.
        Can be called after each batch to maintain adaptive thresholds.
        
        Args:
            batch_embeddings (torch.Tensor): Current batch embeddings.
        """
        thresholds, full_range_thresholds = self.calculate_thresholds(batch_embeddings)
        self.thresholds.data = momentum * self.thresholds.data + (1 - momentum) * thresholds.to(self.thresholds.device)  
        self.thresholds_0_100.data = momentum * self.thresholds_0_100.data + (1 - momentum) * full_range_thresholds.to(self.thresholds_0_100.device)
  
    def forward(self,  
                x: torch.Tensor,  
                training: bool = True) -> torch.Tensor:  
        """  
        Forward pass through the quantization layer.  
  
        Args:  
            x (torch.Tensor): Input embeddings.  
            training (bool): Indicates whether in training mode.  
  
        Returns:  
            torch.Tensor: Quantized embeddings.  
        """  
        if self.quantization_levels == 2:  
            # Binary quantization  
            return self.binary_quantization(x, training)  
        else:  
            # Multi-level quantization (1.5-bit or 2-bit)  
            return self.multi_level_quantization(x, training)  
  
    def binary_quantization(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:  
        """  
        Perform binary quantization on the embeddings.  
  
        Args:  
            x (torch.Tensor): Input embeddings.  
            training (bool): Indicates whether in training mode.  
  
        Returns:  
            torch.Tensor: Quantized embeddings.  
        """  
        threshold = self.thresholds.squeeze(-1)  # Shape: (dim,)  
        assert threshold.dim() == 1, "Expected 1D threshold tensor for binary quantization"  
  
        if training:  
            with torch.no_grad():  
                quantized = torch.zeros_like(x)  
                mask_1 = (x > threshold)  
                quantized[mask_1] = 1.0  
  
            grad_mask = (x > threshold).float()  
            x = quantized.detach() + (grad_mask - grad_mask.detach())  
        else:  
            x = (x > threshold).float()  
        return x  
  
    def multi_level_quantization(self, embeddings: torch.Tensor, training: bool = True) -> torch.Tensor:  
        """  
        Perform multi-level quantization on the embeddings.  
  
        Args:  
            embeddings (torch.Tensor): Input embeddings.  
            training (bool): Indicates whether in training mode.  
  
        Returns:  
            torch.Tensor: Quantized embeddings.  
        """  
        thresholds = torch.sort(self.thresholds, dim=1)[0]  # Shape: (dim, num_thresholds)  
  
        quantized = torch.zeros_like(embeddings)  
  
        if training:  
            with torch.no_grad():  
                num_levels = self.quantization_levels  
                masks = []  
                # Create masks for each level  
                for level in range(num_levels):  
                    if level == 0:  
                        mask = (embeddings <= thresholds[:, 0])  
                    elif level == num_levels - 1:  
                        mask = (embeddings > thresholds[:, -1])  
                    else:  
                        mask = (embeddings > thresholds[:, level - 1]) & (embeddings <= thresholds[:, level])  
                    masks.append(mask)  
                    quantized[mask] = level  
  
            # Create gradient mask  
            grad_mask = torch.zeros_like(embeddings, requires_grad=True)  
            for level, mask in enumerate(masks):  
                grad_mask = grad_mask + level * mask.float()
  
            quantized = quantized.detach() + (grad_mask - grad_mask.detach())  
        else:  
            num_levels = self.quantization_levels  
            for level in range(num_levels):  
                if level == 0:  
                    mask = (embeddings <= thresholds[:, 0])  
                elif level == num_levels - 1:  
                    mask = (embeddings > thresholds[:, -1])  
                else:  
                    mask = (embeddings > thresholds[:, level - 1]) & (embeddings <= thresholds[:, level])  
                quantized[mask] = level  
  
        if self.expand_quantized_bits:  
            quantized = self.expand_codes(quantized)  
  
        return quantized  
  
    def expand_codes(self, quantized_embeddings: torch.Tensor) -> torch.Tensor:  
        """  
        Expand quantized embeddings using the codebook.  
  
        Args:  
            quantized_embeddings (torch.Tensor): Quantized embeddings.  
  
        Returns:  
            torch.Tensor: Expanded embeddings.  
        """  
        batch_size, dim = quantized_embeddings.size()  
        code_length = len(next(iter(self.codebook.values())))  
  
        # Map quantized levels to codes  
        quantized_levels = quantized_embeddings.long().view(-1)  
        codes = torch.stack([self.codebook[level.item()] for level in quantized_levels], dim=0)  
        codes = codes.view(batch_size, dim * code_length)  
        return codes  
  
    def anneal_temperature(self, current_epoch: int, total_epochs: int, num_current_step: int, num_total_steps: int, steps_per_epoch: int):  
        """  
        Anneal the temperature parameter over steps, stopping at the second-to-last epoch.  
  
        Args:  
            current_epoch (int): Current training epoch.  
            total_epochs (int): Total number of training epochs.  
            num_current_step (int): Current step in the training process.  
            num_total_steps (int): Total number of steps in the training process.  
            steps_per_epoch (int): Number of steps per epoch.  
        """  
        # Stop annealing at second-to-last epoch  
        steps_remaining = num_total_steps - num_current_step  
        if steps_remaining <= 100:  
            return  
  
        # Calculate steps until penultimate epoch  
        steps_until_penultimate = num_total_steps - 100  
  
        # Calculate the annealing rate per step to reach min_temperature by penultimate epoch  
        step_annealing_rate = (self.min_temperature / self.initial_temperature) ** (1 / steps_until_penultimate)  
  
        # Update temperature based on current step  
        new_temperature = max(  
            self.min_temperature,  
            self.initial_temperature * (step_annealing_rate ** num_current_step)  
        )  
  
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


def increase_std_dev_over_time_loss(embeddings_dict: dict,  
                                   current_epoch: int,  
                                   total_epochs: int,
                                   num_current_step: int,
                                   num_total_steps: int,
                                   num_steps_per_epoch: int) -> dict:
    """
    Increase the standard deviation of embeddings over time.
    """
    progress = max(0.2, (np.exp(num_current_step / num_total_steps) - 1) / (np.e - 1))  # Progress ratio (0 to 1)
    weight = progress  # Increase weight over time  
    overall_loss = 0.0
    for dim in embeddings_dict:
        emb = embeddings_dict[dim] # emb shape (batch_size, dim)
        # calculate std dev of each dimension
        std_dev = torch.std(emb, dim=0)
        # calculate loss as the sum of std devs
        overall_std_dev = std_dev.mean()
        loss = torch.exp(-overall_std_dev)
        overall_loss += loss.sum()
    return weight * overall_loss

def pull_close_to_quant_levels_loss(embeddings_dict: dict,  
                                   quantization_bits: int,  
                                   current_epoch: int,  
                                   total_epochs: int,
                                   num_current_step: int,
                                   num_total_steps: int,
                                   num_steps_per_epoch: int,
                                   thresholds: dict, 
                                   thresholds_0_100: dict) -> torch.Tensor:  
    """
    Pull embeddings close to quantization levels
    """
    reg_loss = 0.0  
    progress = max(0.2, (np.exp(num_current_step / num_total_steps) - 1) / (np.e - 1))  # Progress ratio (0 to 1)
    weight = progress  # Increase weight over time  
    
    loss = 0.0
    dims = sorted(embeddings_dict.keys())
    

    for dim in dims:
        emb = embeddings_dict[dim]
        if quantization_bits == 1:
            # We also want values to be close to 0 or 1
            distance_to_levels = (torch.minimum(
                torch.abs(emb), 
                torch.abs(emb - 1.0)
            ) ** 2).mean()
            
        elif quantization_bits == 2:
            # We also want values to be close to 0, 1, 2, 3
            # Distance to nearest quantization level
            levels = torch.tensor([0., 1., 2., 3.], device=emb.device)
            level_distances = torch.abs(emb.unsqueeze(-1) - levels)
            distance_to_nearest_level = (level_distances.min(dim=-1)[0] ** 2).mean()
            
        loss += distance_to_nearest_level
        
    return weight * loss
def quantization_regularization_loss(embeddings_dict: dict,  
                                   quantization_bits: int,  
                                   current_epoch: int,  
                                   total_epochs: int,
                                   num_current_step: int,
                                   num_total_steps: int,
                                   num_steps_per_epoch: int,
                                   thresholds: dict, 
                                   thresholds_0_100: dict) -> torch.Tensor:  
    """  
    Compute regularization loss to encourage values to be far from thresholds
    and close to quantization levels.
  
    Args:  
        embeddings_dict (dict): Dictionary of embeddings at different dimensions.  
        quantization_bits (int): Number of bits used for quantization (1 or 2).  
        current_epoch (int): Current training epoch.  
        total_epochs (int): Total number of training epochs.  
        thresholds (torch.Tensor): Quantization thresholds.
  
    Returns:  
        torch.Tensor: Scalar regularization loss.  
    """  
    reg_loss = 0.0  
    progress = max(0.2, (np.exp(num_current_step / num_total_steps) - 1) / (np.e - 1))  # Progress ratio (0 to 1)
    weight = progress  # Increase weight over time  
    
    dims = sorted(embeddings_dict.keys())
    
    for dim in dims:
        emb = embeddings_dict[dim]
        threshold = thresholds[dim] # shape (dim, 1)
        threshold_0_100 = thresholds_0_100[dim] # shape (dim, 2)
        if quantization_bits == 1:
            # Get threshold for each dimension
            threshold = threshold.squeeze(-1)  # Shape: (dim,)
            
            # Compute distance from threshold
            distance_from_threshold = torch.abs(emb - threshold.unsqueeze(0))
            
            # We want values to be far from threshold (maximize distance)
            threshold_repulsion = torch.exp(-distance_from_threshold).mean()
            
            # emb must be greater than threshold_0_100[:, 0]
            # emb must be less than threshold_0_100[:, 1]
            # Common range enforcement loss for both 1-bit and 2-bit cases
            # Only penalize values outside the valid range
            lower_bound = threshold_0_100[:, 0].unsqueeze(0)  # shape (1, dim)
            upper_bound = threshold_0_100[:, 1].unsqueeze(0)  # shape (1, dim)
            
            # ReLU ensures we only penalize violations
            lower_bound_violation = torch.relu(lower_bound - emb)  # Penalize if emb < lower_bound
            upper_bound_violation = torch.relu(emb - upper_bound)  # Penalize if emb > upper_bound
            range_violation_loss = (lower_bound_violation.pow(2) + upper_bound_violation.pow(2)).mean()
        
            
            
            
            reg_loss += weight * (threshold_repulsion)
            
        elif quantization_bits == 2 or quantization_bits == 1.5:
            # For 2-bit quantization (4 levels: 0,1,2,3)
            
            # Distance from thresholds
            cur_thresholds = thresholds[dim] # shape (dim, 3)
            cur_thresholds_0_100 = thresholds_0_100[dim] # shape (dim, 2)
            
            # emb must be greater than threshold_0_100[:, 0]
            # emb must be less than threshold_0_100[:, 1]
            # add loss for this in the for loop below and ahead with proper distances which ensure that the embedding is in between 0 to 100 percentile and only penalize if it is not
            
            # Range enforcement loss - same as 1-bit case
            lower_bound = threshold_0_100[:, 0].unsqueeze(0)  # shape (1, dim)
            upper_bound = threshold_0_100[:, 1].unsqueeze(0)  # shape (1, dim)
            
            # ReLU ensures we only penalize violations
            lower_bound_violation = torch.relu(lower_bound - emb)  # Penalize if emb < lower_bound
            upper_bound_violation = torch.relu(emb - upper_bound)  # Penalize if emb > upper_bound
            range_violation_loss = (lower_bound_violation.pow(2) + upper_bound_violation.pow(2)).mean()
            
            
            distances = []
            for i in range(cur_thresholds.shape[1]):
                threshold = cur_thresholds[:, i].unsqueeze(0)
                distance = torch.abs(emb - threshold)
                distances.append(distance)
            distances = torch.stack(distances, dim=-1)
            
            # Minimum distance to any threshold (we want this to be large)
            min_threshold_distance = distances.min(dim=-1)[0]
            threshold_repulsion = torch.exp(-min_threshold_distance).mean()
            
            # Distance to nearest quantization level
            levels = torch.tensor([0., 1., 2., 3.], device=emb.device)
            level_distances = torch.abs(emb.unsqueeze(-1) - levels)
            distance_to_nearest_level = (level_distances.min(dim=-1)[0] ** 2).mean()
            
            reg_loss += weight * (threshold_repulsion)
    
    return reg_loss


  
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
        # normalize delta and emb_prev
        delta = F.normalize(delta, p=2, dim=1)
        emb_prev = F.normalize(emb_prev, p=2, dim=1)
        # Compute the dot product between delta and emb_prev  
        dot_product = torch.matmul(delta.T, emb_prev)  # Shape: (dim_curr - dim_prev, dim_prev)  
        # Compute Frobenius norm of the dot product matrix  
        reg_loss += torch.norm(dot_product, p='fro')  
    return reg_loss  


def information_bottleneck_operator(embeddings_dict: dict) -> dict:  
    """  
    Compute information bottleneck operator to focus critical information in lower dimensions.  
    """  
    dimensions = sorted(embeddings_dict.keys())  
    new_embeddings_dict = {}
    for idx, dim in enumerate(dimensions, 1):  
        emb = embeddings_dict[dim] 
        norm_before = torch.norm(emb, p=2, dim=1, keepdim=True)
        # as the dimension increases within the emb, the vector is multiplied by a linspace from 1 to 0, so the higher dimensions are multiplied by a smaller number
        multiplier = torch.linspace(1, 1/idx, emb.shape[1], device=emb.device)
        emb = emb * multiplier
        norm_after = torch.norm(emb, p=2, dim=1, keepdim=True)
        # bring each row to the same norm
        emb = emb * (norm_before / norm_after)
        new_embeddings_dict[dim] = emb
    return new_embeddings_dict
  
def information_bottleneck_regularization(embeddings_dict: dict) -> torch.Tensor:  
    """  
    Compute information bottleneck regularization to focus critical information in lower dimensions.  
    Applies progressively stronger L1 regularization to higher dimensions to encourage sparsity.
    Uses linear interpolation to create smooth weight transitions between dimension levels.
  
    Args:  
        embeddings_dict (dict): Dictionary of embeddings at different dimensions.  
  
    Returns:  
        torch.Tensor: Scalar regularization loss.  
    """  
    reg_loss = 0.0  
    dimensions = sorted(embeddings_dict.keys())  
    
    for idx, dim in enumerate(dimensions[1:], 1):  # Skip the smallest dimension
        emb = embeddings_dict[dim]
        prev_dim = dimensions[idx-1]
        segment_size = dim - prev_dim
        
        # Create linearly increasing weights for each dimension in the segment
        # from sqrt(idx-1) to sqrt(idx)
        start_weight = idx-1 # np.sqrt(idx-1)
        end_weight = idx # np.sqrt(idx)
        
        # Generate weights for each dimension in the segment
        dimension_weights = torch.linspace(start_weight, end_weight, segment_size, device=emb.device)
        
        # Apply the weights to each dimension individually
        segment_values = emb[:, prev_dim:dim]  # Shape: (batch_size, segment_size)
        weighted_values = segment_values * dimension_weights.view(1, -1)  # Broadcasting weights across batch
        
        # Compute L1 regularization with dimension-specific weights
        reg_loss += torch.mean(torch.abs(weighted_values))
        
    return reg_loss
  
def train_matryoshka_model(matryoshka_model: MatryoshkaEmbeddingModel,  
                           dataloader,  
                           num_epochs: int = 5,  
                           learning_rate: float = 1e-4,  
                           temperature: float = 0.07):  
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
    if num_epochs > 0:
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
                if len(sample_embeddings_list) >= len(dataloader)//8:  
                    break  
            sample_embeddings = torch.cat(sample_embeddings_list, dim=0)  
    
        # Initialize thresholds in quantization layers  
        sample_embeddings = sample_embeddings.to(device)  
        # Forward pass through transformer  
        embeddings_dict, non_quant_embeddings = matryoshka_model.transformer(sample_embeddings)  
        
        matryoshka_model.init_thresholds(non_quant_embeddings)  
  
    
    num_current_step = 0
    num_total_steps = num_epochs * len(dataloader)
    for epoch in range(num_epochs):  
        total_loss = 0.0  
        loss_dict = {}
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, batch in enumerate(pbar):  
            num_current_step += 1
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
            embeddings_dict, non_quant_embeddings = matryoshka_model.transformer(embeddings)  
            batch_size = embeddings.size(0)  
            positive_pairs = torch.arange(0, batch_size, device=device).view(-1, 2)  
            # Compute losses 
            # loss_contrastive = multi_scale_contrastive_loss(  
            #     embeddings_dict, positive_pairs, temperature=temperature  
            # )  
            # loss_dict['contrastive'] = loss_contrastive.item()
            loss_ortho = 0.01 * orthogonality_regularization(non_quant_embeddings)  
            loss_dict['ortho'] = loss_ortho.item()
            loss_info_bottleneck = information_bottleneck_regularization(non_quant_embeddings)
            loss_dict['info_bottleneck'] = float(loss_info_bottleneck)
            loss_similarity = 0.0  
            loss_kl_similarity = 0.0
            overall_contrastive_loss = 0.0
            dimension_levels = sorted(embeddings_dict.keys())  
            overall_rank_loss = 0.0
            scale_factor = (max(dimension_levels) / min(dimension_levels)) / np.sqrt(max(dimension_levels))
            bottleneck_operator_embeddings_dict = information_bottleneck_operator(non_quant_embeddings)
            for dim, _ in bottleneck_operator_embeddings_dict.items():  
                non_quant_emb = non_quant_embeddings[dim]
                quant_emb = embeddings_dict[dim]
                bottleneck_emb = bottleneck_operator_embeddings_dict[dim]
                weight_large_dim = scale_factor * np.sqrt(dim)
                
                emb = bottleneck_emb if use_information_bottleneck else non_quant_emb
                
                loss_similarity += (weight_large_dim * similarity_preservation_loss(embeddings, quant_emb)  + weight_large_dim * similarity_preservation_loss(embeddings, emb))
                loss_kl_similarity += (weight_large_dim * kl_similarity_preservation_loss(embeddings, quant_emb) + weight_large_dim * kl_similarity_preservation_loss(embeddings, emb))
                overall_contrastive_loss += weight_large_dim * 0.01 * contrastive_loss(quant_emb) + weight_large_dim * 0.01 * contrastive_loss(emb)
                overall_rank_loss += weight_large_dim * rank_preserving_loss(embeddings, quant_emb) + weight_large_dim * rank_preserving_loss(embeddings, emb)
                
            highest_dim = max(dimension_levels)
            embeddings_new = embeddings_dict[highest_dim]
            embeddings_original = embeddings
            
            loss = loss_similarity + loss_kl_similarity + overall_rank_loss + overall_contrastive_loss + (( reg_strength * loss_info_bottleneck ) if use_information_bottleneck_regularization else 0.0) + ((reg_strength * loss_ortho) if use_orthogonality_regularization else 0.0)
            # loss = overall_contrastive_loss
            loss_dict['similarity'] = loss_similarity.item()
            loss_dict['kl_similarity'] = loss_kl_similarity.item()
            
            loss_dict['contrastive'] = overall_contrastive_loss.item()
            loss_dict['rank'] = overall_rank_loss.item()
            # If training for quantization, add quantization regularization loss  
            if matryoshka_model.train_binary or matryoshka_model.train_two_bit or matryoshka_model.train_one_and_half_bit:  
                quantization_bits = 2 if matryoshka_model.train_two_bit else 1.5 if matryoshka_model.train_one_and_half_bit else 1  
                thresholds = {}
                thresholds_0_100 = {}
                for dim, emb in embeddings_dict.items():  
                    thresholds[dim] = matryoshka_model.transformer.quantization_layers[str(dim)].thresholds
                    thresholds_0_100[dim] = matryoshka_model.transformer.quantization_layers[str(dim)].thresholds_0_100
                cur_quant_loss = reg_strength * quantization_regularization_loss(
                    non_quant_embeddings,
                    quantization_bits=quantization_bits,
                    current_epoch=epoch,
                    total_epochs=num_epochs,
                    num_current_step=num_current_step,
                    num_total_steps=num_total_steps,
                    num_steps_per_epoch=len(dataloader),
                    thresholds=thresholds,
                    thresholds_0_100=thresholds_0_100
                )
                cur_increase_std_dev_loss = reg_strength * increase_std_dev_over_time_loss(non_quant_embeddings, epoch, num_epochs, num_current_step, num_total_steps, len(dataloader))
                quant_loss = ((cur_quant_loss if quantization_regularization else 0.0 )+ (cur_increase_std_dev_loss if increase_std_dev_over_time else 0.0))
                loss += quant_loss
                loss_dict['quantization'] = cur_quant_loss.item()
                loss_dict['std_dev'] = float(cur_increase_std_dev_loss)
                # Anneal temperatures in quantization layers  
                for dim, quant_layer in matryoshka_model.transformer.quantization_layers.items():  
                    quant_layer.anneal_temperature(epoch, num_epochs, num_current_step, num_total_steps, len(dataloader))  
                    quant_layer.update_thresholds(non_quant_embeddings[int(dim)])
            optimizer.zero_grad()  
            loss.backward()  
            loss_dict['total'] = loss.item()
            # Add gradient clipping before optimizer step  
            nn.utils.clip_grad_norm_(matryoshka_model.transformer.parameters(), max_grad_norm)  
            optimizer.step()  
            scheduler.step()  
            total_loss += loss.item()  
            pbar.set_postfix(loss_dict)
        avg_loss = total_loss / len(dataloader)  
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')  
        loss_dict['avg_loss'] = avg_loss
        
    if matryoshka_model.train_binary or matryoshka_model.train_two_bit or matryoshka_model.train_one_and_half_bit:
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
                
                embeddings = embeddings.to(device)  
                # Forward pass through transformer  
                embeddings_dict, non_quant_embeddings = matryoshka_model.transformer(embeddings)  
                for dim, quant_layer in matryoshka_model.transformer.quantization_layers.items():  
                    quant_layer.update_thresholds(non_quant_embeddings[int(dim)], momentum=0.99)
    return matryoshka_model