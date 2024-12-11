import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from transformers import AutoModel  
from sentence_transformers import SentenceTransformer  
import numpy as np  
from typing import List  
from tqdm import tqdm  
from config import *  
from common import similarity_preservation_loss  
  
class MatryoshkaEmbeddingModel:  
    """  
    Matryoshka Embedding Model that produces nested embeddings with optional binary quantization.  
  
    Attributes:  
        embedding_model_name (str): Name of the pretrained embedding model.  
        dimension_levels (List[int]): List of embedding dimensions in increasing order.  
        train_binary (bool): Flag indicating whether to train with binary quantization.  
    """  
    def __init__(self, embedding_model_name: str, dimension_levels: List[int], train_binary: bool = False):  
        """  
        Initialize the Matryoshka Embedding Model.  
  
        Args:  
            embedding_model_name (str): Name of the pretrained embedding model.  
            dimension_levels (List[int]): List of embedding dimensions in increasing order.  
            train_binary (bool): Flag to enable binary quantization training.  
        """  
        self.embedding_model_name = embedding_model_name  
        self.dimension_levels = sorted(dimension_levels)  
        self.max_dim = self.dimension_levels[-1]  
        self.train_binary = train_binary  
  
        # Load the frozen embedding model  
        self.embedding_model = SentenceTransformer(self.embedding_model_name)  
        self.embedding_model.eval()  
        for param in self.embedding_model.parameters():  
            param.requires_grad = False  
  
        # Define the transformation network  
        self.transformer = MatryoshkaTransformer(  
            input_dim=self.embedding_model.get_sentence_embedding_dimension(),  
            dimension_levels=self.dimension_levels,  
            train_binary=self.train_binary  
        )  
  
    def encode(self, sentences: List[str], output_dim: int = None, do_binary: bool = False, **kwargs) -> np.ndarray:  
        """  
        Encode sentences to obtain embeddings.  
  
        Args:  
            sentences (List[str]): List of sentences to encode.  
            output_dim (int, optional): Desired embedding dimension. If None, use max dimension.  
            do_binary (bool): Whether to output binary embeddings.  
            **kwargs: Additional arguments for the embedding model.  
  
        Returns:  
            np.ndarray: Embeddings of shape (num_sentences, output_dim)  
        """  
        if output_dim is None:  
            output_dim = self.max_dim  
        assert output_dim in self.dimension_levels, f"Output dimension must be one of {self.dimension_levels}"  
          
        # Check if binary mode is consistent with training  
        if do_binary and not self.train_binary:  
            raise ValueError("Model was not trained with binary quantization. Set train_binary=True during training.")  
  
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
        self.transformer.eval()  
        with torch.no_grad():  
            matryoshka_embeddings = self.transformer(base_embeddings, apply_binary=do_binary)  
            embeddings = matryoshka_embeddings[output_dim]  
  
        # Normalize embeddings  
        embeddings = F.normalize(embeddings, p=2, dim=1)  
  
        # If binary mode, convert embeddings to binary  
        if do_binary:  
            embeddings = (embeddings > 0.5).float()  
  
        return embeddings.cpu().numpy()  
  
class MatryoshkaTransformer(nn.Module):  
    """  
    Transformation network that produces nested embeddings using a Pyramid Network architecture,  
    with optional binary quantization layers.  
    """  
    def __init__(self, input_dim: int, dimension_levels: List[int], train_binary: bool = False):  
        """  
        Initialize the Transformer.  
  
        Args:  
            input_dim (int): Dimension of the input embeddings.  
            dimension_levels (List[int]): List of embedding dimensions in increasing order.  
            train_binary (bool): Whether to include binary quantization layers.  
        """  
        super(MatryoshkaTransformer, self).__init__()  
        self.dimension_levels = sorted(dimension_levels)  
        self.train_binary = train_binary  
        self.blocks = nn.ModuleList()  
        self.quantization_layers = nn.ModuleDict()  
        prev_dim = 0  
        assert input_dim >= self.dimension_levels[-1], "Input dimension must be greater than or equal to the largest dimension level"  
        if input_dim > self.dimension_levels[-1]:  
            self.dimension_levels.append(input_dim)  
  
        self.base_transform = nn.Sequential(  
            nn.Dropout(0.1),  
            nn.Linear(input_dim, input_dim * 2),  
            nn.LeakyReLU(),  
            nn.LayerNorm(input_dim * 2),  
        )  
  
        for dim in self.dimension_levels:  
            block = nn.Sequential(  
                nn.Linear(input_dim * 2, dim - prev_dim)  
            )  
            self.blocks.append(block)  
            if self.train_binary:  
                # Add quantization layer for this dimension level  
                self.quantization_layers[str(dim)] = QuantizationLayer(dim)  
            prev_dim = dim  
  
    def forward(self, x: torch.Tensor, apply_binary: bool = False) -> dict:  
        """  
        Forward pass to obtain nested embeddings.  
  
        Args:  
            x (torch.Tensor): Input embeddings of shape (batch_size, input_dim)  
            apply_binary (bool): Whether to apply binary quantization layers.  
  
        Returns:  
            dict: Dictionary of embeddings at each dimension level.  
        """  
        embeddings = {}  
        x = self.base_transform(x)  
        prev_embedding = x  
        all_embeddings = x  
  
        for idx, block in enumerate(self.blocks):  
            delta = block(prev_embedding)  
            prev_embedding = torch.cat([prev_embedding, delta], dim=1)  
            all_embeddings = prev_embedding  
            dim = self.dimension_levels[idx]  
            
            # Normalize embeddings before quantization  
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1) 
  
            # Apply quantization layer if training for binary  
            if self.train_binary:  
                quant_layer = self.quantization_layers[str(dim)]  
                all_embeddings = quant_layer(all_embeddings, self.training)  
                if apply_binary and not self.training:  
                    # During inference, apply hard quantization  
                    all_embeddings = (all_embeddings > 0.5).float()  
            embeddings[dim] = all_embeddings  
  
        embeddings_dict = {}  
        for dim, emb in embeddings.items():  
            embeddings_dict[dim] = emb  
  
        return embeddings_dict  
  
class QuantizationLayer(nn.Module):  
    """  
    Quantization layer using temperature-scaled sigmoid function with temperature annealing.  
    """  
    def __init__(self, dim: int, initial_temperature: float = 5.0, min_temperature: float = 0.5):  
        """  
        Initialize the Quantization Layer.  
  
        Args:  
            dim (int): Dimension of the input embeddings.  
            initial_temperature (float): Initial temperature for the sigmoid function.  
            min_temperature (float): Minimum temperature to anneal to.  
        """  
        super(QuantizationLayer, self).__init__()  
        self.temperature = initial_temperature  
        self.min_temperature = min_temperature  
        self.scale = nn.Parameter(torch.ones(dim))  # + torch.randn(dim) * init_std
        self.thresholds = nn.Parameter(torch.zeros(dim))  # + torch.randn(dim) * init_std
  
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:  
        """  
        Forward pass through the quantization layer.  
  
        Args:  
            x (torch.Tensor): Input embeddings.  
            training (bool): Indicates whether in training mode.  
  
        Returns:  
            torch.Tensor: Quantized embeddings.  
        """  
        x = x * self.scale - self.thresholds  # Learnable scaling and thresholding  
        if training:  
            # Apply temperature-scaled sigmoid  
            x = torch.sigmoid(x / self.temperature)  
            # Straight-Through Estimator (STE)  
            binary_x = (x > 0.5).float()  
            return x + (binary_x - x).detach()  
        else:  
            # Inference mode - hard thresholding  
            x = (x > 0.0).float()  
            return x  
  
    def anneal_temperature(self, current_epoch: int, total_epochs: int):  
        """  
        Anneal the temperature parameter over epochs.  
  
        Args:  
            current_epoch (int): Current training epoch.  
            total_epochs (int): Total number of training epochs.  
        """  
        # Exponential decay of temperature  
        progress = current_epoch / total_epochs  
        new_temperature = max(self.min_temperature, self.temperature * (0.95 ** progress))  
        self.temperature = new_temperature  
  
def multi_scale_contrastive_loss(embeddings_dict: dict, positive_pairs: torch.Tensor,  
                                 temperature: float = 0.07, weights: dict = None) -> torch.Tensor:  
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
        loss = F.cross_entropy(similarity_matrix / temperature, labels)  
        total_loss += weights[dim] * loss  
  
    return total_loss  
  
def binary_regularization_loss(embeddings_dict: dict, current_epoch: int, total_epochs: int) -> torch.Tensor:  
    """  
    Compute regularization loss to encourage binary outputs.  
  
    Args:  
        embeddings_dict (dict): Dictionary of embeddings at different dimensions.  
        current_epoch (int): Current training epoch.  
        total_epochs (int): Total number of training epochs.  
  
    Returns:  
        torch.Tensor: Scalar regularization loss.  
    """  
    reg_loss = 0.0  
    progress = current_epoch / total_epochs  # Progress ratio (0 to 1)  
    weight = progress  # Increase weight over time  
    for emb in embeddings_dict.values():  
        # Binary cross-entropy loss towards 0.5 (center)  
        bce_loss = F.binary_cross_entropy(emb, torch.ones_like(emb) * 0.5)  
        # L1 regularization towards 0 or 1  
        l1_loss = torch.mean(torch.abs(emb * (1 - emb)))  
        reg_loss += weight * (bce_loss + l1_loss)  
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
  
def train_matryoshka_model(matryoshka_model: MatryoshkaEmbeddingModel, dataloader,  
                           num_epochs: int = 5, learning_rate: float = 1e-4,  
                           temperature: float = 0.07, reg_strength: float = 1e-3):  
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
                                                    epochs=num_epochs, steps_per_epoch=len(dataloader))  
    matryoshka_model.embedding_model.eval()  
    matryoshka_model.transformer.train()  
    for epoch in range(num_epochs):  
        total_loss = 0.0  
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):  
            input_ids = batch['input_ids'].squeeze(1).to(device)  
            attention_mask = batch['attention_mask'].squeeze(1).to(device)  
            with torch.no_grad():  
                embeddings = matryoshka_model.embedding_model(  
                    input_ids=input_ids,  
                    attention_mask=attention_mask,  
                )['sentence_embedding']  
                embeddings = F.normalize(embeddings, p=2, dim=1)  
            embeddings = embeddings.to(device)  
            # Forward pass through transformer  
            embeddings_dict = matryoshka_model.transformer(embeddings)  
            batch_size = embeddings.size(0)  
            positive_pairs = torch.arange(0, batch_size, device=device).view(-1, 2)  
            # Compute losses  
            loss_contrastive = multi_scale_contrastive_loss(  
                embeddings_dict, positive_pairs, temperature=temperature  
            )  
            loss_ortho = orthogonality_regularization(embeddings_dict)  
            loss_info_bottleneck = information_bottleneck_regularization(embeddings_dict)  
            loss_similarity = 0.0  
            dimension_levels = sorted(embeddings_dict.keys())  
            for dim, emb in embeddings_dict.items():  
                weight = 1.0 / (dim / min(dimension_levels))  
                loss_similarity += weight * similarity_preservation_loss(embeddings, emb)  
            loss = loss_contrastive + reg_strength * (loss_ortho + loss_info_bottleneck) + loss_similarity  
            # If training for binary quantization, add binary regularization loss  
            if matryoshka_model.train_binary:  
                loss_binary = binary_regularization_loss(embeddings_dict, epoch, num_epochs)  
                loss += reg_strength * loss_binary  
                # Anneal temperatures in quantization layers  
                for quant_layer in matryoshka_model.transformer.quantization_layers.values():  
                    quant_layer.anneal_temperature(epoch, num_epochs)  
            optimizer.zero_grad()  
            loss.backward()  
            # Add gradient clipping before optimizer step  
            torch.nn.utils.clip_grad_norm_(matryoshka_model.transformer.parameters(), max_grad_norm)  
            optimizer.step()  
            scheduler.step()  
            total_loss += loss.item()  
        avg_loss = total_loss / len(dataloader)  
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')  
  
