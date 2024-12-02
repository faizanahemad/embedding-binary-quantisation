import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import numpy as np  

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
from common import *


class QuantizationModuleOneBitTwoBit(nn.Module):  
    """  
    Quantization Module for multi-threshold quantization with selective bit allocation.  
  
    This module quantizes input embeddings using either 1-bit or 2-bit quantization per dimension,  
    based on the computed importance scores of each dimension. Importance scores can be computed  
    using variance, gradients, or a combination of both. The module supports converting quantized  
    embeddings into binary representations using a predefined codebook suitable for efficient  
    similarity computations (e.g., using Hamming distance).  
  
    Args:  
        embedding_dim (int): Dimension of the embeddings.  
        sample_embeddings (np.ndarray, optional): Sample embeddings to initialize thresholds and importance scores.  
        binary_dims (int, optional): Number of dimensions to quantize using 1 bit (least informative dimensions).  
                                     If 0, all dimensions use 2-bit quantization. Defaults to 0.  
        is_matryoshka (bool, optional): If True, uses matryoshka embeddings and doesn't change which embeddings are used for which bits, using the initial embeddings for 2-bit and the last embeddings for 1-bit. Defaults to False.  
        use_grad_for_importance (bool, optional): If True, uses gradients and variance to compute importance scores.  
                                                  If False, uses only variance. Defaults to False.  
        momentum (float, optional): Momentum parameter for updating importance scores. Defaults to 0.9.  
  
    Attributes:  
        thresholds (nn.Parameter): Learnable thresholds of shape (embedding_dim, num_thresholds).  
        importance_scores (torch.Tensor): Importance scores per dimension.  
        codebook (dict): Mapping from quantization levels to binary codes.  
        high_info_dims (torch.Tensor): Indices of dimensions using 2-bit quantization.  
        low_info_dims (torch.Tensor): Indices of dimensions using 1-bit quantization.  
        hessian_diag (torch.Tensor): Diagonal approximation of the Hessian for embeddings (if gradients are used).  
  
    Notes:  
        - The module updates importance scores during training using a momentum-based approach.  
        - Importance scores are used to determine which dimensions are quantized using 1 bit or 2 bits.  
        - The codebook maps quantization levels to binary codes to facilitate efficient similarity computations.  
    """  
  
    def __init__(self, embedding_dim, sample_embeddings=None, binary_dims=0,  is_matryoshka=False,
                 use_grad_for_importance=False, momentum=0.99):  
        super(QuantizationModuleOneBitTwoBit, self).__init__()  
        self.embedding_dim = embedding_dim  
        self.num_thresholds = 3  # For 2-bit quantization  
        self.use_grad_for_importance = use_grad_for_importance  
        self.momentum = momentum  
        self.is_matryoshka = is_matryoshka
  
        # Initialize thresholds  
        if sample_embeddings is not None:  
            # Compute initial thresholds and importance scores based on sample embeddings  
            self.thresholds = nn.Parameter(self.initialize_thresholds(sample_embeddings))  
            self.importance_scores = self.compute_initial_importance(sample_embeddings)  
        else:  
            # Default initialization if no sample embeddings are provided  
            thresholds = torch.linspace(-1.0, 1.0, steps=self.num_thresholds + 2)[1:-1]  
            thresholds = thresholds.unsqueeze(0).repeat(embedding_dim, 1)  
            self.thresholds = nn.Parameter(thresholds)  
            # Initialize importance scores to ones  
            self.register_buffer('importance_scores', torch.ones(embedding_dim))
  
        # Initialize Hessian diagonal approximation for gradient-based importance (if used)  
        self.register_buffer('hessian_diag', torch.zeros(embedding_dim))
  
        # Create the codebook mapping quantization levels to binary codes  
        self.codebook = {  
            0: torch.tensor([0, 0, 0], dtype=torch.float32),  
            1: torch.tensor([0, 0, 1], dtype=torch.float32),  
            2: torch.tensor([0, 1, 1], dtype=torch.float32),  
            3: torch.tensor([1, 1, 1], dtype=torch.float32)  
        }  
  
        # Determine high and low importance dimensions based on importance scores  
        self.update_dimension_indices(binary_dims)  
  
    def initialize_thresholds(self, sample_embeddings):  
        """  
        Initializes thresholds based on sample embeddings.  
  
        This method computes the percentiles of the sample embeddings to initialize thresholds  
        that bisect each dimension's distribution.  
  
        Args:  
            sample_embeddings (np.ndarray): Sample embeddings, shape (num_samples, embedding_dim).  
  
        Returns:  
            torch.Tensor: Initialized thresholds, shape (embedding_dim, num_thresholds).  
        """  
        sample_embeddings = sample_embeddings.clone().detach() if torch.is_tensor(sample_embeddings) else torch.tensor(sample_embeddings, dtype=torch.float32)  
        thresholds = []  
        quantiles = torch.linspace(0.25, 0.75, self.num_thresholds, device=sample_embeddings.device)
        # print(f"[DEBUG] QuantizationModuleOneBitTwoBit.initialize_thresholds() - quantiles shape: {quantiles.shape}, quantiles: {quantiles}")
        # for dim in range(self.embedding_dim):  
        #     # Get the values for this dimension  
        #     values = sample_embeddings[:, dim]  
        #     # Compute percentiles to bisect the distribution  
        #     percentiles = torch.quantile(values, quantiles)  
        #     thresholds.append(percentiles)  
        # thresholds = torch.stack(thresholds)  
        # print(f"[DEBUG] QuantizationModuleOneBitTwoBit.initialize_thresholds() - thresholds shape: {thresholds.shape}, sample_embeddings shape: {sample_embeddings.shape}")
        percentiles = torch.quantile(sample_embeddings, quantiles, dim=0, keepdim=True)
        # Reshape percentiles from [3, 1, 384] to [384, 3] to match thresholds shape
        percentiles = percentiles.squeeze(1).transpose(0,1)
        # print(f"[DEBUG] QuantizationModuleOneBitTwoBit.initialize_thresholds() - percentiles shape: {percentiles.shape}")
        # check if percentiles are same as thresholds
        thresholds = percentiles
        # assert torch.allclose(percentiles, thresholds)
        # print(f"[DEBUG] QuantizationModuleOneBitTwoBit.initialize_thresholds() - thresholds shape: {thresholds.shape}, percentiles shape: {percentiles.shape}")
        # print(f"[DEBUG] QuantizationModuleOneBitTwoBit.initialize_thresholds() - thresholds: \n{thresholds}")
        return thresholds  
  
    def compute_initial_importance(self, sample_embeddings):  
        """  
        Computes initial importance scores based on variance or combined with gradients.  
  
        Args:  
            sample_embeddings (np.ndarray): Sample embeddings, shape (num_samples, embedding_dim).  
  
        Returns:  
            torch.Tensor: Importance scores per dimension, shape (embedding_dim,).  
        """  
        sample_embeddings = torch.tensor(sample_embeddings, dtype=torch.float32)  
        # Compute variance per dimension as a proxy for information content  
        variance = torch.var(sample_embeddings, dim=0)  
  
        if self.use_grad_for_importance:  
            # Initialize Hessian diagonal approximation to zeros  
            self.hessian_diag = torch.zeros(self.embedding_dim)  
            # Since gradients are not available during initialization, use variance only  
            importance_scores = variance  
        else:  
            importance_scores = variance  
  
        return importance_scores  
  
    def update_dimension_indices(self, binary_dims):  
        """  
        Updates the indices of high and low importance dimensions.
        For matryoshka mode, uses first dims for high info and last dims for low info.
        Otherwise uses importance scores to determine high/low info dims.
  
        Args:  
            binary_dims (int): Number of dimensions to quantize using 1 bit.  
        """  
        if binary_dims <= 0:
            self.low_info_dims = torch.tensor([], dtype=torch.long)
            self.high_info_dims = torch.arange(self.embedding_dim)
            return

        if self.is_matryoshka:
            # For matryoshka mode, first dims are high info, last dims are low info
            self.high_info_dims = torch.arange(self.embedding_dim - binary_dims)
            self.low_info_dims = torch.arange(self.embedding_dim - binary_dims, self.embedding_dim)
        else:
            # Use importance scores to determine high/low info dims
            sorted_dims = torch.argsort(self.importance_scores, descending=True)
            self.low_info_dims = sorted_dims[-binary_dims:]
            self.high_info_dims = sorted_dims[:-binary_dims]
  
    def compute_information_gain(self, embeddings):  
        """  
        Computes importance scores based on variance using a momentum-based update.  
  
        This method updates the stored importance scores using the variance of the current batch  
        and an exponential moving average controlled by the momentum parameter.  
  
        Args:  
            embeddings (torch.Tensor): Embeddings from the current batch, shape (batch_size, embedding_dim).  
  
        Returns:  
            torch.Tensor: Updated importance scores per dimension, shape (embedding_dim,).  
        """  
        batch_variance = torch.var(embeddings, dim=0)  
        self.importance_scores = self.momentum * self.importance_scores + (1 - self.momentum) * batch_variance  
        return self.importance_scores  
  
    def compute_information_gain_with_gradients(self, embeddings):  
        """  
        Computes importance scores based on gradients and variance using a momentum-based update.  
  
        This method combines the absolute mean gradients and variance of the embeddings to compute  
        importance scores and updates them using an exponential moving average.  
  
        Args:  
            embeddings (torch.Tensor): Embeddings from the current batch, shape (batch_size, embedding_dim).  
  
        Returns:  
            torch.Tensor: Updated importance scores per dimension, shape (embedding_dim,).  
        """  
        # Gradient-based importance  
        if embeddings.grad is not None:  
            grad_importance = torch.abs(embeddings.grad).mean(dim=0)  
        else:  
            grad_importance = torch.zeros_like(self.importance_scores)  
  
        # Magnitude-based importance (variance)  
        batch_variance = torch.var(embeddings, dim=0)  
  
        # Combine both metrics  
        importance_scores = grad_importance + batch_variance  
  
        # Update importance scores using momentum  
        self.importance_scores = self.momentum * self.importance_scores + (1 - self.momentum) * importance_scores  
  
        # Update Hessian diagonal approximation (optional)  
        self.update_hessian(embeddings)  
  
        return self.importance_scores  
  
    def update_hessian(self, embeddings):  
        """  
        Updates the diagonal approximation of the Hessian matrix using gradients.  
  
        This method computes the squared gradients of the scaled embeddings and updates  
        the Hessian diagonal approximation using an exponential moving average.  
  
        Args:  
            embeddings (torch.Tensor): Embeddings from the current batch, shape (batch_size, embedding_dim).  
        """  
        if embeddings.grad is not None:  
            grad_sq = torch.pow(embeddings.grad, 2)  
            new_hessian = grad_sq.mean(dim=0)  
            # Use exponential moving average for stability  
            self.hessian_diag = self.momentum * self.hessian_diag + (1 - self.momentum) * new_hessian.detach()  
  
    def forward(self, embeddings, binary=False):  
        """  
        Forward pass to compute the quantized embeddings.  
  
        Args:  
            embeddings (torch.Tensor): Original embeddings, shape (batch_size, embedding_dim).  
            binary (bool, optional): If True, returns binary representations.  
                                     If False, returns quantization levels.  
                                     Defaults to False.  
  
        Returns:  
            torch.Tensor: Quantized embeddings. If binary=True, shape is (batch_size, total_bits).  
                          If binary=False, shape is (batch_size, embedding_dim).  
        """  
        batch_size = embeddings.size(0)  
        if self.training:
            # Enable gradient tracking if gradients are used for importance computation  
            if self.use_grad_for_importance:  
                embeddings.requires_grad_(True)  
                embeddings.retain_grad()  
    
            # Compute or update importance scores  
            if self.use_grad_for_importance:  
                self.compute_information_gain_with_gradients(embeddings)  
            else:  
                self.compute_information_gain(embeddings)  
    
            # Update dimension indices based on new importance scores  
            self.update_dimension_indices(binary_dims=self.low_info_dims.numel())  
  
        if binary:  

            # Prepare codebook tensor  

            code_length = len(next(iter(self.codebook.values())))  

            codebook_tensor = torch.stack([self.codebook[i] for i in range(self.num_thresholds + 1)]).to(embeddings.device)  

            # Prepare dictionaries for dimension to index mapping  

            high_info_dim_to_idx = {dim.item(): idx for idx, dim in enumerate(self.high_info_dims)}  

            low_info_dim_to_idx = {dim.item(): idx for idx, dim in enumerate(self.low_info_dims)}  

            # Quantize high importance dimensions  

            if self.high_info_dims.numel() > 0:  

                high_embeddings = embeddings[:, self.high_info_dims]  

                quantized_high = self.multi_bit_quantization(high_embeddings, self.high_info_dims)  

                quantized_high_codes = self.quantization_levels_to_codes(quantized_high)  

            else:  

                quantized_high_codes = torch.zeros(batch_size, 0, device=embeddings.device)  

            # Quantize low importance dimensions  

            if self.low_info_dims.numel() > 0:  

                low_embeddings = embeddings[:, self.low_info_dims]  

                quantized_low = self.single_bit_quantization(low_embeddings, self.low_info_dims)  

                quantized_low_codes = (quantized_low > 0.5).float()  

            else:  

                quantized_low_codes = torch.zeros(batch_size, 0, device=embeddings.device)  

            # Collect bits per dimension in order  

            bits_list = []  

            for d in range(self.embedding_dim):  

                if d in high_info_dim_to_idx:  

                    idx = high_info_dim_to_idx[d]  

                    start_idx = idx * code_length  

                    end_idx = (idx + 1) * code_length  

                    bits = quantized_high_codes[:, start_idx:end_idx]  

                    bits_list.append(bits)  

                elif d in low_info_dim_to_idx:  

                    idx = low_info_dim_to_idx[d]  

                    bits = quantized_low_codes[:, idx:idx+1]  # Ensure it's of size [batch_size, 1]  

                    bits_list.append(bits)  

                else:  

                    # Should not happen, but handle just in case  

                    pass  

            # Concatenate bits along the feature dimension  

            binary_embeddings = torch.cat(bits_list, dim=1)  

            return binary_embeddings  

        else:  

            # Proceed as before  

            quantized_embeddings = torch.zeros_like(embeddings)  

            # Quantize high importance dimensions using 2-bit quantization  

            if self.high_info_dims.numel() > 0:  

                high_embeddings = embeddings[:, self.high_info_dims]  

                quantized_high = self.multi_bit_quantization(high_embeddings, self.high_info_dims)  

                quantized_embeddings[:, self.high_info_dims] = quantized_high  

            # Quantize low importance dimensions using 1-bit quantization  

            if self.low_info_dims.numel() > 0:  

                low_embeddings = embeddings[:, self.low_info_dims]  

                quantized_low = self.single_bit_quantization(low_embeddings, self.low_info_dims)  

                quantized_embeddings[:, self.low_info_dims] = quantized_low  

            return quantized_embeddings  
  
    def multi_bit_quantization(self, embeddings, dims):
        # Extract thresholds for the specified dimensions
        thresholds = self.thresholds[dims, :]  # Shape: (num_dims, num_thresholds)

        # Enforce ordering constraints on thresholds by sorting
        thresholds = torch.sort(thresholds, dim=1)[0]

        # Expand dimensions for broadcasting
        embeddings_expanded = embeddings.unsqueeze(2)  # Shape: (batch_size, num_dims, 1)
        thresholds_expanded = thresholds.unsqueeze(0)  # Shape: (1, num_dims, num_thresholds)

        # Compute soft assignments
        k = temperature  # Temperature parameter controlling the steepness
        logits = k * (embeddings_expanded - thresholds_expanded)
        sigma = torch.sigmoid(logits)  # Shape: (batch_size, num_dims, num_thresholds)

        # Compute probabilities for each quantization level (0 to 3)
        p0 = 1.0 - sigma[:, :, 0]  # Probability of being less than first threshold
        p1 = sigma[:, :, 0] - sigma[:, :, 1]  # Between first and second threshold
        p2 = sigma[:, :, 1] - sigma[:, :, 2]  # Between second and third threshold
        p3 = sigma[:, :, 2]  # Greater than third threshold
        
        # Stack probabilities
        probabilities = torch.stack([p0, p1, p2, p3], dim=2)  # Shape: (batch_size, num_dims, 4)
        
        # Ensure probabilities sum to 1
        probabilities = F.softmax(probabilities * k, dim=2)

        # Quantization levels (0 to 3)
        levels = torch.arange(4, dtype=torch.float32, device=embeddings.device)
        
        # Compute weighted sum
        quantized_embeddings = torch.einsum('bdk,k->bd', probabilities, levels)

        if self.training:
            print(f"[DEBUG] Value: {embeddings[0,0].item():.4f}")
            print(f"[DEBUG] Thresholds: {thresholds[0]}")
            print(f"[DEBUG] Probabilities: {probabilities[0,0]}")
            print(f"[DEBUG] Quantized value: {quantized_embeddings[0,0].item():.4f}")

        return quantized_embeddings
  
    def single_bit_quantization(self, embeddings, dims):  
        """  
        Applies 1-bit quantization to the specified dimensions.  
  
        This method quantizes the embeddings using a single threshold per dimension,  
        resulting in two quantization levels (0 and 1).  
  
        Args:  
            embeddings (torch.Tensor): Embeddings to quantize, shape (batch_size, num_dims).  
            dims (torch.Tensor): Indices of dimensions being quantized.  
  
        Returns:  
            torch.Tensor: Quantized embeddings, shape (batch_size, num_dims).  
        """  
        # For 1-bit quantization, we need a single threshold per dimension  
        # Use the first threshold from self.thresholds  
        thresholds = self.thresholds[dims, self.num_thresholds//2]  # Shape: (num_dims,)  
  
        # Enforce ordering constraints if needed  
        thresholds = thresholds.clamp(min=embeddings.min().item(), max=embeddings.max().item())  
  
        # Compute soft assignments  
        k = temperature  # Temperature parameter  
        logits = k * (embeddings - thresholds.unsqueeze(0))  # Shape: (batch_size, num_dims)  
        quantized_embeddings = torch.sigmoid(logits)  
  
        # Since it's 1-bit, quantization levels are 0 and 1  
        # Probabilities represent the probability of level 1  
        return quantized_embeddings  
  
    def quantization_levels_to_codes(self, quantized_embeddings):  

        """  
        Converts quantization levels to their binary codes using the codebook.  
        Args:  
            quantized_embeddings (torch.Tensor): Quantized embeddings (levels), shape (batch_size, num_dims).  
        Returns:  
            torch.Tensor: Binary representations, shape (batch_size, num_dims * code_length).  
        """  

        batch_size, num_dims = quantized_embeddings.size()  
        code_length = len(next(iter(self.codebook.values())))  
        # Round quantized embeddings to nearest integer levels  
        quantized_levels = torch.round(quantized_embeddings).long()  

        # Create codebook tensor  
        codebook_tensor = torch.stack([self.codebook[i] for i in range(self.num_thresholds + 1)]).to(quantized_embeddings.device)  

        # Map quantized_levels to codes  
        codes = codebook_tensor[quantized_levels]  # Shape: (batch_size, num_dims, code_length)  

        # Reshape to [batch_size, num_dims * code_length]  
        codes = codes.view(batch_size, num_dims * code_length)  

        return codes  
    
    
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
from tqdm import tqdm  
  
def train_quantization_module_one_bit_two_bit(embedding_model, quantization_module: QuantizationModuleOneBitTwoBit, dataloader, num_epochs=5, lr=1e-3, reg_strength=1e-5):  
    """  
    Train the QuantizationModule.  
  
    This function trains the provided QuantizationModule to quantize embeddings  
    from a frozen embedding model while preserving similarity relationships.  
  
    Args:  
        embedding_model (nn.Module): Pretrained embedding model (frozen during training).  
        quantization_module (QuantizationModule): The quantization module to be trained.  
        dataloader (DataLoader): DataLoader providing the training data.  
        num_epochs (int, optional): Number of training epochs. Defaults to 5.  
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.  
        reg_strength (float, optional): Regularization strength for thresholds. Defaults to 1e-5.  
  
    Returns:  
        QuantizationModule: The trained quantization module.  
    """  
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    # Determine the device to use (GPU if available)  
    
    embedding_model.to(device)  
    quantization_module.to(device)  
    
    with torch.no_grad():
        i = 0
        for batch in tqdm(dataloader, desc="Initializing Thresholds", total=len(dataloader)):
            input_ids = batch['input_ids'].to(device)  
            attention_mask = batch['attention_mask'].to(device)  
            embeddings = embedding_model(input_ids=input_ids, attention_mask=attention_mask)  
            embeddings = mean_pool_and_L2_normalize(embeddings, attention_mask)
            if i == 0:
                quantization_module.thresholds.data = quantization_module.initialize_thresholds(embeddings)
                i += 1
            else:
                quantization_module.thresholds.data = 0.99 * quantization_module.thresholds.data + \
                    0.01 * quantization_module.initialize_thresholds(embeddings)
                
    print(f"[DEBUG] QuantizationModuleOneBitTwoBit.train_quantization_module_one_bit_two_bit() - thresholds: {quantization_module.thresholds}")
    return quantization_module
    
    original_thresholds = quantization_module.thresholds.data.clone().detach()

    # Set up the optimizer and learning rate scheduler  
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, quantization_module.parameters()), lr=lr)  
    scheduler = optim.lr_scheduler.OneCycleLR(  
        optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(dataloader)  
    )  
  
    
  
    # Set the embedding model to evaluation mode and freeze its parameters  
    embedding_model.eval()  
    for param in embedding_model.parameters():  
        param.requires_grad = False  # Ensure embedding model parameters are not updated  
  
    # Set the quantization module to training mode  
    quantization_module.train()  
  
  
    # Training loop  
    for epoch in range(num_epochs):  
        total_loss = 0.0  
  
        # Iterate over batches  
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in progress_bar:  
            # Move input data to the appropriate device  
            input_ids = batch['input_ids'].to(device)  
            attention_mask = batch['attention_mask'].to(device)  
  
            # Obtain embeddings from the embedding model  
            # Note: embeddings need to have requires_grad=True for importance computation  
            with torch.no_grad():   
                embeddings = embedding_model(input_ids=input_ids, attention_mask=attention_mask)  
                embeddings = mean_pool_and_L2_normalize(embeddings, attention_mask)
            embeddings.requires_grad_(True)  # Ensure embeddings have gradients  
  
            # Pass the embeddings through the quantization module  
            quantized_embeddings = quantization_module(embeddings, binary=False)  
  
            # Compute the similarity preservation loss  
            loss_similarity = similarity_preservation_loss(embeddings, quantized_embeddings) + reg_strength * torch.norm(quantization_module.thresholds - original_thresholds, 1)
            
  
            # Regularization: L2 norm of the thresholds to prevent them from growing too large  
            # reg_thresholds = torch.norm(quantization_module.thresholds, p=2)  
  
            # Total loss combines similarity preservation and regularization  
            loss = loss_similarity
            # loss += matching_preserving_loss(embeddings, quantized_embeddings) + rank_preserving_loss(embeddings, quantized_embeddings)
            # loss += contrastive_loss(embeddings)
  
            # Backpropagation and optimizer step  
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
            scheduler.step()  
  
            # Accumulate the loss for reporting  
            total_loss += loss.item()  
            progress_bar.set_description(f'Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}')
  
        # Calculate average loss for the epoch  
        avg_loss = total_loss / len(dataloader)  
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')  
  
    # Return the trained quantization module  
    return quantization_module  

  
  
if __name__ == "__main__":  
    # Test cases for QuantizationModule  
  
    # Set random seed for reproducibility  
    torch.manual_seed(42)  
    np.random.seed(42)  
  
    # Parameters  
    embedding_dim = 10  
    batch_size = 32  
  
    # Generate sample embeddings for initialization  
    num_samples = 1000  
    sample_embeddings = np.random.randn(num_samples, embedding_dim)  
  
    # Create the quantization module  
    quant_module = QuantizationModuleOneBitTwoBit(  
        embedding_dim=embedding_dim,  
        sample_embeddings=sample_embeddings,  
        binary_dims=3,  # Number of low-information dimensions to quantize using 1 bit  
        use_grad_for_importance=True,  # Use gradients for importance computation  
        momentum=0.9  
    )  
  
    # Generate a batch of embeddings  
    embeddings = 5 * torch.randn(batch_size, embedding_dim)  
  
    # Simulate a training step  
    embeddings.requires_grad_(True)  
    embeddings.retain_grad()  # Retain gradients  
  
    # Forward pass to get quantized embeddings  
    quantized_embeddings = quant_module(embeddings, binary=False)  
    print("Quantized Embeddings (Levels):")  
    print(quantized_embeddings)  
  
    # Simulate a loss and backward pass  
    loss = quantized_embeddings.sum()  
    loss.backward()  
  
    # Forward pass to get binary representations  
    binary_embeddings = quant_module(embeddings, binary=True)  
    print("\nBinary Embeddings:")  
    print(binary_embeddings)  
  
    # Check the shapes  
    print("\nShapes:")  
    print(f"Quantized Embeddings Shape: {quantized_embeddings.shape}")  
    print(f"Binary Embeddings Shape: {binary_embeddings.shape}")  # Should be [batch_size, total_bits]  

    # Verify that the binary embeddings have the correct number of bits  
    total_bits = quant_module.high_info_dims.numel() * 3 + quant_module.low_info_dims.numel()  
    assert binary_embeddings.shape == (batch_size, total_bits), "Binary embeddings shape mismatch."  
  
    # Check if importance scores are updated  
    print("\nImportance Scores:")  
    print(quant_module.importance_scores)  
  
    # Check if Hessian diagonal approximation is updated  
    print("\nHessian Diagonal Approximation:")  
    print(quant_module.hessian_diag)
    
    
    # ------------------------- Test Case 1: Identical Embeddings -------------------------  
    print("Test Case 1: Identical Embeddings")  
    embedding_a = torch.randn(1, embedding_dim)  
    embedding_b = embedding_a.clone()  # Identical embedding  
  
    # Quantize embeddings  
    binary_a = quant_module(embedding_a, binary=True)  
    binary_b = quant_module(embedding_b, binary=True)  
  
    # Compute Hamming distance  
    hamming_distance = (binary_a != binary_b).float().sum().item()  
    total_bits = binary_a.numel()  
    normalized_hamming_distance = hamming_distance / total_bits  
    print(f"Hamming Distance: {hamming_distance}")  
    print(f"Normalized Hamming Distance: {normalized_hamming_distance}")  
    assert hamming_distance == 0, "Hamming distance should be 0 for identical embeddings."  
  
    # ------------------------- Test Case 2: Opposite Embeddings -------------------------  
    print("\nTest Case 2: Opposite Embeddings")  
    embedding_c = torch.randn(1, embedding_dim)  
    embedding_d = -embedding_c  # Opposite embedding  
  
    # Quantize embeddings  
    binary_c = quant_module(embedding_c, binary=True)  
    binary_d = quant_module(embedding_d, binary=True)  
  
    # Compute Hamming distance  
    hamming_distance = (binary_c != binary_d).float().sum().item()  
    total_bits = binary_c.numel()  
    normalized_hamming_distance = hamming_distance / total_bits  
    print(f"Hamming Distance: {hamming_distance}")  
    print(f"Normalized Hamming Distance: {normalized_hamming_distance}")  
    # Since quantization may not exactly invert, we expect high but possibly not maximum Hamming distance  
  
    # ------------------------- Test Case 3: Random Embeddings -------------------------  
    print("\nTest Case 3: Random Embeddings")  
    embedding_e = torch.randn(1, embedding_dim)  
    embedding_f = torch.randn(1, embedding_dim)  
  
    # Quantize embeddings  
    binary_e = quant_module(embedding_e, binary=True)  
    binary_f = quant_module(embedding_f, binary=True)  
  
    # Compute Hamming distance  
    hamming_distance = (binary_e != binary_f).float().sum().item()  
    total_bits = binary_e.numel()  
    normalized_hamming_distance = hamming_distance / total_bits  
    print(f"Hamming Distance: {hamming_distance}")  
    print(f"Normalized Hamming Distance: {normalized_hamming_distance}")  
    # Expect normalized Hamming distance around 0.5 for random embeddings  
  
    # ------------------------- Test Case 4: Slightly Different Embeddings -------------------------  
    print("\nTest Case 4: Slightly Different Embeddings")  
    embedding_g = torch.randn(1, embedding_dim)  
    embedding_h = embedding_g + 0.01 * torch.randn(1, embedding_dim)  # Slight perturbation  
  
    # Quantize embeddings  
    binary_g = quant_module(embedding_g, binary=True)  
    binary_h = quant_module(embedding_h, binary=True)  
  
    # Compute Hamming distance  
    hamming_distance = (binary_g != binary_h).float().sum().item()  
    total_bits = binary_g.numel()  
    normalized_hamming_distance = hamming_distance / total_bits  
    print(f"Hamming Distance: {hamming_distance}")  
    print(f"Normalized Hamming Distance: {normalized_hamming_distance}")  
    # Expect low normalized Hamming distance    
