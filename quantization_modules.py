import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
from transformers import AutoTokenizer, AutoModel  
from torch.utils.data import DataLoader, Dataset  
import numpy as np  
from config import base_model_name

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
    date_time = datetime.now().strftime('%Y%m%d_%H%M%S')  
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
            self.thresholds = nn.Parameter(torch.zeros(embedding_dim) + torch.randn(embedding_dim) * 0.01)  
  
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
        k = 10  # Temperature parameter controlling the steepness  
        quantized_embeddings = torch.sigmoid(k * v)  
  
        return quantized_embeddings if not binary else (quantized_embeddings > 0.5).int()  
  
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
        self.thresholds_first_half = nn.Parameter(torch.zeros(self.half_dim) + torch.randn(self.half_dim) * 0.01)

        # Thresholds for second half (pairs of thresholds)
        self.thresholds_second_half = nn.Parameter(torch.zeros(self.half_dim // 2, 2) + torch.randn(self.half_dim // 2, 2) * 0.01)

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
        k = 10
        quantized_first_half = torch.sigmoid(k * v_first)

        # Second half: properly reshape for pairs
        embeddings_second_half = embeddings[:, self.half_dim:]
        # Reshape to (batch_size, num_pairs, 2)
        embeddings_pairs = embeddings_second_half.view(-1, self.half_dim // 2, 2)
        
        # Reshape thresholds to match embedding pairs (add batch dimension)
        thresholds_pairs = self.thresholds_second_half.unsqueeze(0)  # Shape: (1, num_pairs, 2)

        # Rest of the processing remains the same
        v_pairs = embeddings_pairs - thresholds_pairs
        s_pairs = torch.sigmoid(k * v_pairs)
        combined_outputs = 1 - (1 - s_pairs[:, :, 0]) * (1 - s_pairs[:, :, 1])

        quantized_embeddings = torch.cat([quantized_first_half, combined_outputs], dim=1)
        return quantized_embeddings if not binary else (quantized_embeddings > 0.5).int()
  

# Loss function  
  
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
  
# Dataset preparation (Example using random data for demonstration)  
  
class ExampleDataset(Dataset):  
    """  
    Example dataset for demonstration purposes.  
    Replace this with actual data loading from MS MARCO or other dataset.  
    """  
    def __init__(self, texts):  
        self.texts = texts  
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)  
  
    def __len__(self):  
        return len(self.texts)  
  
    def __getitem__(self, idx):  
        text = self.texts[idx]  
        encoded = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)  
        return encoded  
  
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
    optimizer = optim.Adam(quantization_module.parameters(), lr=0.01)  
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
  
            loss = similarity_preservation_loss(embeddings, quantized_embeddings)  
  
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
  
            total_loss += loss.item()  
  
        avg_loss = total_loss / len(dataloader)  
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')  
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
    optimizer = optim.Adam(quantization_module.parameters(), lr=0.01)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_model.eval()  
    quantization_module.train()  
  
    for epoch in range(num_epochs):  
        total_loss = 0.0  
        for batch in dataloader:  
            input_ids = batch['input_ids'].squeeze(1).to(device)  
            attention_mask = batch['attention_mask'].squeeze(1).to(device)  
  
            with torch.no_grad():  
                embeddings = embedding_model(input_ids=input_ids, attention_mask=attention_mask)  
                embeddings = embeddings.last_hidden_state.mean(dim=1)  
  
            quantized_embeddings = quantization_module(embeddings)  
  
            loss = similarity_preservation_loss(embeddings, quantized_embeddings)  
  
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
  
            total_loss += loss.item()  
  
        avg_loss = total_loss / len(dataloader)  
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')  
    save_dir = create_save_directory()  
    save_quantization_module(quantization_module, save_dir, 'quantization_stage2')  
    
  
# Example usage  
  
def main():  
    # Load the frozen embedding model  
    embedding_model = AutoModel.from_pretrained(base_model_name)  
    embedding_dim = embedding_model.config.hidden_size  # e.g., 384  
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
  
    # Freeze the embedding model parameters  
    for param in embedding_model.parameters():  
        param.requires_grad = False  
  
    # Prepare the dataset and dataloader  
    # texts = ["This is a sample sentence.", "Another example text.", "More data for training."]  
    # dataset = ExampleDataset(texts)  
    
    dataset = CombinedSimilarityDataset(tokenizer, max_length=128, max_samples_per_dataset=1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False) 
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    embedding_model.to(device)
    # Stage 1: Train per-dimension thresholds  
    quantization_module_stage1 = QuantizationModuleStage1(embedding_dim)  
    quantization_module_stage1.to(device)
    train_quantization_stage1(embedding_model, quantization_module_stage1, dataloader, num_epochs=5)  
  
    # Stage 2: Train with combined dimensions  
    quantization_module_stage2 = QuantizationModuleStage2(embedding_dim)  
    quantization_module_stage2.to(device)
    train_quantization_stage2(embedding_model, quantization_module_stage2, dataloader, num_epochs=5)  
  
    # Inference example  
    embedding_model.eval()  
    quantization_module_stage2.eval()  
  
    # Example input  
    input_text = "New input text for inference."  
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)  
    encoded_input = tokenizer(input_text, return_tensors='pt')  
  
    with torch.no_grad():  
        embedding = embedding_model(**encoded_input.to(device))  
        embedding = embedding.last_hidden_state.mean(dim=1)  
  
        quantized_embedding = quantization_module_stage2(embedding)  
  
        # Apply hard thresholding for binary output  
        binary_embedding = (quantized_embedding > 0.5).int()  
        print("Binary Embedding:", binary_embedding)  
        
        

  
if __name__ == '__main__':  
    main()  
