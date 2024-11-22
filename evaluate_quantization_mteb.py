# evaluate_quantization_mteb.py  
  
"""  
Evaluate Quantized Embedding Models on MTEB Retrieval Tasks  
  
This script evaluates the original and quantized embedding models on specified MTEB retrieval tasks.  
It compares the performance of the following models:  
1. Original embedding model (without quantization)  
2. QuantizationModuleStage1 with zero thresholds (untrained)  
3. QuantizationModuleStage1 after training (trained thresholds)  
4. QuantizationModuleStage2 after training  
  
The results are output as a markdown table and saved as a CSV file.  
  
Usage:  
    python evaluate_quantization_mteb.py  
  
Requirements:  
    Install the required packages:  
        pip install torch sentence-transformers mteb  
"""  
  
import os  
import torch  
import torch.nn as nn  
import numpy as np  
from sentence_transformers import SentenceTransformer  
from mteb import MTEB, TaskResult  

from typing import List, Dict  
import pandas as pd  
from datetime import datetime  
  
# Assuming QuantizationModuleStage1 and QuantizationModuleStage2 are defined in quantization_modules.py  
from quantization_modules import QuantizationModuleStage1, QuantizationModuleStage2  
  
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OriginalEmbeddingModel:  
    """  
    Original embedding model without any quantization.  
  
    This model uses the pre-trained embedding model directly.  
    """  
    def __init__(self, model_name: str):  
        self.model = SentenceTransformer(model_name)  
        self.model.to(device)
  
    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:  
        """  
        Encode sentences into embeddings.  
  
        Args:  
            sentences (List[str]): List of sentences to encode.  
  
        Returns:  
            np.ndarray: Array of embeddings.  
        """  
        embeddings = self.model.encode(  
            sentences,  
            show_progress_bar=kwargs.get('show_progress_bar', False),  
            batch_size=kwargs.get('batch_size', 32),  
            normalize_embeddings=True  
        )  
        return embeddings  
  
class QuantizedEmbeddingModelStage1:  
    """  
    Embedding model with QuantizationModuleStage1 applied.  
  
    This model applies Stage 1 quantization to the embeddings.  
    """  
    def __init__(self, embedding_model: SentenceTransformer, quantization_module: QuantizationModuleStage1):  
        self.embedding_model = embedding_model  
        self.embedding_model.to(device)  # Move embedding model to GPU
        self.quantization_module = quantization_module
        self.quantization_module.to(device)  # Move quantization module to GPU

  
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
            batch_size=kwargs.get('batch_size', 32),  
            normalize_embeddings=False  # Do not normalize before quantization  
        )  
        embeddings = torch.tensor(embeddings)  
        embeddings = embeddings.to(device)
        # Apply quantization  
        with torch.no_grad():  
            quantized_embeddings = self.quantization_module(embeddings, binary=True).cpu().numpy()  
        # Optionally normalize embeddings  
        if kwargs.get('normalize_embeddings', True):  
            quantized_embeddings = quantized_embeddings / np.linalg.norm(quantized_embeddings, axis=1, keepdims=True)  
        return quantized_embeddings  
  
class QuantizedEmbeddingModelStage2:  
    """  
    Embedding model with QuantizationModuleStage2 applied.  
  
    This model applies Stage 2 quantization to the embeddings.  
    """  
    def __init__(self, embedding_model: SentenceTransformer, quantization_module: QuantizationModuleStage2):  
        self.embedding_model = embedding_model  
        self.embedding_model.to(device)  # Move embedding model to GPU
        self.quantization_module = quantization_module
        self.quantization_module.to(device)  # Move quantization module to GPU

  
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
            batch_size=kwargs.get('batch_size', 32),  
            normalize_embeddings=False  # Do not normalize before quantization  
        )  
        embeddings = torch.tensor(embeddings).to(device) 
        # Apply quantization  
        with torch.no_grad():  
            quantized_embeddings = self.quantization_module(embeddings, binary=True).cpu().numpy()  
        # Optionally normalize embeddings  
        if kwargs.get('normalize_embeddings', True):  
            quantized_embeddings = quantized_embeddings / np.linalg.norm(quantized_embeddings, axis=1, keepdims=True)  
        return quantized_embeddings  
  
def evaluate_model_on_tasks(model, tasks: List[str], model_name: str, results_dir: str) -> Dict:  
    """  
    Evaluate a model on specified MTEB retrieval tasks.  
  
    Args:  
        model: The embedding model to evaluate.  
        tasks (List[str]): List of MTEB task names.  
        model_name (str): Name of the model (used in result reporting).  
        results_dir (str): Directory to save the results.  
  
    Returns:  
        Dict: A dictionary containing evaluation results.  
    """  
    # Create output directory if it doesn't exist  
    os.makedirs(results_dir, exist_ok=True)  
  
    # Initialize MTEB with the specified tasks  
    task_objects = MTEB(tasks=tasks)  
  
    # Run evaluation  
    eval_splits = ['test']  # Evaluate on test split only  
    evaluation = MTEB(tasks=tasks)  
    results = evaluation.run(  
        model,  
        eval_splits=eval_splits,  
        show_progress_bar=True,  
        batch_size=32,  
        output_folder=results_dir  
    )  
  
    # Save results to a file  
    results_file = os.path.join(results_dir, f"{model_name}_results.json")  
    pd.DataFrame(results).to_json(results_file)  
  
    return results  
  
def print_markdown_table(df: pd.DataFrame):  
    """  
    Print a markdown table from the results DataFrame.  
  
    Args:  
        df (pd.DataFrame): DataFrame containing the results.  
    """  
    pivot_df = df.pivot_table(values='Score', index=['Task', 'Metric'], columns='Model')  
    print(pivot_df.to_markdown())  
    
    



def aggregate_results(all_results: Dict[str, List[TaskResult]], tasks: List[str]) -> pd.DataFrame:
    """
    Aggregate results from all models into a DataFrame.

    Args:
        all_results (Dict[str, List]): Dictionary containing results from all models.
        tasks (List[str]): List of task names.

    Returns:
        pd.DataFrame: DataFrame with aggregated results.
    """
    data = []
    for model_name, task_results in all_results.items():
        # task_results is a list of TaskResult objects
        for task_result in task_results:
            # Extract scores from the TaskResult object
            scores = task_result.scores
            # Extract key metrics we want to display
            metrics = {
                'main_score': scores['main_score'],
                'map_at_1': scores['map_at_1'],
                'map_at_10': scores['map_at_10'],
                'ndcg_at_10': scores['ndcg_at_10']
            }
            
            for metric_name, metric_value in metrics.items():
                data.append({
                    'Model': model_name,
                    'Task': task_result.task_name,
                    'Metric': metric_name,
                    'Score': metric_value
                })
    
    return pd.DataFrame(data)

def aggregate_results(all_results: Dict[str, List[TaskResult]], tasks: List[str]) -> pd.DataFrame:
    """
    Aggregate results from all models into a DataFrame.

    Args:
        all_results (Dict[str, List]): Dictionary containing results from all models.
        tasks (List[str]): List of task names.

    Returns:
        pd.DataFrame: DataFrame with aggregated results.
    """
    data = []
    for model_name, task_results in all_results.items():
        # task_results is a list of TaskResult objects
        for task_result in task_results:
            # Extract scores from the TaskResult object
            scores = task_result.scores
            
            # Print available scores for debugging
            print(f"Available scores for {model_name}, {task_result.task_name}:")
            print(scores)
            
            # Extract metrics with safe get operation
            metrics = {
                'ndcg_at_10': scores.get('ndcg_at_10', None),
                'map_at_1': scores.get('map_at_1', None),
                'map_at_10': scores.get('map_at_10', None),
                'mrr_at_10': scores.get('mrr_at_10', None)  # Changed from main_score to mrr_at_10
            }
            
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:  # Only add if metric exists
                    data.append({
                        'Model': model_name,
                        'Task': task_result.task_name,
                        'Metric': metric_name,
                        'Score': metric_value
                    })
    
    return pd.DataFrame(data)


def aggregate_results(all_results: Dict[str, List[TaskResult]], tasks: List[str]) -> pd.DataFrame:
    """
    Aggregate results from all models into a DataFrame.

    Args:
        all_results (Dict[str, List]): Dictionary containing results from all models.
        tasks (List[str]): List of task names.

    Returns:
        pd.DataFrame: DataFrame with aggregated results.
    """
    data = []
    
    # Debug print to see the structure of all_results
    print("All results structure:")
    print(all_results)
    
    for model_name, task_results in all_results.items():
        print(f"\nProcessing model: {model_name}")
        print(f"Task results type: {type(task_results)}")
        print(f"Task results content: {task_results}")
        
        # task_results is a list of TaskResult objects
        for task_result in task_results:
            print(f"\nProcessing task result: {task_result}")
            # Extract scores from the TaskResult object
            scores = task_result.scores
            print(f"Scores: {scores}")
            
            # Extract metrics with safe get operation
            metrics = {
                'ndcg_at_10': scores.get('ndcg_at_10', None),
                'map_at_1': scores.get('map_at_1', None),
                'map_at_10': scores.get('map_at_10', None),
                'mrr_at_10': scores.get('mrr_at_10', None)
            }
            
            print(f"Extracted metrics: {metrics}")
            
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:  # Only add if metric exists
                    entry = {
                        'Model': model_name,
                        'Task': task_result.task_name,
                        'Metric': metric_name,
                        'Score': metric_value
                    }
                    print(f"Adding entry: {entry}")
                    data.append(entry)
    
    df = pd.DataFrame(data)
    print("\nFinal DataFrame:")
    print(df)
    return df


def aggregate_results(all_results: Dict[str, List], tasks: List[str]) -> pd.DataFrame:
    """
    Aggregate results from all models into a DataFrame.

    Args:
        all_results (Dict[str, List]): Dictionary containing results from all models.
        tasks (List[str]): List of task names.

    Returns:
        pd.DataFrame: DataFrame with aggregated results.
    """
    data = []
    for model_name, task_results in all_results.items():
        # task_results is a list of TaskResult objects
        for task_result in task_results:
            # Extract scores from the TaskResult object - handle nested structure
            test_scores = task_result.scores['test'][0]  # Get first element of test scores
            
            # Extract metrics
            metrics = {
                'main_score': test_scores.get('main_score', None),
                'ndcg_at_10': test_scores.get('ndcg_at_10', None),
                'map_at_1': test_scores.get('map_at_1', None),
                'map_at_10': test_scores.get('map_at_10', None),
                'mrr_at_10': test_scores.get('mrr_at_10', None)
            }
            
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:  # Only add if metric exists
                    data.append({
                        'Model': model_name,
                        'Task': task_result.task_name,
                        'Metric': metric_name,
                        'Score': metric_value
                    })
    
    df = pd.DataFrame(data)
    return df


def print_markdown_table(df: pd.DataFrame):
    """
    Print a markdown table from the results DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the results.
    """
    # If df is loaded from CSV, parse the Score column if it contains JSON strings
    if isinstance(df['Score'].iloc[0], str) and df['Score'].iloc[0].startswith('['):
        # Extract main_score from the JSON string
        df['Score'] = df['Score'].apply(lambda x: eval(x)[0]['main_score'])
    
    # Create pivot table with rounded values
    pivot_df = df.pivot_table(
        values='Score',
        index=['Task', 'Metric'],
        columns='Model',
        aggfunc='first'  # Use 'first' since we don't need to aggregate
    ).round(4)
    
    print(pivot_df.to_markdown())
    
def print_markdown_table(df: pd.DataFrame):
    """
    Print a markdown table from the results DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the results.
    """
    # Print the DataFrame structure for debugging
    print("DataFrame columns:", df.columns)
    print("DataFrame head:", df.head())
    
    try:
        # Create pivot table with rounded values
        pivot_df = df.pivot_table(
            values='Score',
            index=['Task', 'Metric'],
            columns='Model',
            aggfunc='first'  # Use 'first' since we don't need to aggregate
        ).round(4)
        
        # Sort the index for better readability
        pivot_df = pivot_df.sort_index(level=['Task', 'Metric'])
        
        print("\nEvaluation Results:")
        print(pivot_df.to_markdown())
        
    except KeyError as e:
        print(f"Error: Column not found - {e}")
        print("Available columns:", df.columns)
  
def main():  
    """  
    Main function to run the evaluation.  
    """  
    # List of MTEB tasks to evaluate on  
    tasks = [  
        'NFCorpus',  
        'TREC-COVID',  
        'ArguAna',  
        'SciDocs',  
        'SciFact'  
        # Add more tasks as needed  
    ]  
  
    # Directory to save results  
    results_dir = 'mteb_evaluation_results'  
    os.makedirs(results_dir, exist_ok=True)  
  
    # Initialize the original embedding model  
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'  
    embedding_model = SentenceTransformer(model_name)  
  
    # Dictionary to store all results  
    all_results = {}  
  
    # 1. Evaluate Original Embedding Model  
    print("Evaluating Original Embedding Model...")  
    original_model = OriginalEmbeddingModel(model_name)  
    results_original = evaluate_model_on_tasks(  
        model=original_model,  
        tasks=tasks,  
        model_name='Original',  
        results_dir=results_dir  
    )  
    all_results['Original'] = results_original  
  
    # 2. Evaluate QuantizationModuleStage1 with Zero Thresholds (Untrained)  
    print("Evaluating QuantizationModuleStage1 with Zero Thresholds (Untrained)...")  
    # Initialize QuantizationModuleStage1 with zero thresholds  
    embedding_dim = embedding_model.get_sentence_embedding_dimension()  
    quantization_module_stage1_zero = QuantizationModuleStage1(embedding_dim)  
    
    # Ensure thresholds are initialized to zero  
    quantization_module_stage1_zero.thresholds.data.fill_(0.0)  
    quantized_model_stage1_zero = QuantizedEmbeddingModelStage1(  
        embedding_model=embedding_model,  
        quantization_module=quantization_module_stage1_zero  
    )  
    results_stage1_zero = evaluate_model_on_tasks(  
        model=quantized_model_stage1_zero,  
        tasks=tasks,  
        model_name='QuantStage1_Untrained',  
        results_dir=results_dir  
    )  
    all_results['QuantStage1_Untrained'] = results_stage1_zero  
  
    # 3. Evaluate QuantizationModuleStage1 after Training  
    print("Evaluating QuantizationModuleStage1 after Training...")  
    # Load the trained QuantizationModuleStage1  
    quantization_module_stage1_trained = QuantizationModuleStage1(embedding_dim)  
    # load the model
    quantization_module_stage1_trained.load_state_dict(torch.load('saved_models/run_20241121_175510/quantization_stage1.pth', map_location=device))
    quantization_module_stage1_trained.to(device)
    quantization_module_stage1_trained.eval()  
    quantized_model_stage1_trained = QuantizedEmbeddingModelStage1(  
        embedding_model=embedding_model,  
        quantization_module=quantization_module_stage1_trained  
    )  
    results_stage1_trained = evaluate_model_on_tasks(  
        model=quantized_model_stage1_trained,  
        tasks=tasks,  
        model_name='QuantStage1_Trained',  
        results_dir=results_dir  
    )  
    all_results['QuantStage1_Trained'] = results_stage1_trained  
  
    # 4. Evaluate QuantizationModuleStage2 after Training  
    print("Evaluating QuantizationModuleStage2 after Training...")  
    # Load the trained QuantizationModuleStage2  
    quantization_module_stage2_trained = QuantizationModuleStage2(embedding_dim)  
    quantization_module_stage2_trained.load_state_dict(torch.load('saved_models/run_20241121_175516/quantization_stage2.pth', map_location=device))  
    quantization_module_stage2_trained.to(device)
    quantization_module_stage2_trained.eval()  
    quantized_model_stage2_trained = QuantizedEmbeddingModelStage2(  
        embedding_model=embedding_model,  
        quantization_module=quantization_module_stage2_trained  
    )  
    results_stage2_trained = evaluate_model_on_tasks(  
        model=quantized_model_stage2_trained,  
        tasks=tasks,  
        model_name='QuantStage2_Trained',  
        results_dir=results_dir  
    )  
    all_results['QuantStage2_Trained'] = results_stage2_trained  
  
    # Aggregate Results  
    print("Aggregating results...") 
    print(all_results)
    df_results = aggregate_results(all_results, tasks)  
  
    # Save results to CSV  
    csv_file = os.path.join(results_dir, 'evaluation_results.csv')  
    df_results.to_csv(csv_file, index=False)  
    print(f"Results saved to {csv_file}")  
    print(df_results)
  
    # Print Markdown Table  
    print("\n### Evaluation Results:\n")  
    print_markdown_table(df_results)  
  
if __name__ == '__main__':  
    main()  
