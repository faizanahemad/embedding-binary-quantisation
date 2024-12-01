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
import sys
import mteb
import torch  
import torch.nn as nn  
import numpy as np  
from sentence_transformers import SentenceTransformer  
from mteb import MTEB, TaskResult  
from mteb.abstasks.AbsTask import AbsTask
from mteb.models.wrapper import Wrapper
from mteb.encoder_interface import Encoder
from config import base_model_name

from typing import List, Dict  
import pandas as pd  
from datetime import datetime  
  
# Assuming QuantizationModuleStage1 and QuantizationModuleStage2 are defined in quantization_modules.py  
from improved_quantisation_module import ImprovedQuantizationModule
from train import QuantizationModuleStage1, QuantizationModuleStage2, QuantizationModuleStage1WithScales, QuantizationModuleOneBitTwoBit
from config import save_dirs, test_modules
from common import OriginalEmbeddingModel, OriginalEmbeddingModelBinary, QuantizedEmbeddingModel
batch_size = 512
  
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
    
  
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
    # task_objects = MTEB(tasks=mteb.get_tasks(tasks=tasks))  
  
    # Run evaluation  
    eval_splits = ['test']  # Evaluate on test split only  
    evaluation = MTEB(tasks=mteb.get_tasks(tasks=tasks))  
    results = evaluation.run(  
        model,  
        eval_splits=eval_splits,  
        show_progress_bar=True,  
        # batch_size=batch_size,  
        encode_kwargs = {'batch_size': batch_size},
        output_folder=results_dir,
        overwrite_results=True  
    )  
  
    # Save results to a file  
    results_file = os.path.join(results_dir, f"{model_name}_results.json")  
    pd.DataFrame(results).to_json(results_file)  
  
    return results  
  

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
        
def print_main_score(df: pd.DataFrame):
    """
    Print the main score from the results DataFrame with task-wise breakdown and total average.
    
    Args:
        df (pd.DataFrame): DataFrame containing the results.
    """
    # Filter for main_score and create pivot table
    main_scores_df = df[df['Metric'] == 'main_score'].pivot_table(
        values='Score',
        index='Task',
        columns='Model',
        aggfunc='first'
    ).round(4)
    
    # Calculate averages across tasks for each model
    averages = main_scores_df.mean().round(4)
    
    # Add Total row with averages
    main_scores_df.loc['Total'] = averages
    
    print("\nMain Score Results:")
    print(main_scores_df.to_markdown())
  

def evaluate_single_task(task: str, model_name: str, embedding_model: SentenceTransformer, results_dir: str) -> Dict:
    """
    Evaluate a single MTEB task across all model variants.
    
    Args:
        task (str): Name of the MTEB task to evaluate
        embedding_model: Base SentenceTransformer model
        results_dir (str): Directory to save results
        
    Returns:
        Dict: Results for this task across all model variants
    """
    print(f"\nEvaluating task: {task}")
    task_results = {}
    try:
        embedding_dim = embedding_model.get_sentence_embedding_dimension()
    except:
        embedding_dim = embedding_model.config.hidden_size
    
    # 1. Original Model
    print("  Evaluating Original Model...")
    original_model = OriginalEmbeddingModel(model_name)
    # print(f"[DEBUG] Created model type: {type(original_model)}")  # Add this debug print
    results_original = evaluate_model_on_tasks(
        model=original_model,
        tasks=[task],
        model_name='Original',
        results_dir=results_dir
    )
    task_results['Original'] = results_original
    
    # 2. Original Binary
    print("  Evaluating Original Binary...")
    original_model_binary = OriginalEmbeddingModelBinary(model_name)
    results_original_binary = evaluate_model_on_tasks(
        model=original_model_binary,
        tasks=[task],
        model_name='OriginalBinary',
        results_dir=results_dir
    )
    task_results['OriginalBinary'] = results_original_binary

    # 2. Stage1 Untrained
    print("  Evaluating Stage1 Untrained...")
    quantization_module_stage1_zero = QuantizationModuleStage1(embedding_dim)
    quantization_module_stage1_zero.thresholds.data.fill_(0.0)
    quantized_model_stage1_zero = QuantizedEmbeddingModel(
        embedding_model=embedding_model,
        quantization_module=quantization_module_stage1_zero
    )
    results_stage1_zero = evaluate_model_on_tasks(
        model=quantized_model_stage1_zero,
        tasks=[task],
        model_name='QuantStage1_Untrained',
        results_dir=results_dir
    )
    task_results['QuantStage1_Untrained'] = results_stage1_zero
    
    if 'stage1' in test_modules:
        # 3. Stage1 Trained
        print("  Evaluating Stage1 Trained...")
        quantization_module_stage1_trained = QuantizationModuleStage1(embedding_dim)
        quantization_module_stage1_trained.load_state_dict(
            torch.load(f'saved_models/{save_dirs[0]}/quantization_stage1.pth', map_location=device, weights_only=False)
        )
        # print(quantization_module_stage1_trained.thresholds)
        quantization_module_stage1_trained.to(device)
        quantization_module_stage1_trained.eval()
        quantized_model_stage1_trained = QuantizedEmbeddingModel(
            embedding_model=embedding_model,
            quantization_module=quantization_module_stage1_trained
        )
        results_stage1_trained = evaluate_model_on_tasks(
            model=quantized_model_stage1_trained,
            tasks=[task],
            model_name='QuantStage1_Trained',
            results_dir=results_dir
        )
        task_results['QuantStage1_Trained'] = results_stage1_trained
    
    if 'stage1.1' in test_modules:
        # 3.5 Stage1.1 Trained
        print("  Evaluating Stage1.1 Trained...")
        quantization_module_stage1_1_trained = QuantizationModuleStage1WithScales(embedding_dim)
        quantization_module_stage1_1_trained.load_state_dict(
            torch.load(f'saved_models/{save_dirs[1]}/quantization_stage1_with_scales.pth', map_location=device, weights_only=False)
        )
        quantization_module_stage1_1_trained.to(device)
        quantization_module_stage1_1_trained.eval()
        quantized_model_stage1_1_trained = QuantizedEmbeddingModel(
            embedding_model=embedding_model,
            quantization_module=quantization_module_stage1_1_trained
        )
        results_stage1_1_trained = evaluate_model_on_tasks(
            model=quantized_model_stage1_1_trained,
            tasks=[task],
            model_name='QuantStage1.1_Trained',
            results_dir=results_dir
        )
        task_results['QuantStage1.1_Trained'] = results_stage1_1_trained
    
    if 'stage2' in test_modules:
        # 3.5 Stage2 Untrained  
        print("  Evaluating Stage2 Untrained...")
        quantization_module_stage2_untrained = QuantizationModuleStage2(embedding_dim)
        quantization_module_stage2_untrained.thresholds_first_half.data.fill_(0.0)
        quantization_module_stage2_untrained.thresholds_second_half.data.fill_(0.0)
        quantization_module_stage2_untrained.to(device)
        quantization_module_stage2_untrained.eval()
        quantized_model_stage2_untrained = QuantizedEmbeddingModel(
            embedding_model=embedding_model,
            quantization_module=quantization_module_stage2_untrained
        )
        results_stage2_untrained = evaluate_model_on_tasks(
            model=quantized_model_stage2_untrained,
            tasks=[task],
            model_name='QuantStage2_Untrained',
            results_dir=results_dir
        )
        task_results['QuantStage2_Untrained'] = results_stage2_untrained

        # 4. Stage2 Trained
        print("  Evaluating Stage2 Trained...")
        quantization_module_stage2_trained = QuantizationModuleStage2(embedding_dim)
        quantization_module_stage2_trained.load_state_dict(
            torch.load(f'saved_models/{save_dirs[2]}/quantization_stage2.pth', map_location=device, weights_only=False)
        )
        
        quantization_module_stage2_trained.to(device)
        quantization_module_stage2_trained.eval()
        quantized_model_stage2_trained = QuantizedEmbeddingModel(
            embedding_model=embedding_model,
            quantization_module=quantization_module_stage2_trained
        )
        results_stage2_trained = evaluate_model_on_tasks(
            model=quantized_model_stage2_trained,
            tasks=[task],
            model_name='QuantStage2_Trained',
            results_dir=results_dir
        )
        task_results['QuantStage2_Trained'] = results_stage2_trained
    
    if 'stage3' in test_modules:
        # 5. Stage3 Trained
        print("  Evaluating Stage3 Trained...")
        quantization_module_stage3_trained = ImprovedQuantizationModule(embedding_dim)
        quantization_module_stage3_trained.load_state_dict(
            torch.load(f'saved_models/{save_dirs[3]}/improved_quantization.pth', map_location=device, weights_only=False)
        )
        
        quantization_module_stage3_trained.to(device)
        quantization_module_stage3_trained.eval()
        quantized_model_stage3_trained = QuantizedEmbeddingModel(
            embedding_model=embedding_model,
            quantization_module=quantization_module_stage3_trained
        )
        results_stage3_trained = evaluate_model_on_tasks(
            model=quantized_model_stage3_trained,
            tasks=[task],
            model_name='QuantStage3_Trained',
            results_dir=results_dir
        )
        task_results['QuantStage3_Trained'] = results_stage3_trained
        
    if 'OneBitTwoBit' in test_modules:
        # 6. OneBitTwoBit Trained
        print("  Evaluating OneBitTwoBit Trained...")
        quantization_module_one_bit_two_bit = QuantizationModuleOneBitTwoBit(embedding_dim)
        quantization_module_one_bit_two_bit.load_state_dict(
            torch.load(f'saved_models/{save_dirs[4]}/one_bit_two_bit_thresholds.pth', map_location=device, weights_only=False)
        )
        quantization_module_one_bit_two_bit.to(device)
        quantization_module_one_bit_two_bit.eval()
        quantized_model_one_bit_two_bit = QuantizedEmbeddingModel(
            embedding_model=embedding_model,
            quantization_module=quantization_module_one_bit_two_bit
        )
        results_one_bit_two_bit = evaluate_model_on_tasks(
            model=quantized_model_one_bit_two_bit,
            tasks=[task],
            model_name='OneBitTwoBit_Trained',
            results_dir=results_dir
        )
        task_results['OneBitTwoBit_Trained'] = results_one_bit_two_bit
        
    # Print individual task results
    df_task_results = aggregate_results(task_results, [task])
    print(f"\nResults for task {task}:")
    print_markdown_table(df_task_results)
    
    return task_results

def main():
    """
    Main function to run the evaluation.
    """
    
    tasks = [
        "MedicalQARetrieval",
        "JaqketRetrieval",
        "FeedbackQARetrieval",
        "AutoRAGRetrieval",
        "IndicQARetrieval",
        "MLQARetrieval",
        "DBPedia",
        "FEVER",
        "MultiLongDocRetrieval",
        "MrTidyRetrieval",
        "CosQA",
        "Core17InstructionRetrieval",
        "CodeSearchNetRetrieval",
        "ESCIReranking",
    ]
    
    tasks = [
        "ArguAna",
        "ClimateFEVER", 
        "CQADupstackTexRetrieval",
        "DBPedia",
        "FEVER",
        "FiQA2018",
        "HotpotQA", 
        "MSMARCO",
        "NFCorpus",
        "NQ",
        "QuoraRetrieval",
        "SCIDOCS",
        "SciFact",
        "Touche2020",
        "TRECCOVID"
    ]
    
    tasks = [
        'NFCorpus',
        'TRECCOVID',
        'ArguAna',
        'SCIDOCS',
        'SciFact'
    ]

    results_dir = 'mteb_evaluation_results'
    os.makedirs(results_dir, exist_ok=True)

    # Initialize the original embedding model
    original_model_name = base_model_name
    embedding_model = SentenceTransformer(original_model_name)

    # Dictionary to store all results
    all_results = {
        'Original': [],
        'OriginalBinary': [],
        'QuantStage1_Untrained': [],
        'QuantStage1_Trained': [],
        'QuantStage1.1_Trained': [],
        'QuantStage2_Untrained': [],
        'QuantStage2_Trained': [],
        'QuantStage3_Trained': [],
        'OneBitTwoBit_Trained': []
    }

    # Evaluate each task individually
    for task in tasks:
        task_results = evaluate_single_task(task, original_model_name, embedding_model, results_dir)
        
        # Accumulate results
        for model_name, results in task_results.items():
            all_results[model_name].extend(results)

    # Aggregate and display final results
    print("\nFinal Results Across All Tasks:")
    df_results = aggregate_results(all_results, tasks)
    
    # Save results to CSV
    csv_file = os.path.join(results_dir, 'evaluation_results.csv')
    df_results.to_csv(csv_file, index=False)
    print(f"\nResults saved to {csv_file}")

    # Print final markdown table
    print("\n### Final Evaluation Results:\n")
    print_markdown_table(df_results)
    
    # Print main score
    print("\n### Main Score Results:\n")
    print_main_score(df_results)

if __name__ == '__main__':  
    main()  
