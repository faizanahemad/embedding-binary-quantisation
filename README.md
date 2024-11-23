# Next Steps
- Better Training with contrastive loss
- Sigmoid saturates so we need a better activation function
- Initialisation of thresholds plays an important role and hence multiple random initialisations are tried and the best one is chosen, Use a different test set for final reporting, while for choosing the right initialisation we use the validation set
- 

-----------
# embedding-binary-quantisation




We want to use mteb leaderboard to test our models.
The below 2 examples help us understand how to use MTEB library.

mteb_example_1
-----------

Let me help you create a comprehensive guide and code example for end-to-end testing of a retrieval model on MTEB datasets.

My understanding:
- Need to create end-to-end code for testing retrieval models on MTEB datasets
- Should include model creation, evaluation, and submission process
- Code needs to be well documented and easy to understand
- Should cover both built-in and custom model scenarios
- Need to handle the complete pipeline from model setup to result submission

My approach:
1. Create a complete pipeline class that handles:
   - Model initialization and setup
   - Dataset selection and loading
   - Evaluation process
   - Result saving and submission preparation
2. Include support for:
   - Different types of models (SentenceTransformers, custom models)
   - Various retrieval tasks
   - Caching for efficiency
   - Result analysis and visualization
3. Provide clear documentation and examples

Here's the comprehensive implementation:

```python
"""
MTEB Retrieval Model Testing Pipeline

This module provides a complete pipeline for testing and evaluating retrieval models
using the MTEB benchmark. It includes functionality for model setup, evaluation,
and result submission.
"""

import os
import json
from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime
import mteb
from sentence_transformers import SentenceTransformer
from mteb.encoder_interface import PromptType

class MTEBRetrievalTester:
    """
    A comprehensive class for testing retrieval models using MTEB benchmark.
    
    This class handles the complete pipeline from model initialization to
    result submission for MTEB retrieval tasks.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        output_dir: str = "mteb_results",
        cache_dir: Optional[str] = None,
        use_custom_model: bool = False
    ):
        """
        Initialize the MTEB retrieval tester.

        Args:
            model_name_or_path: Name or path of the model to test
            output_dir: Directory to save results
            cache_dir: Directory for caching embeddings (optional)
            use_custom_model: Whether to use custom model implementation
        """
        self.model_name = model_name_or_path
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.use_custom_model = use_custom_model
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            
        # Initialize model
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """
        Initialize the retrieval model based on configuration.
        
        Returns:
            Initialized model instance
        """
        if self.use_custom_model:
            return self.CustomRetrievalModel(self.model_name)
        else:
            # Use SentenceTransformer with default configuration
            model = SentenceTransformer(self.model_name)
            # Wrap with caching if cache directory is specified
            if self.cache_dir:
                from mteb.models.cache_wrapper import CachedEmbeddingWrapper
                model = CachedEmbeddingWrapper(model, cache_path=self.cache_dir)
            return model

    class CustomRetrievalModel:
        """
        Custom retrieval model implementation following MTEB interface.
        """
        
        def __init__(self, model_name: str):
            """
            Initialize custom model.

            Args:
                model_name: Name or path of the model to load
            """
            self.model = SentenceTransformer(model_name)
            
        def encode(
            self,
            sentences: List[str],
            task_name: str,
            prompt_type: Optional[PromptType] = None,
            **kwargs
        ) -> np.ndarray:
            """
            Encode input sentences into embeddings.

            Args:
                sentences: List of sentences to encode
                task_name: Name of the current task
                prompt_type: Type of prompt (query/passage)
                **kwargs: Additional encoding arguments

            Returns:
                np.ndarray: Array of embeddings
            """
            # Add task-specific prompts
            if prompt_type == "query":
                sentences = [f"Query: {s}" for s in sentences]
            elif prompt_type == "passage":
                sentences = [f"Passage: {s}" for s in sentences]
                
            # Get embeddings
            embeddings = self.model.encode(
                sentences,
                batch_size=kwargs.get('batch_size', 32),
                show_progress_bar=True,
                normalize_embeddings=True
            )
            return embeddings

    def select_retrieval_tasks(
        self,
        task_names: Optional[List[str]] = None,
        languages: Optional[List[str]] = None
    ) -> List[str]:
        """
        Select retrieval tasks based on criteria.

        Args:
            task_names: Specific task names to include
            languages: Languages to filter tasks by

        Returns:
            List of selected task names
        """
        if task_names:
            tasks = mteb.get_tasks(tasks=task_names)
        else:
            # Get all retrieval tasks
            tasks = mteb.get_tasks(
                task_types=["Retrieval"],
                languages=languages
            )
        return tasks

    def run_evaluation(
        self,
        task_names: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        batch_size: int = 32,
        save_predictions: bool = True
    ) -> Dict:
        """
        Run evaluation on selected retrieval tasks.

        Args:
            task_names: Specific tasks to evaluate on
            languages: Languages to include
            batch_size: Batch size for encoding
            save_predictions: Whether to save predictions

        Returns:
            Dictionary containing evaluation results
        """
        # Select tasks
        tasks = self.select_retrieval_tasks(task_names, languages)
        
        # Initialize evaluation
        evaluation = mteb.MTEB(tasks=tasks)
        
        # Setup output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(
            self.output_dir,
            f"{self.model_name}_{timestamp}"
        )
        
        # Run evaluation
        results = evaluation.run(
            self.model,
            output_folder=output_folder,
            eval_splits=["test"],
            save_predictions=save_predictions,
            encode_kwargs={
                "batch_size": batch_size,
                "show_progress_bar": True
            }
        )
        
        # Save results summary
        self._save_results_summary(results, output_folder)
        
        return results

    def _save_results_summary(self, results: Dict, output_folder: str):
        """
        Save evaluation results summary.

        Args:
            results: Evaluation results
            output_folder: Folder to save summary
        """
        summary_file = os.path.join(output_folder, "results_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Create markdown summary for model card
        self._create_model_card(output_folder)

    def _create_model_card(self, results_folder: str):
        """
        Create model card with evaluation results.

        Args:
            results_folder: Folder containing evaluation results
        """
        cmd = f"mteb create_meta --results_folder {results_folder} --output_path {results_folder}/model_card.md"
        os.system(cmd)

    def analyze_results(self, results: Dict) -> pd.DataFrame:
        """
        Analyze evaluation results.

        Args:
            results: Evaluation results dictionary

        Returns:
            DataFrame containing analysis
        """
        # Convert results to DataFrame for analysis
        analysis = []
        for task_name, task_results in results.items():
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    analysis.append({
                        'Task': task_name,
                        'Metric': metric,
                        'Value': value
                    })
        
        return pd.DataFrame(analysis)

# Example usage
def main():
    """
    Example of how to use the MTEBRetrievalTester.
    """
    # Initialize tester
    tester = MTEBRetrievalTester(
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        output_dir="mteb_results",
        cache_dir="embedding_cache",
        use_custom_model=True
    )
    
    # Run evaluation on specific tasks
    results = tester.run_evaluation(
        task_names=["NFCorpus", "MedicalQARetrieval"],
        batch_size=32,
        save_predictions=True
    )
    
    # Analyze results
    analysis_df = tester.analyze_results(results)
    print("\nEvaluation Results Analysis:")
    print(analysis_df)
    
    print("\nEvaluation complete! Results saved in:", tester.output_dir)
    print("To submit results to MTEB leaderboard:")
    print("1. Push results to: https://github.com/embeddings-benchmark/results")
    print("2. Wait for daily leaderboard refresh")

if __name__ == "__main__":
    main()
```

To use this code:

1. Installation:
```bash
pip install mteb sentence-transformers
```

2. Basic usage:
```python
# Initialize tester with default SentenceTransformer model
tester = MTEBRetrievalTester(
    model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
    output_dir="results"
)

# Run evaluation
results = tester.run_evaluation()
```

3. Using custom model:
```python
# Initialize with custom model implementation
tester = MTEBRetrievalTester(
    model_name_or_path="your-model-name",
    use_custom_model=True
)
```

Key Features:
1. Complete pipeline handling:
   - Model initialization
   - Task selection
   - Evaluation
   - Result saving
2. Support for:
   - Custom models
   - Embedding caching
   - Result analysis
   - Model card generation
3. Well-documented code with clear examples

The code provides:
- Flexible model initialization
- Comprehensive task selection
- Detailed result analysis
- Proper result saving and formatting
- Clear submission instructions

After running evaluation:
1. Results are saved in the specified output directory
2. A model card is generated automatically
3. Results can be analyzed using the provided analysis method
4. Follow the printed instructions for leaderboard submission

This implementation provides a complete end-to-end solution for testing retrieval models on MTEB datasets and preparing submissions for the leaderboard.

mteb_example_1 end
-----------

mteb_example_2
-----------

Let me help you understand how to use MTEB and create a submission for testing an embedding model on retrieval datasets.

My understanding of the task:
- Need to explain how to use MTEB library for evaluating embedding models
- Focus specifically on testing models on retrieval tasks
- Provide code examples for submitting and evaluating a model
- Cover both built-in and custom model scenarios

Here's my detailed plan:

1. Installation and Basic Setup:
- Install MTEB package
- Import required dependencies
- Initialize model and tasks

2. Model Evaluation Approaches:
- Using pre-trained models directly
- Implementing custom models
- Handling different types of encoders

3. Specific Steps for Retrieval Tasks:
- Select retrieval specific tasks
- Configure evaluation parameters
- Save and analyze results

Let me provide a comprehensive guide with code examples:

```python
# 1. Installation
# pip install mteb

# 2. Basic imports
import mteb
from sentence_transformers import SentenceTransformer
import numpy as np

# 3. Define a custom embedding model class
class CustomEmbeddingModel:
    """
    Custom embedding model class that implements the required MTEB interface.
    
    The model must implement an encode function that takes sentences and returns embeddings.
    """
    def __init__(self, model_name_or_path):
        """
        Initialize your model here.
        
        Args:
            model_name_or_path: Path or name of the model to load
        """
        # Initialize your model here
        self.model = SentenceTransformer(model_name_or_path)
        
    def encode(self, sentences: list[str], task_name: str, prompt_type: str = None, **kwargs) -> np.ndarray:
        """
        Encode input sentences into embeddings.
        
        Args:
            sentences: List of sentences to encode
            task_name: Name of the current task
            prompt_type: Type of prompt (query/passage for retrieval tasks)
            **kwargs: Additional encoding arguments
            
        Returns:
            np.ndarray: Array of embeddings
        """
        # Add task-specific prompts if needed
        if prompt_type == "query":
            sentences = [f"Query: {s}" for s in sentences]
        elif prompt_type == "passage":
            sentences = [f"Passage: {s}" for s in sentences]
            
        # Get embeddings
        embeddings = self.model.encode(
            sentences,
            batch_size=kwargs.get('batch_size', 32),
            show_progress_bar=kwargs.get('show_progress_bar', True),
            normalize_embeddings=True
        )
        return embeddings

# 4. Function to run evaluation on retrieval tasks
def evaluate_on_retrieval_tasks(model_name_or_path: str, 
                              task_names: list = None,
                              output_folder: str = "results",
                              save_predictions: bool = True):
    """
    Evaluate a model on MTEB retrieval tasks.
    
    Args:
        model_name_or_path: Model to evaluate
        task_names: List of retrieval tasks to evaluate on
        output_folder: Where to save results
        save_predictions: Whether to save predictions
    """
    # Initialize model
    model = CustomEmbeddingModel(model_name_or_path)
    
    # Get retrieval tasks
    if task_names is None:
        tasks = mteb.get_tasks(task_types=["Retrieval"])
    else:
        tasks = mteb.get_tasks(tasks=task_names)
    
    # Initialize evaluation
    evaluation = mteb.MTEB(tasks=tasks)
    
    # Run evaluation
    results = evaluation.run(
        model,
        output_folder=output_folder,
        eval_splits=["test"],  # Evaluate on test split
        save_predictions=save_predictions,
        encode_kwargs={
            "batch_size": 32,
            "show_progress_bar": True
        }
    )
    
    return results

# 5. Example usage with caching for efficiency
def evaluate_with_caching(model_name_or_path: str,
                         cache_folder: str = "cache"):
    """
    Evaluate model with embedding caching for efficiency.
    
    Args:
        model_name_or_path: Model to evaluate
        cache_folder: Where to cache embeddings
    """
    model = CustomEmbeddingModel(model_name_or_path)
    
    # Wrap model with caching
    from mteb.models.cache_wrapper import CachedEmbeddingWrapper
    cached_model = CachedEmbeddingWrapper(model, cache_path=cache_folder)
    
    # Select specific retrieval tasks
    tasks = [
        "NFCorpus",  # Medical domain
        "MedicalQARetrieval",  # Medical QA
        "BelebeleRetrieval"  # Multilingual
    ]
    
    evaluation = mteb.MTEB(tasks=mteb.get_tasks(tasks=tasks))
    results = evaluation.run(
        cached_model,
        output_folder="results_cached",
        save_predictions=True
    )
    
    return results

# Example usage:
if __name__ == "__main__":
    # 1. Evaluate using a pre-trained model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Run evaluation on all retrieval tasks
    results = evaluate_on_retrieval_tasks(
        model_name,
        output_folder=f"results/{model_name}",
        save_predictions=True
    )
    
    # 2. Run evaluation with caching
    cached_results = evaluate_with_caching(
        model_name,
        cache_folder="embedding_cache"
    )
```

Key points about the code:

1. Model Requirements:
- Must implement `encode` function
- Handle different prompt types for queries vs passages
- Return numpy array of embeddings

2. Evaluation Features:
- Supports multiple retrieval tasks
- Saves predictions for analysis
- Includes embedding caching for efficiency
- Configurable batch size and other parameters

3. Best Practices:
- Use test splits for final evaluation
- Save predictions for reproducibility
- Cache embeddings for frequently used corpora
- Add appropriate prompts for queries/passages

4. Additional Tips:
- For production use, consider implementing error handling
- Monitor memory usage with large datasets
- Use appropriate batch sizes for your GPU/CPU
- Consider implementing multi-GPU support for large-scale evaluation

To submit results to the MTEB leaderboard:
1. Run your model on desired tasks
2. Push results to the results repository
3. Wait for leaderboard refresh (updates daily)
4. Optionally add results to your model card using `mteb create_meta`

This implementation provides a flexible framework for evaluating embedding models on MTEB retrieval tasks while following best practices for reproducibility and efficiency.

mteb_example_2 end
-----------

A task list of MTEB retrieval tasks is also given below.
```

| Dataset Name | Languages | Number of Samples (Test Set) |  
|--------------|-----------|----------------------------|  
| JaqketRetrieval | Japanese | 115,226 |  
| BelebeleRetrieval | Multiple (90+ languages) | 521,866 |  
| CodeSearchNetRetrieval | Go, Java, JavaScript, PHP, Python, Ruby | 12,000 |  
| CodeTransOceanContest | C++, Python | 1,229 |  
| CodeTransOceanDL | Python | 996 |  
| AutoRAGRetrieval | Korean | 834 |  
| Core17InstructionRetrieval | English | 19,919 |  
| CosQA | English, Python | 21,104 |  
| ESCIReranking | English, Japanese, Spanish | 29,285 |  
| NFCorpus | English | 3,956 |  
| Touche2020Retrieval.v3 | English | 303,781 |  
| MedicalQARetrieval | English | Not specified |  
| MedicalRetrieval | Chinese | Not specified |  
| MintakaRetrieval | Arabic, German, French, Hindi, Italian, Japanese, Portuguese, Spanish | Not specified |  
| MrTidyRetrieval | Arabic, Bengali, English, Finnish, Indonesian, Japanese, Korean, Russian, Swahili, Telugu, Thai | Not specified |  
| MultiLongDocRetrieval | Arabic, Chinese, German, English, French, Hindi, Italian, Japanese, Korean, Portuguese, Russian, Spanish, Thai | Not specified |  
| DBPedia | English | Not specified |  
| FEVER | English | Not specified |  
| FQuADRetrieval | French | Not specified |  
| LegalQuAD | German | Not specified |  
| IndicQARetrieval | Multiple Indian Languages | Not specified |  
| MLQARetrieval | Arabic, German, English, Hindi, Spanish, Vietnamese, Chinese | Not specified |  
| NeuCLIR2022Retrieval | Persian, Russian, Chinese | Not specified |  
| XQuADRetrieval | Arabic, German, Greek, English, Hindi, Romanian, Russian, Spanish, Thai, Turkish, Vietnamese, Chinese | Not specified |  
```

```
| Dataset Name | Languages | Number of Samples (Test Set) |  
|--------------|-----------|----------------------------|  
| MedicalQARetrieval | English | Not specified |  
| MedicalRetrieval | Chinese | Not specified |  
| MintakaRetrieval | Arabic, German, French, Hindi, Italian, Japanese, Portuguese, Spanish | Not specified |  
| MrTidyRetrieval | Arabic, Bengali, English, Finnish, Indonesian, Japanese, Korean, Russian, Swahili, Telugu, Thai | Not specified |  
| MultiLongDocRetrieval | Arabic, Chinese, German, English, French, Hindi, Italian, Japanese, Korean, Portuguese, Russian, Spanish, Thai | Not specified |  
| ESCIReranking | English, Japanese, Spanish | 29,285 |  
| JaqketRetrieval | Japanese | 115,226 |  
| BelebeleRetrieval | Multiple (100+ languages) | 521,866 |  
| CodeEditSearchRetrieval | C, C++, Go, Java, JavaScript, PHP, Python, Ruby, Rust, Scala, Shell, Swift, TypeScript | 26,000 (Train) |  
| AutoRAGRetrieval | Korean | 834 |  
| FeedbackQARetrieval | English | Not specified |  
| LegalBenchConsumerContractsQA | English | Not specified |  
| LegalBenchCorporateLobbying | English | Not specified |  
| LegalQuAD | German | Not specified |  
| IndicQARetrieval | Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Oriya, Punjabi, Tamil, Telugu | Not specified |  
| CmedqaRetrieval | Chinese | Not specified |  
| BSARDRetrieval | French | Not specified |  
| CovidRetrieval | Chinese | Not specified |  
```

Now lets get to understanding what we need to do and writing the code after that.

We already have code for `QuantizationModuleStage1` , `QuantizationModuleStage2`, `similarity_preservation_loss`, `train_quantization_stage1`, `train_quantization_stage2`,  and the overall training code in one file called `quantization_modules.py` . We also saved these models to file system post training. 
Now we want another file called `evaluate_quantization_mteb.py`. 
 
We want to test before our optimisation how the initial embedding model does in a mteb retrieval task, then how does a method which just uses `0` as boundary to decide quantised value does (QuantizationModuleStage1 with zero init for self.thresholds without any training) and then post our stage 1 and stage 2 method we want to test again how model does for a given task. We would give the task names as a list to test. It should print out a markdown table which shows the results per task and per method and then save it as a csv as well.  Put all new requirements in `evaluate_quantization_mteb.py`.