import torch  
from torch.utils.data import Dataset  
from datasets import load_dataset, DatasetDict, concatenate_datasets  
import random  
  
class CombinedSimilarityDataset(Dataset):  
    """  
    Combined Dataset for Training Quantization Modules.  
    This dataset loads and combines multiple similarity and retrieval datasets.  
    Positive pairs (e.g., query-passage) are placed consecutively.  
    """  
  
    def __init__(self, tokenizer, max_length=128, max_samples_per_dataset=None):  
        """  
        Initializes the dataset by loading and combining multiple datasets.  
  
        Args:  
            max_samples_per_dataset (int, optional): Maximum number of samples to load from each dataset.  
                                                     If None, load all samples.  
        """  
        super().__init__()  
        self.samples = []  
        self.tokenizer = tokenizer
        self.max_length = max_length
  
        # List of datasets to load along with their specific configurations  
        dataset_configs = [  
            # MS MARCO Passage Ranking Dataset  
            {  
                'name': 'ms_marco',  
                'config': 'v2.1',  
                'split': 'train',  
                'loader_function': self.load_ms_marco  
            },  
            # Quora Question Pairs (QQP)  
            {  
                'name': 'quora',  
                'config': None,  
                'split': 'train',  
                'loader_function': self.load_quora  
            },  
            # Semantic Textual Similarity Benchmark (STS-B)  
            {  
                'name': 'glue',  
                'config': 'stsb',  
                'split': 'train',  
                'loader_function': self.load_stsb  
            },  
            # SNLI (Stanford Natural Language Inference)  
            {  
                'name': 'snli',  
                'config': None,  
                'split': 'train',  
                'loader_function': self.load_snli  
            },  
            # Add more datasets as needed  
        ]  
  
        for config in dataset_configs:  
            dataset = load_dataset(config['name'], config['config'], split=config['split'])  
            if max_samples_per_dataset:  
                dataset = dataset.shuffle(seed=42).select(range(max_samples_per_dataset))  
            samples = config['loader_function'](dataset)  
            self.samples.extend(samples)  
  
        # Shuffle the combined samples  
        random.shuffle(self.samples)  
  
    def load_ms_marco(self, dataset):  
        """  
        Loads and processes the MS MARCO dataset.  
  
        Args:  
            dataset (Dataset): Hugging Face dataset object for MS MARCO.  
  
        Returns:  
            List of tuples containing positive pairs.  
        """  
        samples = []  
        for example in dataset:  
            query = example['query']  
            passages = example['passages']  
            try:
                correct_index = passages['is_selected'].index(1)
                positive_passage = passages['passage_text'][correct_index]
            except ValueError:
                positive_passage = passages['passage_text'][0]
                # print(f"No positive passage found for query: {query}")
            
            samples.append((query, positive_passage))  
        return samples  
  
    def load_quora(self, dataset):  
        """  
        Loads and processes the Quora Question Pairs dataset.  
  
        Args:  
            dataset (Dataset): Hugging Face dataset object for QQP.  
  
        Returns:  
            List of tuples containing positive pairs.  
        """  
        samples = []  
        for example in dataset:  
            if example['is_duplicate']:  
                question1 = example['questions']['text'][0]  
                question2 = example['questions']['text'][1]  
                samples.append((question1, question2))  
        return samples  
  
    def load_stsb(self, dataset):  
        """  
        Loads and processes the STS-B dataset.  
  
        Args:  
            dataset (Dataset): Hugging Face dataset object for STS-B.  
  
        Returns:  
            List of tuples containing positive pairs.  
        """  
        samples = []  
        for example in dataset:  
            score = example['label']  
            # Use a threshold to consider as positive pairs  
            if score >= 4.0:  # STS-B scores range from 0 to 5  
                sentence1 = example['sentence1']  
                sentence2 = example['sentence2']  
                samples.append((sentence1, sentence2))  
        return samples  
  
    def load_snli(self, dataset):  
        """  
        Loads and processes the SNLI dataset.  
  
        Args:  
            dataset (Dataset): Hugging Face dataset object for SNLI.  
  
        Returns:  
            List of tuples containing positive pairs.  
        """  
        samples = []  
        for example in dataset:  
            if example['label'] == 0:  # Label 0 corresponds to 'entailment'  
                premise = example['premise']  
                hypothesis = example['hypothesis']  
                samples.append((premise, hypothesis))  
        return samples  
  
    def __len__(self):  
        return len(self.samples) * 2  # Because we place positive pairs consecutively  
  
    def __getitem__(self, idx):  
        """  
        Retrieves the item at the given index.  
  
        Args:  
            idx (int): Index of the item.  
  
        Returns:  
            Dict containing input data for the model.  
        """  
        pair_idx = idx // 2  
        is_first_in_pair = idx % 2 == 0  
  
        text_a, text_b = self.samples[pair_idx]
        text = text_a if is_first_in_pair else text_b

        # Tokenize the text
        encoded = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )
        
        # Remove the batch dimension added by return_tensors='pt'
        return {k: v.squeeze(0) for k, v in encoded.items()}


class GeneralizedPairDataset(Dataset):  
    """  
    Generalized Dataset that can handle various types of datasets and combine them.  
  
    The dataset ensures that positive pairs (e.g., query and corresponding passage) come consecutively.  
    """  
  
    def __init__(self, datasets_list, tokenizer, max_length=128):  
        """  
        Initializes the dataset.  
  
        Args:  
            datasets_list (list): List of dataset names to be loaded from Hugging Face datasets.  
            tokenizer (transformers.Tokenizer): Tokenizer to be used.  
            max_length (int): Maximum token length for truncation/padding.  
        """  
        self.examples = []  
        self.tokenizer = tokenizer  
        self.max_length = max_length  
  
        for dataset_name in datasets_list:  
            print(f"Loading dataset: {dataset_name}")  
            dataset = load_dataset(dataset_name)  
  
            # Process dataset based on its format  
            if dataset_name == 'ms_marco':  
                # Handle MS MARCO dataset  
                self.process_ms_marco(dataset)  
            elif dataset_name == 'glue':  
                # Handle GLUE datasets like QQP  
                self.process_qqp(dataset['qqp'])  
            elif dataset_name == 'quora':  
                # Handle Quora Question Pairs  
                self.process_quora(dataset)  
            # Add more datasets as needed  
            else:  
                print(f"Dataset {dataset_name} not specifically handled. Skipping.")  
  
    def process_ms_marco(self, dataset):  
        """  
        Processes the MS MARCO dataset.  
  
        Args:  
            dataset (DatasetDict): MS MARCO dataset loaded from Hugging Face datasets.  
        """  
        # Assuming 'train' split and 'query' and 'passage' fields  
        for item in dataset['train']:  
            query = item['query']  
            passage = item['passage']  
            # Append query and passage as consecutive pairs  
            self.examples.append(query)  
            self.examples.append(passage)  
  
    def process_qqp(self, dataset):  
        """  
        Processes the QQP dataset.  
  
        Args:  
            dataset (Dataset): QQP dataset loaded from Hugging Face datasets.  
        """  
        for item in dataset:  
            if item['label'] == 1:  # Only positive pairs  
                question1 = item['question1']  
                question2 = item['question2']  
                # Append question pairs as consecutive items  
                self.examples.append(question1)  
                self.examples.append(question2)  
  
    def process_quora(self, dataset):  
        """  
        Processes the Quora Question Pairs dataset.  
  
        Args:  
            dataset (DatasetDict): Quora dataset loaded from Hugging Face datasets.  
        """  
        # Similar to QQP processing  
        for item in dataset['train']:  
            if item['is_duplicate'] == 1:  
                question1 = item['questions']['text'][item['questions']['id'].index(item['qid1'])]  
                question2 = item['questions']['text'][item['questions']['id'].index(item['qid2'])]  
                self.examples.append(question1)  
                self.examples.append(question2)  
  
    def __len__(self):  
        return len(self.examples)  
  
    def __getitem__(self, idx):  
        text = self.examples[idx]  
        encoded = self.tokenizer(  
            text,  
            return_tensors='pt',  
            truncation=True,  
            padding='max_length',  
            max_length=self.max_length  
        )  
        return encoded  
