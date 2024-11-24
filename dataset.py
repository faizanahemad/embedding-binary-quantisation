import traceback
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
        self.max_samples_per_dataset = max_samples_per_dataset
  
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
            
            # Natural Questions (Query-Passage)
            {
                'name': 'natural_questions',
                'config': 'default',
                'split': 'train',
                'loader_function': self.load_natural_questions
            },
            # MRPC (Microsoft Research Paraphrase Corpus)
            {
                'name': 'glue',
                'config': 'mrpc',
                'split': 'train',
                'loader_function': self.load_mrpc
            },
            # PAWS (Paraphrase Adversaries from Word Scrambling)
            {
                'name': 'paws',
                'config': 'labeled_final',
                'split': 'train',
                'loader_function': self.load_paws
            },
            # MultiNews (Document-Document Similarity)
            {
                'name': 'multi_news',
                'config': None,
                'split': 'train',
                'loader_function': self.load_multi_news
            },
            # Wiki-CS (Wikipedia Sections Similarity)
            # {
            #     'name': 'wiki_snippets',
            #     'config': 'wiki40b_en_100_0',
            #     'split': 'train',
            #     'loader_function': self.load_wiki_snippets
            # },
            
            # TriviaQA (Query-Document)
            {
                'name': 'trivia_qa',
                'config': 'rc.wikipedia',
                'split': 'train',
                'loader_function': self.load_trivia_qa
            },
            # HotpotQA (Multi-hop Query-Document)
            {
                'name': 'hotpot_qa',
                'config': 'distractor',
                'split': 'train',
                'loader_function': self.load_hotpot_qa
            },
            # XNLI (Cross-lingual Natural Language Inference)
            {
                'name': 'xnli',
                'config': 'all_languages',
                'split': 'train',
                'loader_function': self.load_xnli
            },
            # CodeSearchNet (Code-Documentation Similarity)
            {
                'name': 'code_search_net',
                'config': 'python',
                'split': 'train',
                'loader_function': self.load_code_search_net
            },
            # S2ORC (Scientific Paper Similarity)
            {
                'name': 'sentence-transformers/s2orc',
                'config': 'title-abstract-pair',
                'split': 'train',
                'loader_function': self.load_s2orc
            },
            # CNN/DailyMail (News Article Similarity)
            {
                'name': 'cnn_dailymail',
                'config': '3.0.0',
                'split': 'train',
                'loader_function': self.load_cnn_dailymail
            },
            # Twitter Paraphrase Corpus
            # {
            #     'name': 'twitter_pairs',
            #     'config': None,
            #     'split': 'train',
            #     'loader_function': self.load_twitter_pairs
            # },
            # Amazon Product Reviews
            # {
            #     'name': 'amazon_reviews_multi',
            #     'config': 'en',
            #     'split': 'train',
            #     'loader_function': self.load_amazon_reviews
            # },
            # Legal-BERT Dataset
            # {
            #     'name': 'legal_bert',
            #     'config': None,
            #     'split': 'train',
            #     'loader_function': self.load_legal_bert
            # },
            # WikiHow (Step Similarity)
            {
                'name': 'wikihow',
                'config': 'all',
                'split': 'train',
                'loader_function': self.load_wikihow
            },
            # Stack Exchange (Question Similarity)
            {
                'name': 'stack_exchange',
                'config': 'paired',
                'split': 'train',
                'loader_function': self.load_stack_exchange
            },
            # Common Crawl News (Article Similarity)
            {
                'name': 'cc_news',
                'config': None,
                'split': 'train',
                'loader_function': self.load_cc_news
            },
            # ArXiv Dataset (Academic Paper Similarity)
            # {
            #     'name': 'arxiv_dataset',
            #     'config': None,
            #     'split': 'train',
            #     'loader_function': self.load_arxiv
            # },
            # Reddit TIFU (Story Similarity)
            {
                'name': 'reddit_tifu',
                'config': 'long',
                'split': 'train',
                'loader_function': self.load_reddit_tifu
            },
            
            # XSum Dataset
            {
                'name': 'xsum',
                'config': None,
                'split': 'train',
                'loader_function': self.load_xsum
            },
            # Gigaword Dataset
            {
                'name': 'gigaword',
                'config': 'english',
                'split': 'train',
                'loader_function': self.load_gigaword
            },
            # BigPatent Dataset
            {
                'name': 'big_patent',
                'config': 'all',
                'split': 'train',
                'loader_function': self.load_big_patent
            },
            # AESLC (Enron Email Subject Line Corpus)
            {
                'name': 'aeslc',
                'config': None,
                'split': 'train',
                'loader_function': self.load_aeslc
            },
            # BillSum Dataset
            {
                'name': 'billsum',
                'config': 'us',
                'split': 'train',
                'loader_function': self.load_billsum
            },
            # Newsroom Dataset
            {
                'name': 'newsroom',
                'config': None,
                'split': 'train',
                'loader_function': self.load_newsroom
            },
            # Reddit TIFU (Short version)
            {
                'name': 'reddit_tifu',
                'config': 'short',
                'split': 'train',
                'loader_function': self.load_reddit_tifu_short
            },
            # MultiXScience (Scientific Paper Summarization)
            {
                'name': 'multi_x_science_sum',
                'config': None,
                'split': 'train',
                'loader_function': self.load_multixscience
            },
            # SAMSum Dataset (Dialogue Summarization)
            {
                'name': 'samsum',
                'config': None,
                'split': 'train',
                'loader_function': self.load_samsum
            },
            # Opinosis Dataset
            {
                'name': 'opinosis',
                'config': None,
                'split': 'train',
                'loader_function': self.load_opinosis
            },
            # WikiSum Dataset
            {
                'name': 'wikimedia/wikisum',
                'config': None,
                'split': 'train',
                'loader_function': self.load_wikisum
            }
        ]  
  
        for config in dataset_configs:  
            try:
                dataset = load_dataset(config['name'], config['config'], split=config['split'], trust_remote_code=True) 
                if max_samples_per_dataset:  
                    dataset = dataset.shuffle(seed=42).select(range(min(len(dataset), max_samples_per_dataset)))  
                samples = config['loader_function'](dataset)  
                self.samples.extend(samples) 
            except Exception as e:
                print(f"Error loading dataset {config['name']}: {e}")
                traceback.print_exc()
  
        # Shuffle the combined samples  
        random.shuffle(self.samples)  
        
    # In the CombinedSimilarityDataset class

    def load_xsum(self, dataset):
        """
        Loads the XSum dataset for summarization pairs.
        """
        samples = []
        for example in dataset:
            document = example.get('document')
            summary = example.get('summary')
            if document and summary:
                samples.append((document, summary))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if self.max_samples_per_dataset and len(samples) >= self.max_samples_per_dataset:
                break
        return samples

    def load_gigaword(self, dataset):
        """
        Loads the Gigaword dataset for headline-generation pairs.
        """
        samples = []
        for example in dataset:
            document = example.get('document')
            summary = example.get('summary')
            if document and summary:
                samples.append((document, summary))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if self.max_samples_per_dataset and len(samples) >= self.max_samples_per_dataset:
                break
        return samples

    def load_big_patent(self, dataset):
        """
        Loads the BigPatent dataset for patent-summary pairs.
        """
        samples = []
        for example in dataset:
            description = example.get('description')
            abstract = example.get('abstract')
            if description and abstract:
                samples.append((description, abstract))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if self.max_samples_per_dataset and len(samples) >= self.max_samples_per_dataset:
                break
        return samples

    def load_aeslc(self, dataset):
        """
        Loads the AESLC dataset for email-summary pairs.
        """
        samples = []
        for example in dataset:
            email_body = example.get('email_body')
            subject_line = example.get('subject_line')
            if email_body and subject_line:
                samples.append((email_body, subject_line))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if self.max_samples_per_dataset and len(samples) >= self.max_samples_per_dataset:
                break
        return samples

    def load_billsum(self, dataset):
        """
        Loads the BillSum dataset for bill-summary pairs.
        """
        samples = []
        for example in dataset:
            text = example.get('text')
            summary = example.get('summary')
            if text and summary:
                samples.append((text, summary))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if self.max_samples_per_dataset and len(samples) >= self.max_samples_per_dataset:
                break
        return samples

    def load_newsroom(self, dataset):
        """
        Loads the Newsroom dataset for article-summary pairs.
        """
        samples = []
        for example in dataset:
            text = example.get('text')
            summary = example.get('summary')
            if text and summary:
                samples.append((text, summary))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if self.max_samples_per_dataset and len(samples) >= self.max_samples_per_dataset:
                break
        return samples

    def load_reddit_tifu_short(self, dataset):
        """
        Loads the Reddit TIFU (short) dataset for summarization pairs.
        """
        samples = []
        for example in dataset:
            document = example.get('documents')
            tldr = example.get('tldr')
            if document and tldr:
                samples.append((document, tldr))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if self.max_samples_per_dataset and len(samples) >= self.max_samples_per_dataset:
                break
        return samples

    def load_multixscience(self, dataset):
        """
        Loads the MultiXScience dataset for scientific paper summarization.
        """
        samples = []
        for example in dataset:
            source = example.get('related_work')
            target = example.get('abstract')
            if source and target:
                samples.append((source, target))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if self.max_samples_per_dataset and len(samples) >= self.max_samples_per_dataset:
                break
        return samples

    def load_samsum(self, dataset):
        """
        Loads the SAMSum dataset for dialogue summarization.
        """
        samples = []
        for example in dataset:
            dialogue = example.get('dialogue')
            summary = example.get('summary')
            if dialogue and summary:
                samples.append((dialogue, summary))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if self.max_samples_per_dataset and len(samples) >= self.max_samples_per_dataset:
                break
        return samples

    def load_opinosis(self, dataset):
        """
        Loads the Opinosis dataset for opinion summarization.
        """
        samples = []
        for example in dataset:
            review = example.get('review')
            summary = example.get('summary')
            if review and summary:
                samples.append((review, summary))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if self.max_samples_per_dataset and len(samples) >= self.max_samples_per_dataset:
                break
        return samples

    def load_wikisum(self, dataset):
        """
        Loads the WikiSum dataset for Wikipedia article summarization.
        """
        samples = []
        for example in dataset:
            source_text = example.get('source_text')
            target_text = example.get('target_text')
            if source_text and target_text:
                samples.append((source_text, target_text))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if self.max_samples_per_dataset and len(samples) >= self.max_samples_per_dataset:
                break
        return samples

    def load_wikihow_summary(self, dataset):
        """
        Loads the WikiHow dataset for article-summary pairs.
        """
        samples = []
        for example in dataset:
            text = example.get('text')
            headline = example.get('headline')
            if text and headline:
                samples.append((text, headline))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if self.max_samples_per_dataset and len(samples) >= self.max_samples_per_dataset:
                break
        return samples
  
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
            assert isinstance(samples[-1][0], str)
            assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
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
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
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
            if score >= 3.0:  # STS-B scores range from 0 to 5  
                sentence1 = example['sentence1']  
                sentence2 = example['sentence2']  
                samples.append((sentence1, sentence2))  
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
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
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
        return samples  
    
    def load_trivia_qa(self, dataset):
        """
        Loads TriviaQA dataset for complex query-document pairs.
        """
        samples = []
        for example in dataset:
            question = example['question']
            # Handle case where evidence field may have different structure
            if 'evidence' in example:
                for evidence in example['evidence']:
                    # Check if evidence is a dict with 'text' field
                    if isinstance(evidence, dict) and 'text' in evidence:
                        if evidence['text'].strip():
                            samples.append((question, evidence['text']))
                            assert isinstance(samples[-1][0], str)
                            assert isinstance(samples[-1][1], str)
                    # If evidence is directly a string
                    elif isinstance(evidence, str) and evidence.strip():
                        samples.append((question, evidence))
                        assert isinstance(samples[-1][0], str)
                        assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
        return samples

    def load_hotpot_qa(self, dataset):
        """
        Loads HotpotQA dataset for multi-hop reasoning pairs.
        """
        samples = []
        for example in dataset:
            question = example['question']
            context = ' '.join([p[0] for p in example['context'] if p[0].strip()])
            if context:
                samples.append((question, context))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
        return samples

    def load_xnli(self, dataset):
        """
        Loads XNLI dataset for cross-lingual similarity.
        """
        samples = []
        for example in dataset:
            if example['label'] == 0:  # entailment
                samples.append((example['premise'], example['hypothesis']))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
        return samples

    def load_code_search_net(self, dataset):
        """
        Loads CodeSearchNet for code-documentation similarity.
        """
        samples = []
        for example in dataset:
            if example['func_documentation_string'] and example['func_code_string']:
                samples.append((example['func_documentation_string'], example['func_code_string']))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
        return samples

    def load_s2orc(self, dataset):
        """
        Loads S2ORC dataset for scientific paper similarity.
        """
        samples = []
        for example in dataset:
            if example['abstract'] and example['title']:
                # Create pairs from abstract and relevant sections
                abstract = example['abstract']
                sections = example['title'].split('\n\n')
                for section in sections:
                    if len(section) > 30:  # Filter short sections
                        samples.append((abstract, section))
                        assert isinstance(samples[-1][0], str)
                        assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
        return samples

    def load_cnn_dailymail(self, dataset):
        """
        Loads CNN/DailyMail dataset for news article similarity.
        """
        samples = []
        for example in dataset:
            article_parts = example['article'].split('\n')
            for i in range(len(article_parts)-1):
                if len(article_parts[i]) > 50 and len(article_parts[i+1]) > 50:
                    samples.append((article_parts[i], article_parts[i+1]))
                    assert isinstance(samples[-1][0], str)
                    assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
        return samples

    def load_twitter_pairs(self, dataset):
        """
        Loads Twitter paraphrase pairs.
        """
        samples = []
        for example in dataset:
            if example['is_paraphrase']:
                samples.append((example['tweet1'], example['tweet2']))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
        return samples

    def load_amazon_reviews(self, dataset):
        """
        Loads Amazon product reviews for text similarity.
        """
        samples = []
        for example in dataset:
            if example['stars'] >= 4:  # Use highly rated products
                # Create pairs from review title and body
                samples.append((example['review_title'], example['review_body']))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
        return samples

    def load_legal_bert(self, dataset):
        """
        Loads legal document pairs for similarity.
        """
        samples = []
        for example in dataset:
            if example['similarity_score'] > 0.7:
                samples.append((example['document1'], example['document2']))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
        return samples

    def load_wikihow(self, dataset):
        """
        Loads WikiHow steps for procedural similarity.
        """
        samples = []
        for example in dataset:
            steps = example['text'].split('\n')
            for i in range(len(steps)-1):
                if len(steps[i]) > 30 and len(steps[i+1]) > 30:
                    samples.append((steps[i], steps[i+1]))
                    assert isinstance(samples[-1][0], str)
                    assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
        return samples

    def load_stack_exchange(self, dataset):
        """
        Loads Stack Exchange similar questions.
        """
        samples = []
        for example in dataset:
            if example['score'] > 0:  # Use positively scored pairs
                samples.append((example['question1'], example['question2']))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
        return samples

    def load_cc_news(self, dataset):
        """
        Loads Common Crawl news articles.
        """
        samples = []
        for example in dataset:
            if example['text'] and example['title']:
                # Create pairs from title and first paragraph
                paragraphs = example['text'].split('\n\n')
                if paragraphs:
                    samples.append((example['title'], paragraphs[0]))
                    assert isinstance(samples[-1][0], str)
                    assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
        return samples

    def load_arxiv(self, dataset):
        """
        Loads arXiv papers for academic text similarity.
        """
        samples = []
        for example in dataset:
            if example['abstract'] and example['title']:
                samples.append((example['title'], example['abstract']))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
        return samples

    def load_reddit_tifu(self, dataset):
        """
        Loads Reddit TIFU posts for story similarity.
        """
        samples = []
        for example in dataset:
            if example['title'] and example['documents']:
                # Create pairs from title and post body
                samples.append((example['title'], example['documents']))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if example['tldr'] and example['documents']:
                # Create pairs from title and post body
                samples.append((example['tldr'], example['documents']))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
        return samples
    
    def load_natural_questions(self, dataset):
        """
        Loads Natural Questions dataset for query-passage pairs.
        """
        samples = []
        for example in dataset:
            try:
                if example['annotations']['yes_no_answer'][0] != 'NONE':
                    question = example['question']['text']
                    # Use the long answer as the positive passage
                    if example['annotations']['long_answer'][0]:
                        long_answer_loc = example['annotations']['long_answer'][0]
                        answer = example['document']['tokens']['token'][long_answer_loc['start_token']:
                                                long_answer_loc['end_token']]
                        answer = " ".join(answer)
                        samples.append((question, answer))
                        assert isinstance(samples[-1][0], str)
                        assert isinstance(samples[-1][1], str)
            except KeyError:
                # Skip examples with missing fields
                continue
            if len(samples) > self.max_samples_per_dataset:
                break
        return samples

    def load_mrpc(self, dataset):
        """
        Loads Microsoft Research Paraphrase Corpus.
        """
        samples = []
        for example in dataset:
            if example['label'] == 1:  # Paraphrase pairs
                samples.append((example['sentence1'], example['sentence2']))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
        return samples

    def load_paws(self, dataset):
        """
        Loads PAWS dataset for challenging paraphrase detection.
        """
        samples = []
        for example in dataset:
            if example['label'] == 1:  # Paraphrase pairs
                samples.append((example['sentence1'], example['sentence2']))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
        return samples

    def load_multi_news(self, dataset):
        """
        Loads MultiNews dataset for document similarity.
        """
        samples = []
        for example in dataset:
            # Split document into paragraphs and create pairs of related paragraphs
            documents = example['document'].split('\n\n')
            for i in range(len(documents)-1):
                if len(documents[i]) > 50 and len(documents[i+1]) > 50:  # Filter short paragraphs
                    samples.append((documents[i], documents[i+1]))
                    assert isinstance(samples[-1][0], str)
                    assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
        return samples

    def load_wiki_snippets(self, dataset):
        """
        Loads Wikipedia sections dataset for paragraph similarity.
        """
        samples = []
        for example in dataset:
            if 'section_text' in example and 'context' in example:
                # Section text and its context are semantically related
                samples.append((example['section_text'], example['context']))
                assert isinstance(samples[-1][0], str)
                assert isinstance(samples[-1][1], str)
            if len(samples) > self.max_samples_per_dataset:
                break
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
        assert isinstance(text, str)

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

