"""Load BRIGHT dataset from HuggingFace and extract subsets."""

import os
import sys
from typing import Dict, List, Optional, Union
from pathlib import Path
from datasets import load_dataset, DatasetDict
import pandas as pd
import yaml
# Import helper function (handle both package and direct import)
try:
    from utils.helpers import get_data_base_dir
except ImportError:
    # Fallback for relative import
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root / 'src'))
    from utils.helpers import get_data_base_dir

class BRIGHTLoader:
    """Loader for BRIGHT dataset from HuggingFace."""
    
    def __init__(self, config_path: str):
        """Initialize BRIGHT loader."""
        project_root = Path(__file__).parent.parent.parent
        config_file = project_root / config_path
        
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_name = self.config['dataset']['name']
        self.examples_config = self.config['dataset'].get('examples_config', 'Gemini-1.0_reason')
        # Always use DelftBlue structure: /scratch/${USER}/dense-retrieval-SOTA/data/bright
        # Can override with BRIGHT_CACHE_DIR env var if needed
        base_dir = get_data_base_dir()
        self.cache_dir = os.environ.get('BRIGHT_CACHE_DIR') or f'{base_dir}/data/bright'
        self.documents_dataset = None
        self.examples_dataset = None
    
    def load_dataset(self, cache_dir: Optional[str] = None) -> Dict[str, DatasetDict]:
        """
        Load BRIGHT dataset from HuggingFace.
        
        Returns:
            Dictionary with 'documents' and 'examples' DatasetDict objects
        """
        cache = cache_dir or self.cache_dir
        os.makedirs(cache, exist_ok=True)
        
        # CRITICAL: Set HuggingFace cache environment variables to ensure
        # datasets are downloaded to the specified cache_dir, not ~/.cache/huggingface/
        # This is needed because load_dataset() respects HF_DATASETS_CACHE env var
        original_hf_cache = os.environ.get('HF_DATASETS_CACHE')
        original_hf_home = os.environ.get('HF_HOME')
        
        # Set to our custom cache directory
        os.environ['HF_DATASETS_CACHE'] = cache
        os.environ['HF_HOME'] = cache
        
        try:
            print(f"Loading BRIGHT 'documents' from: {self.dataset_name}")
            print(f"Using cache directory: {cache}")
            # 'documents' subset contains the corpus for all domains
            self.documents_dataset = load_dataset(
                self.dataset_name,
                'documents',
                cache_dir=cache
            )
            
            print(f"Loading BRIGHT '{self.examples_config}' (queries/qrels) from: {self.dataset_name}")
            # 'examples' (or reasoning subsets) contains queries and gold_ids
            self.examples_dataset = load_dataset(
                self.dataset_name,
                self.examples_config,
                cache_dir=cache
            )
        finally:
            # Restore original environment variables if they existed
            if original_hf_cache is not None:
                os.environ['HF_DATASETS_CACHE'] = original_hf_cache
            elif 'HF_DATASETS_CACHE' in os.environ:
                del os.environ['HF_DATASETS_CACHE']
                
            if original_hf_home is not None:
                os.environ['HF_HOME'] = original_hf_home
            elif 'HF_HOME' in os.environ:
                del os.environ['HF_HOME']
        
        # Verify available domains overlap
        doc_domains = set(self.documents_dataset.keys())
        ex_domains = set(self.examples_dataset.keys())
        print(f"Loaded Documents Domains: {list(doc_domains)}")
        print(f"Loaded Examples Domains: {list(ex_domains)}")
        
        return {
            'documents': self.documents_dataset,
            'examples': self.examples_dataset
        }
    
    def get_corpus(self, domain: str) -> pd.DataFrame:
        """Extract corpus (textbook documents) for a domain."""
        if self.documents_dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        if domain not in self.documents_dataset:
            raise ValueError(f"Domain '{domain}' not found in documents.")
        
        domain_data = self.documents_dataset[domain]
        
        # BRIGHT documents use 'id' and 'content'
        return pd.DataFrame({
            'doc_id': domain_data['id'],
            'text': domain_data['content']
        })
    
    def get_queries(self, domain: str) -> pd.DataFrame:
        """Extract queries for a domain."""
        if self.examples_dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        if domain not in self.examples_dataset:
            raise ValueError(f"Domain '{domain}' not found in examples.")
        
        domain_data = self.examples_dataset[domain]
        
        # BRIGHT examples use 'id' and 'query'
        return pd.DataFrame({
            'query_id': domain_data['id'],
            'query': domain_data['query']
        })
    
    def get_qrels(self, domain: str) -> pd.DataFrame:
        """
        Extract qrels (gold_ids) for a domain.
        
        GOAL: Convert the raw HF dataset format into a standard TREC qrels format.
        
        Input (HF Dataset):
          Row 1: {id: "q1", gold_ids: ["docA", "docB"]}  <- List of strings
          Row 2: {id: "q2", gold_ids: "docC"}            <- Single string (Dangerous!)
          Row 3: {id: "q3", gold_ids: "docD,docE"}       <- Comma-separated string
          
        Output (Standard QRELS):
          q1  docA  1
          q1  docB  1
          q2  docC  1
          q3  docD  1
          q3  docE  1
        """
        if self.examples_dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        if domain not in self.examples_dataset:
            raise ValueError(f"Domain '{domain}' not found in examples.")
        
        domain_data = self.examples_dataset[domain]
        
        # BRIGHT examples use 'id' (query_id) and 'gold_ids' (list of relevant doc_ids)
        qrels_list = []
        
        # Iterate safely
        for i in range(len(domain_data)):
            qid = domain_data[i]['id']
            golds = domain_data[i]['gold_ids']
            
            # Normalize gold_ids to list
            if isinstance(golds, str):
                # Only split if it looks like a list string, otherwise treat as single ID
                # BRIGHT IDs can be strings, so be careful not to split separate IDs incorrectly
                if ',' in golds:
                     doc_ids = [d.strip() for d in golds.split(',')]
                else:
                     doc_ids = [golds]
            elif isinstance(golds, (list, tuple)):
                doc_ids = golds
            else:
                doc_ids = [str(golds)]
                
            for doc_id in doc_ids:
                qrels_list.append({
                    'query_id': str(qid),
                    'doc_id': str(doc_id),
                    'relevance': 1
                })
                
        return pd.DataFrame(qrels_list)

    def get_data_split(self, domain: str) -> Dict[str, pd.DataFrame]:
        """
        Get all data for a specific domain task.
        In BRIGHT, typically the 'examples' split IS the evaluation set.
        """
        return {
            'corpus': self.get_corpus(domain),
            'queries': self.get_queries(domain),
            'qrels': self.get_qrels(domain)
        }
    
    def get_all_documents_id_map(self) -> Dict[str, str]:
        """
        Create a mapping from document ID to document text for ALL domains.
        
        This is used to map ReasonIR-HQ document IDs to their corresponding texts.
        Based on the approach from ReasonIR dataset card:
        https://huggingface.co/datasets/reasonir/reasonir-data
        
        Returns:
            Dictionary mapping doc_id -> doc_text
        """
        if self.documents_dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        id2doc = {}
        
        # Iterate through all domains/tasks in BRIGHT documents
        for task in self.documents_dataset.keys():
            domain_data = self.documents_dataset[task]
            
            # Extract IDs and texts for this domain
            for i in range(len(domain_data)):
                doc_id = str(domain_data[i]['id'])
                doc_text = domain_data[i]['content']
                id2doc[doc_id] = doc_text
        
        print(f"Created ID-to-text mapping for {len(id2doc)} documents across {len(self.documents_dataset)} domains")
        return id2doc
    
    @staticmethod
    def cache_reasonir_hq_dataset(dataset_name: str = "reasonir/reasonir-data",
                                  subset: str = "hq",
                                  cache_dir: Optional[str] = None) -> None:
        """
        Cache ReasonIR-HQ dataset (downloads if not already cached).
        Minimal method to pre-download dataset for offline training.
        
        Args:
            dataset_name: ReasonIR dataset name (default: "reasonir/reasonir-data")
            subset: Dataset subset to use (default: "hq")
            cache_dir: Optional cache directory for HuggingFace datasets
        """
        print(f"Caching ReasonIR dataset: {dataset_name} (subset: {subset})...")
        load_dataset(dataset_name, subset, cache_dir=cache_dir)
        print(f"✅ ReasonIR-HQ dataset cached successfully!")

if __name__ == "__main__":
    loader = BRIGHTLoader(config_path='config/config.yaml')
    loader.load_dataset()
    
    # Example: 'biology' is one of the standard domains in BRIGHT
    try:
        data = loader.get_data_split('biology')
        print(f"\nSuccessfully loaded 'biology' domain:")
        print(f"- Corpus size: {len(data['corpus'])}")
        print(f"- Queries size: {len(data['queries'])}")
        print(f"- Qrels size: {len(data['qrels'])}")
        print(f"- Sample Query: {data['queries'].iloc[0]['query']}")
    except Exception as e:
        print(f"\nError loading domain: {e}")
