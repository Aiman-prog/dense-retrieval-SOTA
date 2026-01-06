"""Load BRIGHT dataset from HuggingFace and extract subsets."""

import os
from typing import Dict, List, Optional, Union
from pathlib import Path
from datasets import load_dataset, DatasetDict
import pandas as pd
import yaml

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
        self.cache_dir = self.config['dataset'].get('cache_dir', 'data/bright')
        self.documents_dataset = None
        self.examples_dataset = None
    
    def load_dataset(self, cache_dir: Optional[str] = None) -> Dict[str, DatasetDict]:
        """
        Load BRIGHT dataset from HuggingFace.
        """
        cache = cache_dir or self.cache_dir
        os.makedirs(cache, exist_ok=True)
        
        print(f"Loading BRIGHT 'documents' from: {self.dataset_name}")
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