"""Load BRIGHT dataset from HuggingFace and extract subsets."""

import os
import sys
from typing import Dict, List, Optional, Union
from pathlib import Path
from datasets import load_dataset, DatasetDict
import pandas as pd
import yaml

# Simplified import handling
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import get_data_base_dir, load_config


class BRIGHTLoader:
    """Loader for BRIGHT dataset from HuggingFace."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
            """Initialize BRIGHT loader using unified config."""
            self.config = load_config(config_path)
            
            # 1. FIXED PATHS: Navigating the new nested YAML structure
            bright_cfg = self.config['data']['bright']
            self.dataset_name = bright_cfg['name']
            self.examples_config = bright_cfg.get('examples_config', 'Gemini-1.0_reason')
            
            # 2. RESOLVED PATHS: Using get_data_base_dir for scratch/local compatibility
            base_dir = get_data_base_dir()
            # Paths from config are relative to the scratch base
            self.cache_dir = Path(base_dir) / bright_cfg.get('cache_dir', 'data/bright')
            
            self.documents_dataset = None
            self.examples_dataset = None
            
            # 3. FIXED PATHS: ReasonIR caching block
            if 'reasonir' in self.config.get('data', {}):
                reasonir_cfg = self.config['data']['reasonir']
                try:
                    self.cache_reasonir_hq_dataset(
                        dataset_name=reasonir_cfg.get('name', 'reasonir/reasonir-data'),
                        subset=reasonir_cfg.get('subset', 'hq'),
                        cache_dir=str(self.cache_dir)
                    )
                except Exception as e:
                    print(f"⚠️ Warning: Could not cache ReasonIR-HQ dataset: {e}")
    
    def load_dataset(self, cache_dir: Optional[str] = None) -> Dict[str, DatasetDict]:
        """
        Load BRIGHT dataset from HuggingFace.
        
        Returns:
            Dictionary with 'documents' and 'examples' DatasetDict objects
        """
        cache = cache_dir or self.cache_dir
        os.makedirs(cache, exist_ok=True)
        
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
        
        print(f"DEBUG: Creating ID mapping for {len(self.documents_dataset)} domains...", flush=True)
        id2doc = {}
        
        try:
            for task_name, domain_data in self.documents_dataset.items():
                print(f"DEBUG: Processing domain '{task_name}' with {len(domain_data)} documents...", flush=True)
                domain_count_before = len(id2doc)
                for i in range(len(domain_data)):
                    try:
                        doc_id = str(domain_data[i]['id'])
                        doc_content = domain_data[i]['content']
                        id2doc[doc_id] = doc_content
                    except KeyError as e:
                        print(f"ERROR: Missing key '{e}' in domain '{task_name}', document {i}", flush=True)
                        print(f"DEBUG: Available keys: {domain_data[i].keys() if i < len(domain_data) else 'N/A'}", flush=True)
                        raise
                    except Exception as e:
                        print(f"ERROR: Failed to process domain '{task_name}', document {i}: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                        raise
                
                domain_count_after = len(id2doc)
                documents_added = domain_count_after - domain_count_before
                print(f"DEBUG: Completed domain '{task_name}': {documents_added} documents added (total: {domain_count_after})", flush=True)
            
            print(f"✅ Created ID-to-text mapping for {len(id2doc)} documents across {len(self.documents_dataset)} domains", flush=True)
        except Exception as e:
            print(f"FATAL ERROR in get_all_documents_id_map: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        
        return id2doc
    
    def cache_reasonir_hq_dataset(self, dataset_name: str = "reasonir/reasonir-data",
                                  subset: str = "hq",
                                  cache_dir: Optional[str] = None) -> None:
        """
        Cache ReasonIR-HQ dataset (downloads if not already cached).
        Automatically called during initialization.
        
        IMPORTANT: This must be run ONLINE (without HF_HUB_OFFLINE=1) to download the dataset.
        If offline, it will skip caching (dataset should already be cached).
        
        Args:
            dataset_name: ReasonIR dataset name (default: "reasonir/reasonir-data")
            subset: Dataset subset to use (default: "hq")
            cache_dir: Optional cache directory for HuggingFace datasets (if None, uses self.cache_dir)
        """
        cache = cache_dir or self.cache_dir
        os.makedirs(cache, exist_ok=True)
        
        # Check if dataset is already cached (skip if offline mode)
        if os.environ.get('HF_HUB_OFFLINE') == '1' or os.environ.get('TRANSFORMERS_OFFLINE') == '1':
            print(f"⚠️ Offline mode detected - skipping ReasonIR-HQ caching (assuming already cached)")
            return
        
        print(f"Caching ReasonIR dataset: {dataset_name} (subset: {subset})...")
        print(f"Using cache directory: {cache}")
        try:
            load_dataset(dataset_name, subset, cache_dir=cache)
            print(f"✅ ReasonIR-HQ dataset cached successfully!")
        except Exception as e:
            # If caching fails, it might be because dataset is already cached or offline
            print(f"⚠️ Could not cache ReasonIR-HQ dataset: {e}")
            print("   This is OK if dataset is already cached or you're in offline mode.")

if __name__ == "__main__":
    loader = BRIGHTLoader()
    loader.load_dataset()
    
    test_domain = 'biology'
    try:
        data = loader.get_data_split(test_domain)
        print(f"\n--- {test_domain.upper()} SANITY CHECK ---")
        print(f"Corpus size:  {len(data['corpus']):,}") # Adds commas for readability
        print(f"Queries size: {len(data['queries'])}")
        print(f"Qrels size:   {len(data['qrels'])}")
        print(f"Sample Query: {data['queries'].iloc[0]['query'][:100]}...") # Truncates long queries
    except Exception as e:
        print(f"\nError loading {test_domain}: {e}")