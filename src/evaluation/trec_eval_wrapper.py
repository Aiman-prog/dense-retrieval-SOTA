"""Wrapper for TREC eval using pytrec_eval (Python bindings)."""

from typing import Dict, List, Set, Optional, Union
import pandas as pd
import pytrec_eval

class TrecEvalWrapper:
    """
    Wrapper for pytrec_eval to compute standard IR metrics directly in Python.
    Optimized for in-memory evaluation (no temporary files).
    """
    
    def __init__(self, qrels_source: Union[pd.DataFrame, Dict[str, Set[str]]]):
        """
        Initialize with ground truth data (qrels).
        
        Args:
            qrels_source: Either a DataFrame ['query_id', 'doc_id', 'relevance']
                          OR a Dict {query_id: {doc_id1, doc_id2}}
        """
        self.qrels = self._parse_qrels(qrels_source)
    
    def _parse_qrels(self, qrels_source) -> Dict[str, Dict[str, int]]:
        """Internal helper to standardize qrels into pytrec_eval format."""
        qrels_pytrec = {}
        
        if isinstance(qrels_source, pd.DataFrame):
            for _, row in qrels_source.iterrows():
                qid = str(row['query_id'])
                doc_id = str(row['doc_id'])
                # Default relevance to 1 if column missing
                rel = int(row.get('relevance', 1)) 
                
                if qid not in qrels_pytrec:
                    qrels_pytrec[qid] = {}
                qrels_pytrec[qid][doc_id] = rel
                
        elif isinstance(qrels_source, dict):
            for qid, doc_ids in qrels_source.items():
                qrels_pytrec[str(qid)] = {str(doc_id): 1 for doc_id in doc_ids}
                
        return qrels_pytrec

    def evaluate(self, 
                 run_results: Dict[str, Dict[str, float]], 
                 metrics={"ndcg_cut_10", "recall_10", "recip_rank"}) -> Dict[str, float]:
        """
        Calculate metrics for the given run results.
        
        Args:
            run_results: Dictionary {query_id: {doc_id: score}} 
                         (Output from format_faiss_results)
            metrics: Set of metrics to calculate.
            
        Returns:
            Dictionary with average scores (e.g., {'ndcg_cut_10': 0.45})
        """
        if not run_results:
            return {m: 0.0 for m in metrics}

        # Initialize Evaluator
        evaluator = pytrec_eval.RelevanceEvaluator(self.qrels, metrics)

        # Compute metrics per query
        results_per_query = evaluator.evaluate(run_results)

        # Aggregate (Average) results
        final_metrics = {}
        for metric in metrics:
            values = [query_scores[metric] for query_scores in results_per_query.values()]
            final_metrics[metric] = sum(values) / len(values) if values else 0.0

        return final_metrics

    @staticmethod
    def format_faiss_results(rankings: Dict[str, List[str]], k: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Helper to convert simple FAISS ranking lists to pytrec_eval format.
        
        Args:
            rankings: Dictionary {query_id: [doc_id1, doc_id2, ...]}
            k: Cutoff rank
        """
        formatted_run = {}
        for qid, doc_ids in rankings.items():
            formatted_run[str(qid)] = {}
            # Take top-k
            top_docs = doc_ids[:k]
            for rank, doc_id in enumerate(top_docs, start=1):
                # Create a dummy score based on rank (Rank 1 = Score 10...)
                score = (k + 1) - rank 
                formatted_run[str(qid)][str(doc_id)] = float(score)
                
        return formatted_run