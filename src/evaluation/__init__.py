"""Evaluation modules for retrieval metrics using trec_eval/pyserini."""

from .trec_eval_wrapper import TrecEvalWrapper
from .retriever import (
    encode_corpus,
    encode_queries,
    build_faiss_index,
    retrieve_top_k,
    evaluate_model
)

__all__ = [
    'TrecEvalWrapper',
    'encode_corpus',
    'encode_queries',
    'build_faiss_index',
    'retrieve_top_k',
    'evaluate_model',
]
