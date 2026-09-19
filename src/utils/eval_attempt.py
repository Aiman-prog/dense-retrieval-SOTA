"""Small, stdlib-only boundaries for isolated evaluation attempts."""
import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path


def configure_cache():
    """Call before importing datasets; children inherit this invocation's cache."""
    if not os.environ.get('DENSE_EVAL_ATTEMPT_ID'):
        os.environ['DENSE_EVAL_ATTEMPT_ID'] = uuid.uuid4().hex
        root = os.environ.get('EVAL_CACHE_ROOT')
        if root:
            Path(root).mkdir(parents=True, exist_ok=True)
        os.environ['HF_DATASETS_CACHE'] = tempfile.mkdtemp(prefix='bright-', dir=root)
    if not os.environ.get('HF_DATASETS_CACHE'):
        raise ValueError('An evaluation attempt must have a private HF_DATASETS_CACHE')
    return os.environ['DENSE_EVAL_ATTEMPT_ID']


def model_identity(model_path):
    """Hash weights and all tokenizer/config assets, including sharded/LoRA models."""
    root = Path(model_path)
    files = [p for p in root.iterdir() if p.is_file() and (
        p.suffix in ('.safetensors', '.json', '.model', '.txt') or
        p.name.startswith(('pytorch_model', 'adapter_model', 'merges', 'vocab')))]
    # Training logs/manifests do not determine the encoded vectors.
    files = [p for p in files if p.name not in (
        'run_manifest.json', 'ance_trainer_summary.json', 'trainer_state.json')]
    if not any(p.suffix == '.safetensors' or p.name.startswith(
            ('pytorch_model', 'adapter_model')) for p in files):
        raise FileNotFoundError(f'No model weights in {root}')
    result = {}
    for path in sorted(files):
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        result[path.name] = digest.hexdigest()
    return result


def identity_digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
