import os
import sys
import subprocess
import argparse
from pathlib import Path

# Resolve project root and add to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

# Import your helpers and classes
from utils.helpers import load_config, get_data_base_dir, get_path
from data.preprocessor import BRIGHTPreprocessor
from data.bright_loader import BRIGHTLoader

def check_and_prepare_data(domains, config):
    """Checks if processed data exists; if not, triggers preprocessor."""
    processed_dir = get_path("processed")
    loader = None 
    preprocessor = BRIGHTPreprocessor()
    
    for domain in domains:
        required_files = [
            processed_dir / f"{domain}_corpus.jsonl",
            processed_dir / f"{domain}_queries.jsonl",
            processed_dir / f"{domain}_qrels.txt"
        ]
        
        if not all(f.exists() for f in required_files):
            print(f"📦 Data for '{domain}' missing in {processed_dir}. Processing...")
            if loader is None:
                loader = BRIGHTLoader()
                loader.load_dataset()
            
            domain_data = loader.get_data_split(domain)
            preprocessor.prepare_tevatron_corpus(domain_data['corpus'], f"{domain}_corpus.jsonl")
            preprocessor.prepare_tevatron_queries(domain_data['queries'], f"{domain}_queries.jsonl")
            preprocessor.prepare_trec_qrels(domain_data['qrels'], f"{domain}_qrels.txt")
        else:
            print(f"✅ Data for '{domain}' verified.")

def main():
    config = load_config()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to model")
    args = parser.parse_args()

    # 1. Resolve Model Path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        base_dir = Path(get_data_base_dir())
        model_name = config['training']['crossbatch']['model_name']
        model_path = base_dir / 'models' / model_name

    # 2. Resolve Eval Script Path (The Fix)
    eval_script = Path(__file__).parent.parent / "src" / "evaluation" / "evaluate.py"

    print(f"🕵️  Starting Evaluation Runner")
    print(f"🏗️  Model: {model_path}")
    print(f"📄 Script: {eval_script}\n")

    # Final Safety Checks
    if not model_path.exists():
        print(f"❌ ERROR: Model path does not exist: {model_path}")
        sys.exit(1)
    if not eval_script.exists():
        print(f"❌ ERROR: Evaluation script not found at {eval_script}")
        sys.exit(1)

    domains = config['evaluation'].get('eval_domains', [])
    
    # Check/Prepare data before loop
    check_and_prepare_data(domains, config)

    for domain in domains:
        print(f"\n--- 🌐 Evaluating Domain: {domain} ---")
        cmd = [
            sys.executable, str(eval_script),
            "--model_path", str(model_path),
            "--domain", domain,
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"❌ Domain {domain} failed. Moving to next...")
            continue

    print("\n🏁 All evaluations complete.")

if __name__ == "__main__":
    main()