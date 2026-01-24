import os
import sys
import subprocess
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import load_config

def main():
    config = load_config()
    domains = config['evaluation'].get('eval_domains', [])
    
    # Path to your existing evaluation script
    eval_script = project_root / 'scripts' / 'evaluate.py'
    
    # We need to know which model to evaluate
    # Defaulting to the crossbatch output directory
    model_path = Path(os.environ.get('DATA_BASE_DIR', '/scratch/' + os.getlogin() + '/dense-retrieval-SOTA')) / 'models' / config['training']['crossbatch']['model_name']

    print(f"🕵️  Starting Evaluation for {len(domains)} domains...")
    print(f"🏗️  Model Path: {model_path}\n")

    for domain in domains:
        print(f"--- 🌐 Evaluating Domain: {domain} ---")
        cmd = [
            sys.executable, str(eval_script),
            "--model_path", str(model_path),
            "--domain", domain,
            "--k", str(config['evaluation'].get('top_k', 1000)),
            "--batch_size", "128"
        ]
        
        # We run this and wait for it to finish before moving to the next
        subprocess.run(cmd, check=True)
        print(f"✅ Finished {domain}\n")

    print("🏁 All evaluations complete.")

if __name__ == "__main__":
    main()