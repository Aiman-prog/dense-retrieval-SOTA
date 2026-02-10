"""Patch Tevatron to remove Qwen/multimodal dependencies.

Tevatron commit 8f31cd8 added Qwen multimodal support that requires
packages (qwen_omni_utils, Qwen2_5Omni models) not available in our
PyTorch 2.1 container. This script removes all multimodal references
so Tevatron works for text-only dense retrieval.

Usage: python scripts/patch_tevatron.py [tevatron_base_path]
"""

import sys
import os
import re
import glob

def patch_tevatron(base_path):
    if not os.path.isdir(base_path):
        print(f"ERROR: Tevatron not found at {base_path}")
        sys.exit(1)

    patched = 0

    for pyfile in glob.glob(os.path.join(base_path, "**", "*.py"), recursive=True):
        with open(pyfile, 'r') as f:
            original = f.read()

        content = original

        # 1. Comment out qwen_omni_utils imports (e.g. collator.py)
        content = re.sub(
            r'^(.*qwen_omni_utils.*)',
            r'# \1',
            content, flags=re.MULTILINE
        )

        # 2. Comment out Qwen2_5Omni references (e.g. dense.py imports + assignments)
        content = re.sub(
            r'^(.*Qwen2_5Omni.*)',
            r'# \1',
            content, flags=re.MULTILINE
        )

        # 3. Remove entire MultiModalDenseModel class definition + body
        content = re.sub(
            r'^class MultiModalDenseModel.*?(?=\nclass |\n[^\s#]|\Z)',
            '',
            content, flags=re.DOTALL | re.MULTILINE
        )

        # 4. Remove visual encoder freezing block (for loop + indented body)
        content = re.sub(
            r'^\s*for param in self\.encoder\.visual\.parameters\(\):.*?\n(?:\s+.*\n)*',
            '',
            content, flags=re.MULTILINE
        )

        # 5. Fix __init__.py: remove MultiModalDenseModel from imports
        content = content.replace(
            'from .dense import DenseModel, MultiModalDenseModel',
            'from .dense import DenseModel'
        )

        if content != original:
            with open(pyfile, 'w') as f:
                f.write(content)
            patched += 1
            print(f"  Patched: {os.path.relpath(pyfile, base_path)}")

    print(f"  Total: {patched} file(s) patched")
    return patched

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/.local/lib/python3.10/site-packages/tevatron"
    )
    print(f"Patching Tevatron at: {base}")
    patch_tevatron(base)
