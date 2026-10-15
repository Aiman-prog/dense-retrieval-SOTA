"""Build the versioned training-data directory every MaxP arm reads. CPU only.

Copies the original (unsplit) mixture and its derived corpus/queries/qrels from
SOURCE into config `paths.training_data_dir`, then writes the manifest
`helpers.require_training_data` demands -- and only if the copy hashes identically to
SOURCE and no query id carries the chunk-row suffix. The destination must not exist:
nothing is overwritten, and SOURCE is never modified.

    python scripts/prepare_training_data.py --source $DATA_BASE_DIR/data/unchunked_backup
"""
import re
import sys
import json
import shutil
import argparse
import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'scripts'))

from utils.helpers import get_path, _sha256, atomic_write, TRAINING_DATA_MANIFEST
from data.preprocessor import MIXTURE_FILES, QRELS_FILE, QUERIES_FILE, CORPUS_FILE
from maxp_training import CHUNK_ROW_SUFFIX


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, type=Path)
    args = parser.parse_args()
    dest = get_path("training")
    if dest.exists():
        raise SystemExit(f"{dest} already exists; refusing to touch it.")

    names = [f"training_mixture/{n}" for n in MIXTURE_FILES] + [
        CORPUS_FILE, QUERIES_FILE, QRELS_FILE]
    staging = dest.with_name(dest.name + ".staging")
    if staging.exists():
        raise SystemExit(f"{staging} exists from an earlier attempt; move it aside.")
    (staging / "training_mixture").mkdir(parents=True)

    files, chunk_rows, rows = {}, 0, {}
    suffix = re.compile(CHUNK_ROW_SUFFIX)
    for name in names:
        src, dst = args.source / name, staging / name
        shutil.copy2(src, dst)
        digest = _sha256(dst)
        if digest != _sha256(src):
            raise SystemExit(f"{name}: the copy does not hash like its source")
        files[name] = {'bytes': dst.stat().st_size, 'sha256': digest}
        print(f"  {name}: {files[name]['bytes']:,} bytes  {digest[:12]}", flush=True)
        if name.startswith("training_mixture/"):
            n = 0
            with open(dst, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        n += 1
                        chunk_rows += bool(suffix.search(str(json.loads(line)['query_id'])))
            rows[name] = n
    print(f"  mixture rows: {rows} = {sum(rows.values()):,}; chunk-row ids: {chunk_rows}")
    if chunk_rows:
        raise SystemExit(f"{chunk_rows} chunk-row id(s) in {args.source}: not the original data")

    with atomic_write(staging / TRAINING_DATA_MANIFEST) as f:
        json.dump({'schema': 'training-data-v1', 'source': str(args.source),
                   'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'files': files, 'mixture_rows': rows, 'chunk_rows': chunk_rows},
                  f, indent=2)
    staging.rename(dest)
    print(f"✅ {dest} ({sum(rows.values()):,} rows) — nothing else was modified.")


if __name__ == "__main__":
    main()
