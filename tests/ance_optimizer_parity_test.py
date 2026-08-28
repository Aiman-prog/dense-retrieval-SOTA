"""BRIGHT ANCE and GRASS must build the same optimizer.

The two arms are compared to isolate NEGATIVE SELECTION, so every other moving part
has to be pinned. The optimizer was not: `run_grass.py` chose between
`bnb.optim.AdamW8bit` and `torch.optim.AdamW` on whether bitsandbytes happened to
import, while `run_ance_train.py` always built `torch.optim.AdamW`. bitsandbytes is
absent here and in the container, so the arms agreed by luck; installing it as a
transitive dependency of anything would have switched one arm to a quantized
optimizer with no visible signal and no test to catch it.

Betas and eps were also left implicit on both sides, which matches only for as long
as torch's defaults do not move.

Run: python tests/ance_optimizer_parity_test.py
"""
import ast
import os
import sys
import traceback
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

import torch                                                       # noqa: E402
from utils.helpers import (                                        # noqa: E402
    ADAMW_BETAS, ADAMW_EPS, build_adamw, load_config, optimizer_specs_agree,
)

GRASS_SRC = project_root / 'scripts' / 'run_grass.py'
ANCE_SRC = project_root / 'scripts' / 'run_ance_train.py'


def _assert_raises(exc, fn, contains=None):
    try:
        fn()
    except exc as e:
        assert contains is None or contains in str(e), str(e)
        return str(e)
    raise AssertionError(f"expected {exc.__name__}")


def _params():
    return [torch.nn.Parameter(torch.zeros(2))]


# ---- the factory ------------------------------------------------------------

def test_builds_torch_adamw_with_explicit_hyperparameters():
    opt, spec = build_adamw(_params(), lr=1e-5, weight_decay=0.01, label='x')
    assert isinstance(opt, torch.optim.AdamW), type(opt)
    group = opt.param_groups[0]
    assert group['lr'] == 1e-5
    assert tuple(group['betas']) == ADAMW_BETAS, group['betas']
    assert group['eps'] == ADAMW_EPS, group['eps']
    assert group['weight_decay'] == 0.01


def test_spec_describes_the_optimizer_that_was_built():
    opt, spec = build_adamw(_params(), lr=2e-5, weight_decay=0.1, label='x')
    group = opt.param_groups[0]
    assert spec['optimizer'] == 'torch.optim.AdamW'
    assert spec['lr'] == group['lr']
    assert tuple(spec['betas']) == tuple(group['betas'])
    assert spec['eps'] == group['eps']
    assert spec['weight_decay'] == group['weight_decay']


def test_betas_and_eps_are_not_left_to_torch_defaults():
    """Pinned explicitly, so a future torch default change cannot move one arm."""
    src = (project_root / 'src' / 'utils' / 'helpers.py').read_text()
    assert 'ADAMW_BETAS = (0.9, 0.999)' in src
    assert 'ADAMW_EPS = 1e-8' in src
    call = src[src.index('optimizer = torch.optim.AdamW('):]
    call = call[:call.index(')') + 1]
    for kw in ('betas=ADAMW_BETAS', 'eps=ADAMW_EPS', 'weight_decay=weight_decay'):
        assert kw in call, f"{kw} missing from {call!r}"


def test_string_hyperparameters_are_coerced():
    """config.yaml writes 1e-5 as a string under some loaders; a str lr silently
    breaks AdamW's arithmetic rather than raising."""
    opt, spec = build_adamw(_params(), lr='1e-5', weight_decay='0.01', label='x')
    assert opt.param_groups[0]['lr'] == 1e-5
    assert isinstance(spec['lr'], float) and isinstance(spec['weight_decay'], float)


# ---- the two arms agree -----------------------------------------------------

def _spec_for(recipe_name, label):
    recipe = load_config()['training'][recipe_name]
    _, spec = build_adamw(_params(), lr=recipe['learning_rate'],
                          weight_decay=recipe['weight_decay'], label=label)
    return spec


def test_ance_and_grass_specs_are_identical():
    """Built from the REAL config.yaml recipes, not from fixtures."""
    ance, grass = _spec_for('ance', 'ance'), _spec_for('grass', 'grass')
    assert optimizer_specs_agree(ance, grass), (ance, grass)


def test_only_the_label_distinguishes_them():
    ance, grass = _spec_for('ance', 'ance'), _spec_for('grass', 'grass')
    assert ance['label'] != grass['label']
    assert {k: v for k, v in ance.items() if k != 'label'} == \
           {k: v for k, v in grass.items() if k != 'label'}


def test_a_differing_learning_rate_is_detected():
    """The check must actually bite: config drift in either recipe fails here."""
    a = _spec_for('ance', 'ance')
    b = dict(_spec_for('grass', 'grass'), lr=a['lr'] * 2)
    assert not optimizer_specs_agree(a, b)


def test_a_differing_optimizer_class_is_detected():
    a = _spec_for('ance', 'ance')
    b = dict(_spec_for('grass', 'grass'), optimizer='bnb.optim.AdamW8bit')
    assert not optimizer_specs_agree(a, b)


# ---- no ambient import can change the class ---------------------------------

def _imported_names(path):
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split('.')[0])
    return names


def test_grass_no_longer_imports_bitsandbytes():
    """Re-adding the ambient branch fails here rather than at comparison time."""
    assert 'bitsandbytes' not in _imported_names(GRASS_SRC)
    src = GRASS_SRC.read_text()
    for token in ('_BNB_AVAILABLE', 'AdamW8bit', 'bnb.optim'):
        # the explanatory comment names AdamW8bit; only executable code may not
        code = "\n".join(l for l in src.splitlines()
                         if not l.lstrip().startswith('#'))
        assert token not in code, f"{token} is still reachable in run_grass.py"


def test_neither_arm_constructs_an_optimizer_directly():
    """Both must go through the factory, or the pinning is decorative."""
    for path in (GRASS_SRC, ANCE_SRC):
        code = "\n".join(l for l in path.read_text().splitlines()
                         if not l.lstrip().startswith('#'))
        assert 'build_adamw(' in code, f"{path.name} does not use the factory"
        assert 'torch.optim.AdamW(' not in code, f"{path.name} builds AdamW directly"
        assert 'AdamW(' not in code.replace('build_adamw(', ''), \
            f"{path.name} builds AdamW directly"


def test_grass_keeps_gradient_checkpointing_unconditional():
    """It used to sit in the else-arm of the bitsandbytes branch; removing that
    branch without hoisting this line strips GRASS's memory fix at q_max_len 1024."""
    code = "\n".join(l for l in GRASS_SRC.read_text().splitlines()
                     if not l.lstrip().startswith('#'))
    assert 'student.gradient_checkpointing_enable()' in code
    for line in code.splitlines():
        if 'student.gradient_checkpointing_enable()' in line:
            indent = len(line) - len(line.lstrip())
            assert indent == 4, f"still nested under a branch (indent {indent})"


TESTS = [
    ("factory: torch AdamW with explicit hyperparameters", test_builds_torch_adamw_with_explicit_hyperparameters),
    ("factory: spec describes what was built", test_spec_describes_the_optimizer_that_was_built),
    ("factory: betas/eps not left to torch defaults", test_betas_and_eps_are_not_left_to_torch_defaults),
    ("factory: string hyperparameters coerced", test_string_hyperparameters_are_coerced),
    ("parity: ANCE and GRASS specs identical", test_ance_and_grass_specs_are_identical),
    ("parity: only the label differs", test_only_the_label_distinguishes_them),
    ("parity: a differing lr is detected", test_a_differing_learning_rate_is_detected),
    ("parity: a differing class is detected", test_a_differing_optimizer_class_is_detected),
    ("ambient: GRASS does not import bitsandbytes", test_grass_no_longer_imports_bitsandbytes),
    ("ambient: neither arm builds AdamW directly", test_neither_arm_constructs_an_optimizer_directly),
    ("grass: gradient checkpointing unconditional", test_grass_keeps_gradient_checkpointing_unconditional),
]


def _run(name, fn):
    try:
        fn()
    except Exception as e:                                        # noqa: BLE001
        print(f"  ❌ {name}\n       {type(e).__name__}: {e}")
        if os.environ.get("TEST_TRACE"):
            traceback.print_exc()
        return False
    print(f"  ✅ {name}")
    return True


def main():
    print("\nANCE / GRASS optimizer parity tests")
    print("=" * 58)
    passed = sum(_run(n, f) for n, f in TESTS)
    print("=" * 58)
    print(f"  {passed}/{len(TESTS)} passed")
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
