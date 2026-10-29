"""
Fix: checkpoint files for feature-subset models were named only by feature COUNT (e.g. ae_lodo3_dim5.pt), so the
Stable, Sensitive and Middle models (5 features each) all pointed at ONE file. The first model trained (Stable) was
silently reused for Sensitive and Middle, which were then scored on their own columns.

This patch makes the file name include a short hash of the feature names for every subset except the full
decorrelated set (whose existing *_dim15 checkpoints are unchanged). Old *_dim5 files are simply ignored.

Run once from the repo root:   python patch_subset_checkpoints.py
Backs up each file as <name>.bak and is safe to run twice.
"""
import os, re, shutil

FILES = ["ae_dae_svdd_pipeline_iec104.py", "ae_dae_svdd_pipeline_smd.py", "ae_dae_svdd_pipeline_unsw.py"]

for fn in FILES:
    if not os.path.exists(fn):
        print("skip (not found):", fn)
        continue
    s = open(fn, encoding="utf-8").read()
    if "subset_tag" in s:
        print("already patched:", fn)
        continue
    shutil.copy(fn, fn + ".bak")

    s = s.replace("import json\n", "import hashlib\nimport json\n", 1)

    s = re.sub(r"def checkpoint_path\(seed, family, condition, input_dim=N_FEATURES\):",
               "def checkpoint_path(seed, family, condition, input_dim=N_FEATURES, subset_tag=\"\"):", s, count=1)
    s = re.sub(r'(    suffix = "" if input_dim == N_FEATURES else f"_dim\{input_dim\}"\n)',
               r'\1    if subset_tag:\n        suffix += f"_{subset_tag}"\n', s, count=1)

    old = "    path = checkpoint_path(seed, family, condition, input_dim)\n"
    new = ("    subset_tag = \"\"\n"
           "    if feature_subset and set(feature_subset) != set(DECORRELATED_FEATURES):\n"
           "        subset_tag = hashlib.md5(\",\".join(feature_subset).encode()).hexdigest()[:8]\n"
           "    path = checkpoint_path(seed, family, condition, input_dim, subset_tag)\n")
    assert old in s, f"{fn}: ensure_checkpoint line not found"
    s = s.replace(old, new, 1)
    open(fn, "w", encoding="utf-8").write(s)
    print("patched:", fn)
