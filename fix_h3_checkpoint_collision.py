"""
Fix: H3 ablation checkpoint-path collision (IEC-104 / SMD, AE+DAE+SVDD)
========================================================================
BUG (confirmed by reading the source directly):

    def checkpoint_path(seed, family, condition, input_dim=N_FEATURES):
        ckpt_dir = os.path.join(DEDUP_DIR, f"seed{seed}_{family}")
        fname = "dae_" if family == "dae" else f"{family}_"
        suffix = "" if input_dim == N_FEATURES else f"_dim{input_dim}"
        return os.path.join(ckpt_dir, f"{fname}{condition}{suffix}.pt")

    def ensure_checkpoint(seed, family, condition, feature_subset=None):
        input_dim = len(feature_subset) if feature_subset else N_FEATURES
        path = checkpoint_path(seed, family, condition, input_dim)
        if os.path.exists(path):
            return path          # <-- reuses WHATEVER is already saved here
        ... trains and saves to `path` ...

The checkpoint filename is derived only from len(feature_subset), not from
which features are in it. STABLE_V2, SENSITIVE_V2 and MIDDLE_V2 are each
exactly 5 features (both in ae_dae_svdd_pipeline_iec104.py and
ae_dae_svdd_pipeline_smd.py), so h3_ablation()'s three calls to
train_and_score_subset(seed, family, <group>, ...) all resolve to the
SAME checkpoint path (".../lodo3_dim5.pt"). Call order is Stable ->
Sensitive -> Middle, so:
  1. Stable group trains a real model, saves it to *_dim5.pt.
  2. Sensitive group finds *_dim5.pt already exists -> reuses the
     STABLE-trained model, scores it on the Sensitive group's (different)
     feature columns.
  3. Middle group: same -> reuses the Stable-trained model again.

So delta_sensitive/delta_middle (and auroc_sensitive/auroc_middle) for
IEC-104 and SMD's AE/DAE/SVDD H3 results were never computed from a model
actually trained on those groups' features.

NOT affected (checked directly, no fix needed there):
  - ablation_h3_lodo3.py (UNSW)        -- trains fresh inline, no checkpoint reuse.
  - flow_full_pipeline_iec104.py/_smd.py -- train_flow() called directly each
    time, no checkpoint caching at all.

THE FIX
-------
Key the checkpoint filename on the *content* of feature_subset (a short
hash of the sorted feature names), not just its length, whenever a subset
is given. This is permanent and size-agnostic: it cannot collide again
even if group sizes change later. When feature_subset is None (the H1 /
representation-analysis / H3-full-condition calls), behavior and
filenames are UNCHANGED, so none of those existing (correct, expensive)
checkpoints are invalidated or need retraining.

This script:
  1. Patches checkpoint_path()/ensure_checkpoint() in both
     ae_dae_svdd_pipeline_iec104.py and ae_dae_svdd_pipeline_smd.py to use
     the new content-keyed filename for feature subsets.
  2. Adds a `_feature_subset_key()` helper and an `import hashlib` if not
     already present.
  3. Deletes the stale per-seed/family result cache JSON files
     (CACHE_DIR/{family}_seed{seed}.json for both pipelines) so the next
     run recomputes H1 + H3 + representation-analysis with the corrected
     checkpoints instead of reloading the old (partly wrong) cached
     dicts. H1 and representation-analysis results will come out
     numerically identical (they only ever used the full-feature model,
     which was never affected) -- only H3's sensitive/middle numbers will
     actually change.
  4. Does NOT touch or delete the old ambiguous "*_dim5.pt" checkpoints --
     they simply stop being referenced by the new filename scheme and
     become orphaned. They're listed at the end so you can delete them
     by hand if you want the disk space back; nothing depends on that.

It does NOT retrain anything itself -- the next normal run of each
pipeline script will retrain Stable/Sensitive/Middle under their own
distinct checkpoint files (3 extra short trainings per seed/family
instead of 1, since they're no longer able to share), then save fresh,
now-correct, h1/h3/repr results.

Usage
-----
Dry run (show diffs and what would be deleted, change nothing):
    python fix_h3_checkpoint_collision.py --dry-run /path/to/repo

Apply:
    python fix_h3_checkpoint_collision.py /path/to/repo
"""

import argparse
import difflib
import glob
import os
import re
import sys

TARGET_FILES = [
    "ae_dae_svdd_pipeline_iec104.py",
    "ae_dae_svdd_pipeline_smd.py",
]

# Cache dirs as defined in each file (CACHE_DIR = os.path.join(PILOT_DIR, ...)).
# Resolved relative to repo_root/pilot by default; override with --pilot-dir
# if your PILOT_DIR lives elsewhere (e.g. the C:\MLProject\...\paper2\pilot
# path baked into the scripts).
CACHE_SUBDIRS = {
    "ae_dae_svdd_pipeline_iec104.py": "ae_dae_svdd_iec104_cache",
    "ae_dae_svdd_pipeline_smd.py": "ae_dae_svdd_smd_cache",
}

HELPER_FUNC = '''def _feature_subset_key(feature_subset):
    """Stable identifier for a feature subset, for use in checkpoint
    filenames. Keyed on the actual sorted feature names (not just the
    count), so two different same-size subsets never collide on one
    checkpoint file."""
    if not feature_subset:
        return ""
    names = ",".join(sorted(feature_subset))
    digest = hashlib.sha1(names.encode("utf-8")).hexdigest()[:10]
    return f"_feat{len(feature_subset)}_{digest}"


'''

OLD_BLOCK_RE = re.compile(
    r"def checkpoint_path\(seed, family, condition, input_dim=N_FEATURES\):\n"
    r"(?P<ckpt_dir_line>[ \t]*ckpt_dir = .*\n)"
    r"[ \t]*fname = \"dae_\" if family == \"dae\" else f\"\{family\}_\"\n"
    r"[ \t]*suffix = \"\" if input_dim == N_FEATURES else f\"_dim\{input_dim\}\"\n"
    r"[ \t]*return os\.path\.join\(ckpt_dir, f\"\{fname\}\{condition\}\{suffix\}\.pt\"\)\n"
    r"\n\n"
    r"def ensure_checkpoint\(seed, family, condition, feature_subset=None\):\n"
    r"[ \t]*input_dim = len\(feature_subset\) if feature_subset else N_FEATURES\n"
    r"[ \t]*path = checkpoint_path\(seed, family, condition, input_dim\)\n"
    r"[ \t]*if os\.path\.exists\(path\):\n"
    r"[ \t]*return path\n"
)


def build_new_block(ckpt_dir_line: str) -> str:
    return (
        "def checkpoint_path(seed, family, condition, input_dim=N_FEATURES, feature_subset=None):\n"
        f"{ckpt_dir_line}"
        '    fname = "dae_" if family == "dae" else f"{family}_"\n'
        "    if feature_subset:\n"
        "        suffix = _feature_subset_key(feature_subset)\n"
        "    else:\n"
        '        suffix = "" if input_dim == N_FEATURES else f"_dim{input_dim}"\n'
        '    return os.path.join(ckpt_dir, f"{fname}{condition}{suffix}.pt")\n'
        "\n\n"
        "def ensure_checkpoint(seed, family, condition, feature_subset=None):\n"
        "    input_dim = len(feature_subset) if feature_subset else N_FEATURES\n"
        "    path = checkpoint_path(seed, family, condition, input_dim, feature_subset=feature_subset)\n"
        "    if os.path.exists(path):\n"
        "        return path\n"
    )


def patch_source(src: str):
    """Return (patched_src, matched, already_done)."""
    if "_feature_subset_key" in src:
        return src, True, True

    m = OLD_BLOCK_RE.search(src)
    if not m:
        return src, False, False

    new_block = build_new_block(m.group("ckpt_dir_line"))
    patched = src[:m.start()] + new_block + src[m.end():]

    # Insert the helper function right before `def checkpoint_path`.
    helper_anchor = "def checkpoint_path(seed, family, condition, input_dim=N_FEATURES, feature_subset=None):"
    idx = patched.index(helper_anchor)
    patched = patched[:idx] + HELPER_FUNC + patched[idx:]

    # Make sure `import hashlib` is present.
    if "import hashlib" not in patched:
        patched = patched.replace("import json\n", "import json\nimport hashlib\n", 1)
        if "import hashlib" not in patched:
            # Fallback: prepend at top of file.
            patched = "import hashlib\n" + patched

    return patched, True, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_root", nargs="?", default=".")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pilot-dir", default=None,
                     help="Override where CACHE_DIR's parent (PILOT_DIR) lives, "
                          "if different from <repo_root>/pilot.")
    args = ap.parse_args()

    any_failed = False
    stale_cache_files = []

    for fname in TARGET_FILES:
        path = os.path.join(args.repo_root, fname)
        print(f"\n{'=' * 70}\n{fname}\n{'=' * 70}")
        if not os.path.exists(path):
            print("  NOT FOUND -- skipping.")
            any_failed = True
            continue

        with open(path, "r", encoding="utf-8") as f:
            src = f.read()

        patched, matched, already_done = patch_source(src)

        if already_done:
            print("  Already patched (_feature_subset_key present) -- skipping.")
            continue
        if not matched:
            print("  NO MATCH -- the checkpoint_path()/ensure_checkpoint() source didn't "
                  "match the expected pattern exactly. Paste these two functions here and "
                  "I'll hand-write the patch for this file.")
            any_failed = True
            continue

        diff = difflib.unified_diff(
            src.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"{fname} (original)",
            tofile=f"{fname} (patched)",
        )
        print("".join(diff))

        pilot_dir = args.pilot_dir or os.path.join(args.repo_root, "pilot")
        cache_dir = os.path.join(pilot_dir, CACHE_SUBDIRS[fname])
        matches = sorted(glob.glob(os.path.join(cache_dir, "*.json")))
        stale_cache_files.extend(matches)
        print(f"\n  Stale result cache ({cache_dir}): {len(matches)} file(s) to delete "
              f"so H1/H3/representation-analysis recompute with the fixed checkpoints.")

        if args.dry_run:
            continue

        bak_path = path + ".bak"
        if not os.path.exists(bak_path):
            with open(bak_path, "w", encoding="utf-8") as f:
                f.write(src)
            print(f"  Backup written: {bak_path}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(patched)
        print(f"  Patched: {path}")

        for cache_file in matches:
            os.remove(cache_file)
        if matches:
            print(f"  Deleted {len(matches)} stale cache file(s).")

    if args.dry_run:
        print(f"\n{'=' * 70}\nDry run complete -- no files written, nothing deleted.")
        print(f"Would delete {len(stale_cache_files)} stale cache file(s) total:")
        for f in stale_cache_files:
            print(f"  {f}")
        print("Re-run without --dry-run to apply.")

    print(f"\n{'=' * 70}\nNext step: rerun ae_dae_svdd_pipeline_iec104.py and "
          f"ae_dae_svdd_pipeline_smd.py. Each seed/family's H1 and representation-analysis "
          f"will recompute (same numbers as before, just re-derived) and all three H3 "
          f"groups will now train genuinely separate checkpoints -- expect 2 extra short "
          f"trainings per seed/family (Sensitive, Middle) that were previously skipped.")

    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
