"""
Patch script: add raw AUROC values to the H3 ablation return dicts
====================================================================
Targets the four files where the H3 ablation function computes
auroc_full / auroc_stable / auroc_sensitive / auroc_middle locally but
only returns delta_stable / delta_sensitive / delta_middle:

  - ae_dae_svdd_pipeline_iec104.py
  - ae_dae_svdd_pipeline_smd.py
  - flow_full_pipeline_iec104.py
  - flow_full_pipeline_smd.py

(ablation_h3_lodo3.py and ablation_h3_lodo3_extra_seeds.py already save
"auroc" per record and are NOT touched by this script.)

What it does
------------
For each target file, finds the `return {...}` dict inside the H3
ablation function whose body contains the four delta_* variable names,
and inserts

    "auroc_full": auroc_full,
    "auroc_stable": auroc_stable,
    "auroc_sensitive": auroc_sensitive,
    "auroc_middle": auroc_middle,

right before the closing `}`, matching the existing indentation. It is
idempotent -- if "auroc_full" is already a key in that dict, the file
is skipped. A `.bak` copy of the original is written next to each file
before it is modified, and a unified diff is printed for review.

This only edits the return statement. It assumes auroc_full/stable/
sensitive/middle are already local variables in scope at the return
(per the earlier scan of these files) -- if a given file computes them
under different variable names, that file's diff will come back empty
and is listed as "no match" so you can patch it by hand.

Usage
-----
Dry run (show diffs, write nothing):
    python patch_h3_raw_auroc.py --dry-run /path/to/repo

Apply:
    python patch_h3_raw_auroc.py /path/to/repo
"""

import argparse
import difflib
import os
import re
import sys

TARGET_FILES = [
    "ae_dae_svdd_pipeline_iec104.py",
    "ae_dae_svdd_pipeline_smd.py",
    "flow_full_pipeline_iec104.py",
    "flow_full_pipeline_smd.py",
]

NEW_KEYS_TEMPLATE = (
    '{indent}"auroc_full": auroc_full,\n'
    '{indent}"auroc_stable": auroc_stable,\n'
    '{indent}"auroc_sensitive": auroc_sensitive,\n'
    '{indent}"auroc_middle": auroc_middle,\n'
)

# Matches a `return {` ... `}` block that references all three delta_*
# names somewhere inside it, capturing the indentation of its keys and
# the position of the final closing brace line.
RETURN_BLOCK_RE = re.compile(
    r"(?P<return_kw>return\s*\{\n)"
    r"(?P<body>(?:.*\n)*?)"
    r"(?P<close_indent>[ \t]*)\}",
)


def already_patched(body: str) -> bool:
    return '"auroc_full"' in body


def has_all_deltas(body: str) -> bool:
    return all(k in body for k in ("delta_stable", "delta_sensitive", "delta_middle"))


def guess_key_indent(body: str) -> str:
    for line in body.splitlines():
        m = re.match(r"([ \t]+)\S", line)
        if m:
            return m.group(1)
    return "    "


def patch_source(src: str):
    """Return (patched_src, matched: bool, already_done: bool)."""
    match = None
    for m in RETURN_BLOCK_RE.finditer(src):
        body = m.group("body")
        if has_all_deltas(body):
            match = m
            break
    if match is None:
        return src, False, False

    body = match.group("body")
    if already_patched(body):
        return src, True, True

    indent = guess_key_indent(body)
    insertion = NEW_KEYS_TEMPLATE.format(indent=indent)
    insert_pos = match.start("close_indent")
    patched = src[:insert_pos] + insertion + src[insert_pos:]
    return patched, True, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_root", nargs="?", default=".")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    any_failed = False
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
            print("  Already has \"auroc_full\" in the return dict -- skipping.")
            continue
        if not matched:
            print("  NO MATCH -- could not find a `return {...}` block containing "
                  "delta_stable/delta_sensitive/delta_middle. Patch this one by hand.")
            any_failed = True
            continue

        diff = difflib.unified_diff(
            src.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"{fname} (original)",
            tofile=f"{fname} (patched)",
        )
        print("".join(diff))

        if args.dry_run:
            continue

        bak_path = path + ".bak"
        if not os.path.exists(bak_path):
            with open(bak_path, "w", encoding="utf-8") as f:
                f.write(src)
            print(f"  Backup written: {bak_path}")
        else:
            print(f"  Backup already exists at {bak_path} (not overwritten).")

        with open(path, "w", encoding="utf-8") as f:
            f.write(patched)
        print(f"  Patched: {path}")

    if args.dry_run:
        print("\nDry run complete -- no files written. Re-run without --dry-run to apply.")
    if any_failed:
        print("\nOne or more files need a manual patch -- see NOT FOUND / NO MATCH above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
