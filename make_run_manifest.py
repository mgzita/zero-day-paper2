"""
Reviewer comment #13: write a reproducibility manifest (software versions, git commit, seeds, SHA-256 of scripts, data
and result files, and the checkpoint file listing). Run from the repo root:   python make_run_manifest.py
Output: <pilot>\\RUN_MANIFEST.json  (ship this with the code)
"""
import glob, hashlib, importlib, json, os, platform, re, subprocess, sys, datetime

PILOT = r"C:\MLProject\zero_day_project\paper2\pilot"
ARTEFACTS = r"C:\MLProject\zero_day_project\artefacts_v2"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ver(m):
    try:
        return importlib.import_module(m).__version__
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def git(*a):
    try:
        return subprocess.check_output(["git", *a], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


man = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "python": sys.version,
       "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
       "packages": {m: ver(m) for m in ("numpy", "scipy", "pandas", "sklearn", "torch", "shap", "statsmodels", "captum")},
       "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
       "scripts": {}, "results": {}, "data": {}, "checkpoints": []}
try:
    import torch
    man["torch_cuda_available"] = torch.cuda.is_available()
    man["torch_threads"] = torch.get_num_threads()
    man["torch_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
except Exception:
    pass

for p in sorted(glob.glob("*.py")):
    txt = open(p, encoding="utf-8", errors="ignore").read()
    m = re.search(r"^SEEDS_TO_RUN\s*=\s*(\[[^\]]*\])", txt, re.M)
    man["scripts"][p] = {"sha256": sha(p), "seeds": m.group(1) if m else None,
                         "sets_torch_seed": "torch.manual_seed" in txt,
                         "sets_numpy_seed": "np.random.seed" in txt or "default_rng" in txt}
for p in sorted(glob.glob(os.path.join(PILOT, "*.json"))):
    man["results"][os.path.basename(p)] = sha(p)
for p in sorted(glob.glob(os.path.join(ARTEFACTS, "**", "*.npz"), recursive=True)):
    man["data"][os.path.relpath(p, ARTEFACTS)] = sha(p)
for p in sorted(glob.glob(os.path.join(ARTEFACTS, "**", "*.pt"), recursive=True)):
    man["checkpoints"].append({"file": os.path.relpath(p, ARTEFACTS), "bytes": os.path.getsize(p),
                               "modified": datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds")})
out = os.path.join(PILOT, "RUN_MANIFEST.json")
json.dump(man, open(out, "w"), indent=2)
print(f"Saved {out}: {len(man['scripts'])} scripts, {len(man['results'])} result files, "
      f"{len(man['data'])} data files, {len(man['checkpoints'])} checkpoints")
print("git commit:", man["git_commit"], "| dirty working tree:", man["git_dirty"])
legacy = [c["file"] for c in man["checkpoints"] if re.search(r"_dim5\.pt$", c["file"])]
if legacy:
    print(f"\nNOTE: {len(legacy)} legacy *_dim5.pt checkpoints (pre-13 Oct, shared by Stable/Sensitive/Middle) are still on disk.")
    print("They are no longer read by the fixed pipelines; consider moving them to a _legacy folder so nobody reuses them.")
