"""Build tab_detection.tex (AUPRC, recall@1%, recall@5% FPR; Single vs Diverse)
from eval_extra_followup_stats.json. Run from the repo folder."""
import json, os, sys

PILOT = r"C:\MLProject\zero_day_project\paper2\pilot"
src = os.path.join(PILOT, "eval_extra_followup_stats.json")
d = json.load(open(src))["detection_tests"]

DS = [("unsw", "UNSW-NB15"), ("iec104", "IEC-104"), ("smd", "SMD")]
FAM = [("ae", "AE"), ("dae", "DAE"), ("svdd", "Deep SVDD"), ("flow", "Flow")]
MET = [("auprc", "AUPRC"), ("recall_at_fpr1", "Recall@1\\%"), ("recall_at_fpr5", "Recall@5\\%")]

idx = {(r["dataset"], r["family"], r["metric"]): r for r in d}

def fp(x):
    return "$<$.001" if x < 0.001 else f"{x:.3f}".lstrip("0")

head = "Target & Model & " + " & ".join(
    f"\\multicolumn{{3}}{{c}}{{{m[1]}}}" for m in MET) + "\\\\"
cm = "".join(f"\\cmidrule(lr){{{3+3*i}-{5+3*i}}}" for i in range(len(MET)))
sub = " & & " + " & ".join("Single$\\to$Div. & $d$ & $q$" for _ in MET) + "\\\\"

L = ["\\begin{tabular}{ll" + "rrr" * len(MET) + "}\\toprule", head, cm, sub, "\\midrule"]
missing = []
for di, (dk, dn) in enumerate(DS):
    for fi, (fk, fn) in enumerate(FAM):
        cells = []
        for mk, _ in MET:
            r = idx.get((dk, fk, mk))
            if r is None:
                missing.append((dk, fk, mk))
                cells.append("-- & -- & --")
                continue
            cells.append(f"{r['single']:.3f}$\\to${r['lodo3']:.3f} & {r['d']:+.2f} & {fp(r['bh_adj_p'])}")
        L.append((dn if fi == 0 else "") + " & " + fn + " & " + " & ".join(cells) + "\\\\")
    L.append("\\midrule" if di < len(DS) - 1 else "\\bottomrule")
L.append("\\end{tabular}")

out = "tab_detection.tex"
open(out, "w").write("\n".join(L))
print(open(out).read())
print("\nSaved", os.path.abspath(out))
if missing:
    print("MISSING cells:", missing)
print("n rows in detection_tests:", len(d), "| metrics:", sorted({r['metric'] for r in d}))
