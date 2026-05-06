"""CB Dim scatter — one subplot per codebook size, trained + WF theory.

Generates: ae_warmup_scatter_v3_allsamples_nox_cbdim.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOTS = [
    Path("results"),
]
OUT = ROOTS[0]


def find_run(pattern):
    cands = []
    for root in ROOTS:
        cands.extend(sorted(root.glob(pattern)))
    return cands[-1] if cands else None


def get_final_metrics(run_dir):
    if not run_dir:
        return None
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return None
    last_train = None
    for line in open(p):
        r = json.loads(line)
        if "train/codebook_eff_dim_99" in r:
            last_train = r
    if not last_train:
        return None
    return {"eff_dim": last_train.get("train/codebook_eff_dim_99")}


WARM_STEPS = [1000, 5000, 10000, 15000, 20000, 30000, 40000]
CB_CONFIGS = [
    (1024, "cb1k", "tab:blue", r"$|\mathcal{C}|$ = 1k", 10),
    (16384, "cb16k", "tab:orange", r"$|\mathcal{C}|$ = 16k", 14),
    (65536, "cb65k", "tab:green", r"$|\mathcal{C}|$ = 65k", 16),
]


def load_trained():
    trained = {}
    for ae_steps in WARM_STEPS:
        for n_embed, cb_tag, _, _, _ in CB_CONFIGS:
            d = find_run(f"2*_ae_constlr_warm{ae_steps//1000}k_v3_ref128_{cb_tag}_const2.88e-4_to100k")
            m = get_final_metrics(d) if d else None
            if m:
                trained[(ae_steps, n_embed)] = m
    for n_embed, cb_tag, _, _, _ in CB_CONFIGS:
        d = find_run(f"2*_ref128_v3_{cb_tag}_fresh_const2.88e-4_100k")
        m = get_final_metrics(d) if d else None
        if m:
            trained[(0, n_embed)] = m
    return trained


def load_wf():
    f = OUT / "waterfilling_per_checkpoint.json"
    if f.exists():
        return {r["ae_steps"]: r for r in json.load(open(f))}
    return {}


def main():
    trained = load_trained()
    wf_by_step = load_wf()
    all_steps = [0] + WARM_STEPS

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for col, (n_embed, cb_tag, color, label, R) in enumerate(CB_CONFIGS):
        ax = axes[col]

        # Trained
        tx, ty = [], []
        for s in all_steps:
            m = trained.get((s, n_embed))
            if m and m.get("eff_dim") is not None:
                tx.append(s)
                ty.append(m["eff_dim"])
        if tx:
            ax.plot(tx, ty, "o-", color=color, ms=6, lw=1.5, label="Trained")

        # WF theory
        wf_x, wf_y = [], []
        for s in WARM_STEPS:
            wf = wf_by_step.get(s)
            if wf and f"active_R{R}" in wf:
                wf_x.append(s)
                wf_y.append(wf[f"active_R{R}"])
        if wf_x:
            ax.plot(wf_x, wf_y, "X--", color=color, ms=10, mew=2.5, lw=1,
                    alpha=0.5, label="WF theory")

        ax.set_title(label, fontsize=12)
        ax.set_xlabel("AE Warm-up Steps", fontsize=10)
        if col == 0:
            ax.set_ylabel("CB Dim \u2191", fontsize=11)
        ax.set_xticks(all_steps)
        ax.set_xticklabels(["0", "1k", "5k", "10k", "15k", "20k", "30k", "40k"], fontsize=8)
        ax.set_xlim(-1000, 42000)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle("VQGAN on ImageNet-100 (20k): Water-filling Dimension vs. Final Codebook Effective Dimension", fontsize=13, y=1.0)
    fig.tight_layout()
    out = OUT / "ae_warmup_scatter_v3_allsamples_nox_cbdim.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
