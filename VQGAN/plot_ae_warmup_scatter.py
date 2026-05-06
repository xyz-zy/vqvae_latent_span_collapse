"""AE warmup scatter plots — final metrics vs warmup duration.

Usage:
  python3 plot_ae_warmup_scatter.py                  # full 2×3 with no-FT and WF theory
  python3 plot_ae_warmup_scatter.py --nox             # trained only, no no-FT points
  python3 plot_ae_warmup_scatter.py --recloss          # 1×3: L1, LPIPS, rFID only
  python3 plot_ae_warmup_scatter.py --nox --recloss    # combined
  python3 plot_ae_warmup_scatter.py --nox --recloss --errorbars  # with std error bars
"""
import argparse
import json
import numpy as np
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
    last_val, last_train = None, None
    for line in open(p):
        r = json.loads(line)
        if "val" in r:
            last_val = r
        if "train/codebook_eff_dim_99" in r:
            last_train = r
    if not last_val:
        return None
    v = last_val["val"]
    rec = v.get("val/rec_loss")
    lpips = v.get("val/lpips")
    return {
        "rec_loss": rec,
        "l1": (rec - lpips) if rec and lpips else None,
        "lpips": lpips,
        "rfid": v.get("val/rfid"),
        "cb_used": v.get("val/codebook_used"),
        "eff_dim": last_train.get("train/codebook_eff_dim_99") if last_train else None,
    }


WARM_STEPS = [1000, 5000, 10000, 15000, 20000, 30000, 40000]
STEP0_STEPS = [1000, 5000, 10000, 15000, 20000, 30000, 40000, 60000, 80000, 100000]
CB_CONFIGS = [
    (1024, "cb1k", "tab:blue", "$|\mathcal{C}|$ = 1k", 10),
    (16384, "cb16k", "tab:orange", "$|\mathcal{C}|$ = 16k", 14),
    (65536, "cb65k", "tab:green", "$|\mathcal{C}|$ = 65k", 16),
]

ALL_METRICS = [
    ("l1", "L1 \u2193"),
    ("lpips", "LPIPS \u2193"),
    ("rec_loss", "Rec Loss \u2193"),
    ("eff_dim", "CB Dim \u2191"),
    ("cb_util", "CB Util %"),
    ("rfid", "rFID \u2193"),
]
RECLOSS_METRICS = [
    ("l1", "L1 \u2193"),
    ("lpips", "LPIPS \u2193"),
    ("rfid", "rFID \u2193"),
]


def load_trained():
    trained = {}
    for ae_steps in WARM_STEPS:
        for n_embed, cb_tag, _, _, _ in CB_CONFIGS:
            d = find_run(f"2*_ae_constlr_warm{ae_steps//1000}k_v3_ref128_{cb_tag}_const2.88e-4_to100k")
            m = get_final_metrics(d) if d else None
            if m:
                m["cb_util"] = m["cb_used"] / n_embed * 100 if m.get("cb_used") else None
                trained[(ae_steps, n_embed)] = m
    for n_embed, cb_tag, _, _, _ in CB_CONFIGS:
        d = find_run(f"2*_ref128_v3_{cb_tag}_fresh_const2.88e-4_100k")
        m = get_final_metrics(d) if d else None
        if m:
            m["cb_util"] = m["cb_used"] / n_embed * 100 if m.get("cb_used") else None
            trained[(0, n_embed)] = m
    return trained


def load_step0():
    step0 = {}
    f = OUT / "ae_kmeans_step0_eval_allsamples.jsonl"
    if f.exists():
        for line in open(f):
            r = json.loads(line)
            r["cb_util"] = r.get("util_pct", 0)
            step0[(r["ae_steps"], r["n_embed"])] = r
    return step0


def get_final_std(run_dir, n=5):
    if not run_dir:
        return {}
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return {}
    vals = []
    for line in open(p):
        r = json.loads(line)
        if "val" in r:
            v = r["val"]
            rec = v.get("val/rec_loss")
            lpips = v.get("val/lpips")
            vals.append({
                "l1": (rec - lpips) if rec is not None and lpips is not None else None,
                "lpips": lpips,
                "rec_loss": rec,
                "rfid": v.get("val/rfid"),
            })
    vals = vals[-n:]
    if len(vals) < 2:
        return {}
    out = {}
    for key in ["l1", "lpips", "rec_loss", "rfid"]:
        vs = [v[key] for v in vals if v[key] is not None]
        out[key] = float(np.std(vs)) if len(vs) >= 2 else 0.0
    return out


def load_trained_std():
    std = {}
    for ae_steps in WARM_STEPS:
        for n_embed, cb_tag, _, _, _ in CB_CONFIGS:
            d = find_run(f"2*_ae_constlr_warm{ae_steps//1000}k_v3_ref128_{cb_tag}_const2.88e-4_to100k")
            s = get_final_std(d) if d else {}
            if s:
                std[(ae_steps, n_embed)] = s
    for n_embed, cb_tag, _, _, _ in CB_CONFIGS:
        d = find_run(f"2*_ref128_v3_{cb_tag}_fresh_const2.88e-4_100k")
        s = get_final_std(d) if d else {}
        if s:
            std[(0, n_embed)] = s
    return std


def load_wf():
    f = OUT / "waterfilling_per_checkpoint.json"
    if f.exists():
        return {r["ae_steps"]: r for r in json.load(open(f))}
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nox", action="store_true", help="exclude no-FT (step-0) points")
    parser.add_argument("--recloss", action="store_true", help="1×3 plot: L1, LPIPS, rFID only")
    parser.add_argument("--errorbars", action="store_true", help="add std error bars to trained points")
    args = parser.parse_args()

    trained = load_trained()
    trained_std = load_trained_std() if args.errorbars else {}
    step0 = load_step0()
    wf_by_step = load_wf()

    metrics = RECLOSS_METRICS if args.recloss else ALL_METRICS
    nrows = 1 if args.recloss else 2
    ncols = 3
    all_trained_steps = [0] + WARM_STEPS

    fig, axes_raw = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows))
    if nrows == 1:
        axes = [axes_raw]
    else:
        axes = axes_raw

    for idx, (metric_key, ylabel) in enumerate(metrics):
        ax = axes[idx // ncols][idx % ncols] if nrows > 1 else axes[0][idx]
        for n_embed, cb_tag, color, label, R in CB_CONFIGS:
            # Trained (O)
            tx, ty, te = [], [], []
            for s in all_trained_steps:
                m = trained.get((s, n_embed))
                if m and m.get(metric_key) is not None:
                    tx.append(s)
                    ty.append(m[metric_key])
                    sd = trained_std.get((s, n_embed), {})
                    te.append(sd.get(metric_key, 0.0))
            if tx:
                lbl = label if args.nox else f"{label} (trained)"
                if args.errorbars:
                    ax.errorbar(tx, ty, yerr=te, fmt="o-", color=color, ms=6, lw=1.5,
                                capsize=3, capthick=1, label=lbl)
                else:
                    ax.plot(tx, ty, "o-", color=color, ms=6, lw=1.5, label=lbl)

            if not args.nox:
                # No-FT (X)
                sx, sy = [], []
                key_name = "eff_dim_99" if metric_key == "eff_dim" else metric_key
                for s in STEP0_STEPS:
                    r = step0.get((s, n_embed))
                    if r and r.get(key_name) is not None:
                        sx.append(s)
                        sy.append(r[key_name])
                if sx:
                    ax.plot(sx, sy, "x--", color=color, ms=7, lw=1, alpha=0.6,
                            label=f"{label} (no FT)")

            # WF theory on CB Dim subplot
            if metric_key == "eff_dim":
                wf_x, wf_y = [], []
                wf_steps = WARM_STEPS if args.nox else STEP0_STEPS
                for s in wf_steps:
                    wf = wf_by_step.get(s)
                    if wf and f"active_R{R}" in wf:
                        wf_x.append(s)
                        wf_y.append(wf[f"active_R{R}"])
                if wf_x:
                    ax.plot(wf_x, wf_y, "X--", color=color, ms=10, mew=2.5, lw=1,
                            alpha=0.5, label=f"{label} (WF theory)")

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel("AE Warm-up Steps", fontsize=10)
        if args.nox:
            ax.set_xticks(all_trained_steps)
            ax.set_xticklabels(["0", "1k", "5k", "10k", "15k", "20k", "30k", "40k"], fontsize=8)
            ax.set_xlim(-1000, 42000)
        else:
            all_ticks = sorted(set(all_trained_steps + STEP0_STEPS))
            ax.set_xticks(all_ticks)
            ax.set_xticklabels(["0", "1k", "5k", "10k", "15k", "20k", "30k", "40k",
                                "60k", "80k", "100k"], fontsize=7)
        ax.grid(True, alpha=0.3)
        lfs = 9 if args.recloss else (8 if args.nox else 5.5)
        if idx == 0:
            ax.legend(fontsize=lfs, loc="best", ncol=1 if args.recloss else 2)
        elif metric_key == "eff_dim":
            ax.legend(fontsize=min(lfs, 6), loc="best", ncol=2 if not args.nox else 2)

    fig.suptitle("VQGAN on ImageNet-100 (20k): Final Metrics at 100k Steps", fontsize=13,
                 y=1.0 if args.recloss else 0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96] if not args.recloss else None)

    parts = ["ae_warmup_scatter_v3_allsamples"]
    if args.nox:
        parts.append("nox")
    if args.recloss:
        parts.append("recloss")
    if args.errorbars:
        parts.append("errorbars")
    out = OUT / f"{'_'.join(parts)}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
