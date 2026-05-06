"""3-panel cb=16k summary: Val. Rec. Loss, Codebook Dimension (over training), CB Dim scatter.

Usage: python3 plot_cb16k_3panel.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ROOTS = [
    Path("results"),
]
OUT = ROOTS[0]

FAMILY = "cb16k"
N_EMBED = 16384
R = 14

_cmap = cm.get_cmap("inferno", 9)
COLORS = {
    "warm1k":  _cmap(3),
    "warm5k":  _cmap(4),
    "warm10k": _cmap(5),
    "warm15k": _cmap(6),
    "warm20k": _cmap(7),
    "warm30k": _cmap(8),
    "warm40k": _cmap(9),
}

WARM_RUNS = [
    ("warm1k",  1000,  "2*_ae_constlr_warm1k_v3_ref128_{family}_const2.88e-4_to100k"),
    ("warm5k",  5000,  "2*_ae_constlr_warm5k_v3_ref128_{family}_const2.88e-4_to100k"),
    ("warm10k", 10000, "2*_ae_constlr_warm10k_v3_ref128_{family}_const2.88e-4_to100k"),
    ("warm15k", 15000, "2*_ae_constlr_warm15k_v3_ref128_{family}_const2.88e-4_to100k"),
    ("warm20k", 20000, "2*_ae_constlr_warm20k_v3_ref128_{family}_const2.88e-4_to100k"),
    ("warm30k", 30000, "2*_ae_constlr_warm30k_v3_ref128_{family}_const2.88e-4_to100k"),
    ("warm40k", 40000, "2*_ae_constlr_warm40k_v3_ref128_{family}_const2.88e-4_to100k"),
]

WARM_STEPS = [1000, 5000, 10000, 15000, 20000, 30000, 40000]


def find_run(pattern):
    cands = []
    for root in ROOTS:
        cands.extend(sorted(root.glob(pattern)))
    return cands[-1] if cands else None


def load_metric(run_dir, metric_key, step_offset=0, source="val"):
    xs, ys = [], []
    p = Path(run_dir) / "metrics.jsonl"
    if not p.exists():
        return xs, ys
    for line in open(p):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if source == "val" and "val" in r:
            v = r["val"].get(metric_key)
            if v is not None:
                xs.append(r["step"] + step_offset)
                ys.append(v)
        elif source == "train" and metric_key in r:
            v = r[metric_key]
            if isinstance(v, (int, float)) and v != 0:
                xs.append(r["step"] + step_offset)
                ys.append(v)
    return xs, ys


def get_curves(metric_key="val/rec_loss", source="val", ae_metric_key=None):
    curves = []

    # AE baseline (gray)
    ae_dir = find_run("2*_ae_ref128_constlr_100k")
    if ae_dir:
        ae_mk = ae_metric_key if ae_metric_key else metric_key
        xs, ys = load_metric(ae_dir, ae_mk, 0, source)
        if xs:
            curves.append(("Autoencoder", xs, ys, "gray", "-"))

    d = find_run(f"2*_ref128_{FAMILY}_vanilla_fresh_100k")
    if d:
        xs, ys = load_metric(d, metric_key, 0, source)
        if xs:
            curves.append(("Vanilla VQGAN", xs, ys, "#1560BD", "--"))

    d = find_run(f"2*_ref128_v3_{FAMILY}_fresh_const2.88e-4_100k")
    if d:
        xs, ys = load_metric(d, metric_key, 0, source)
        if xs:
            curves.append(("VQGAN w/ Respawn", xs, ys, "#1560BD", "-"))

    for warm_tag, ae_steps, pattern in WARM_RUNS:
        d = find_run(pattern.format(family=FAMILY))
        if d:
            xs, ys = load_metric(d, metric_key, ae_steps, source)
            if xs:
                pretty = warm_tag.replace("warm", "$T_{wu}$=")
                curves.append((pretty, xs, ys, COLORS[warm_tag], "-"))
    return curves


def get_scatter_data():
    trained = {}
    for ae_steps in WARM_STEPS:
        d = find_run(f"2*_ae_constlr_warm{ae_steps//1000}k_v3_ref128_{FAMILY}_const2.88e-4_to100k")
        if d:
            p = d / "metrics.jsonl"
            if p.exists():
                last_train = None
                for line in open(p):
                    r = json.loads(line)
                    if "train/codebook_eff_dim_99" in r:
                        last_train = r
                if last_train:
                    trained[ae_steps] = last_train.get("train/codebook_eff_dim_99")

    d = find_run(f"2*_ref128_v3_{FAMILY}_fresh_const2.88e-4_100k")
    if d:
        p = d / "metrics.jsonl"
        if p.exists():
            last_train = None
            for line in open(p):
                r = json.loads(line)
                if "train/codebook_eff_dim_99" in r:
                    last_train = r
            if last_train:
                trained[0] = last_train.get("train/codebook_eff_dim_99")

    return trained


def load_wf():
    f = OUT / "waterfilling_per_checkpoint.json"
    if f.exists():
        return {r["ae_steps"]: r for r in json.load(open(f))}
    return {}


def set_step_xticks(ax):
    ticks = [0, 20000, 40000, 60000, 80000, 100000]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0", "20k", "40k", "60k", "80k", "100k"])

def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    lfs = 8
    subplot_title_size = 15
    xlabel_size = 11
    ylabel_size = 11

    # --- Panel 0: Latent Dimension over training ---
    ax = axes[0]
    for lbl, xs, ys, color, ls in get_curves("train/eff_dim_99", "train",
                                              ae_metric_key="train/eff_dim_99"):
        trunc = [(x, y) for x, y in zip(xs, ys) if x <= 100000]
        if not trunc:
            continue
        tx, ty = zip(*trunc)
        ax.plot(tx, ty, label=lbl, color=color, lw=1.8, linestyle=ls)
    ax.set_ylabel("Latent Dimension (99% Var.)", fontsize=ylabel_size)
    ax.set_xlabel("Training Steps", fontsize=xlabel_size)
    ax.set_yscale("log")
    ax.set_xlim(-4000, 104000)
    set_step_xticks(ax)
    ax.grid(True, alpha=0.3, which="both")
    handles, labels = ax.get_legend_handles_labels()
    blank = plt.Line2D([], [], color="none", label="")
    handles = handles[:3] + [blank] + handles[3:]
    labels = labels[:3] + [""] + labels[3:]
    ax.legend(handles, labels, fontsize=lfs, loc="upper center", ncol=3)
    ax.set_title("Latent Dimension ($d_{eff}$)", ha="center", fontsize=subplot_title_size)

    # --- Panel 1: Val. Rec. Loss over training ---
    ax = axes[1]
    for lbl, xs, ys, color, ls in get_curves("val/rec_loss", "val"):
        trunc = [(x, y) for x, y in zip(xs, ys) if x <= 100000]
        if not trunc:
            continue
        tx, ty = zip(*trunc)
        ax.plot(tx, ty, label=lbl, color=color, lw=1.8, linestyle=ls)
    ax.set_ylabel("Val. Rec. Loss (L1 + LPIPS)", fontsize=ylabel_size)
    ax.set_xlabel("Training Steps", fontsize=xlabel_size)
    # ax.set_yscale("log")
    ax.set_xlim(-4000, 104000)
    set_step_xticks(ax)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title("Validation Loss", fontsize=subplot_title_size)

    # --- Panel 2: CB Dim scatter (final value vs warmup duration) ---
    ax = axes[2]
    trained = get_scatter_data()
    wf_by_step = load_wf()
    all_steps = [0] + WARM_STEPS

    tx, ty = [], []
    for s in all_steps:
        if s in trained and trained[s] is not None:
            tx.append(s)
            ty.append(trained[s])
    if tx:
        ax.plot(tx, ty, "o-", color="tab:orange", ms=6, lw=1.5, label="Trained")

    wf_x, wf_y = [], []
    for s in WARM_STEPS:
        wf = wf_by_step.get(s)
        if wf and f"active_R{R}" in wf:
            wf_x.append(s)
            wf_y.append(wf[f"active_R{R}"])
    if wf_x:
        ax.plot(wf_x, wf_y, "X--", color="tab:orange", ms=10, mew=2.5, lw=1,
                alpha=0.5, label="WF Prediction")

    ax.set_xlabel("AE Warm-up Steps", fontsize=xlabel_size)
    ax.set_ylabel("CB Dim \u2191", fontsize=ylabel_size)
    ax.set_xticks(all_steps)
    ax.set_xticklabels(["0", "1k", "5k", "10k", "15k", "20k", "30k", "40k"], fontsize=8)
    ax.set_xlim(-1000, 42000)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=lfs, loc="best")
    ax.set_title("Final Codebook Dimension vs Prediction", fontsize=subplot_title_size)

    fig.suptitle(r"VQGAN on ImageNet-100 (20k): Codebook Size = 16 384 ($2^{14}$), 100k Total Steps",
                 fontsize=15, y=1.02)
    out = OUT / "cb16k_3panel_v3.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
