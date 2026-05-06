"""ref128 comparison v3 — all runs use fresh AE ckpts, full k-means (312 batches, 100 iters).

Usage: python3 plot_ref128_comparison_v3.py
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


def find_run(pattern):
    cands = []
    for root in ROOTS:
        cands.extend(sorted(root.glob(pattern)))
    return cands[-1] if cands else None


_cmap = cm.get_cmap("inferno", 9)
COLORS = {
    "warm1k":  _cmap(2),
    "warm5k":  _cmap(3),
    "warm10k": _cmap(4),
    "warm15k": _cmap(5),
    "warm20k": _cmap(6),
    "warm30k": _cmap(7),
    "warm40k": _cmap(8),
}

FAMILIES = ["cb1k", "cb16k", "cb65k"]
FAMILY_TITLES = {
    "cb1k":  "Codebook Size = 1 024 (2\u00b9\u2070)",
    "cb16k": "Codebook Size = 16 384 (2\u00b9\u2074)",
    "cb65k": "Codebook Size = 65 536 (2\u00b9\u2076)",
}
FAMILY_N_EMBED = {"cb1k": 1024, "cb16k": 16384, "cb65k": 65536}

WARM_RUNS = [
    ("warm1k",  1000,  "2*_ae_constlr_warm1k_v3_ref128_{family}_const2.88e-4_to100k"),
    ("warm5k",  5000,  "2*_ae_constlr_warm5k_v3_ref128_{family}_const2.88e-4_to100k"),
    ("warm10k", 10000, "2*_ae_constlr_warm10k_v3_ref128_{family}_const2.88e-4_to100k"),
    ("warm15k", 15000, "2*_ae_constlr_warm15k_v3_ref128_{family}_const2.88e-4_to100k"),
    ("warm20k", 20000, "2*_ae_constlr_warm20k_v3_ref128_{family}_const2.88e-4_to100k"),
    ("warm30k", 30000, "2*_ae_constlr_warm30k_v3_ref128_{family}_const2.88e-4_to100k"),
    ("warm40k", 40000, "2*_ae_constlr_warm40k_v3_ref128_{family}_const2.88e-4_to100k"),
]


def get_curves(family, metric_key="val/rec_loss", source="val",
               include_vanilla=False, include_beta0=False):
    curves = []
    if include_vanilla:
        d = find_run(f"2*_ref128_{family}_vanilla_fresh_100k")
        if d:
            xs, ys = load_metric(d, metric_key, 0, source)
            if xs:
                curves.append(("Vanilla VQGAN", xs, ys, "black", "--"))

    d = find_run(f"2*_ref128_v3_{family}_fresh_const2.88e-4_100k")
    if d:
        xs, ys = load_metric(d, metric_key, 0, source)
        if xs:
            curves.append(("VQGAN w/ Respawn", xs, ys, "black", "-"))

    if include_beta0:
        d = find_run(f"2*_ref128_v3_{family}_vanilla_beta0_100k")
        if d:
            xs, ys = load_metric(d, metric_key, 0, source)
            if xs:
                curves.append(("Vanilla \u03b2=0", xs, ys, "tab:green", "--"))
        d = find_run(f"2*_ref128_v3_{family}_respawn_beta0_100k")
        if d:
            xs, ys = load_metric(d, metric_key, 0, source)
            if xs:
                curves.append(("Respawn \u03b2=0", xs, ys, "tab:green", "-"))

    for warm_tag, ae_steps, pattern in WARM_RUNS:
        d = find_run(pattern.format(family=family))
        if d:
            xs, ys = load_metric(d, metric_key, ae_steps, source)
            if xs:
                pretty = warm_tag.replace("warm", "AE Warm-up ")
                curves.append((pretty, xs, ys, COLORS[warm_tag], "-"))
    return curves


def get_curves_util(family, include_vanilla=False, include_beta0=False):
    n_embed = FAMILY_N_EMBED[family]
    raw = get_curves(family, "val/codebook_used", source="val",
                     include_vanilla=include_vanilla, include_beta0=include_beta0)
    return [(lbl, xs, [y / n_embed * 100 for y in ys], color, ls)
            for lbl, xs, ys, color, ls in raw]


def make_plot(include_beta0=False):
    lfs = 6 if include_beta0 else 7
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for col, family in enumerate(FAMILIES):
        ax = axes[0, col]
        for lbl, xs, ys, color, ls in get_curves(family, "val/rec_loss", "val", True, include_beta0):
            trunc = [(x, y) for x, y in zip(xs, ys) if x <= 100000]
            if not trunc: continue
            tx, ty = zip(*trunc)
            ax.plot(tx, ty, label=lbl, color=color, lw=1.8, linestyle=ls)
        ax.set_title(FAMILY_TITLES[family], fontsize=18)
        if col == 0: ax.set_ylabel("Val. Rec. Loss (L1 + LPIPS)")
        ax.set_yscale("log"); ax.set_xlim(0, 105000)
        ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=lfs, loc="upper right")

        ax = axes[1, col]
        for lbl, xs, ys, color, ls in get_curves(family, "train/codebook_eff_dim_99", "train", True, include_beta0):
            trunc = [(x, y) for x, y in zip(xs, ys) if x <= 100000]
            if not trunc: continue
            tx, ty = zip(*trunc)
            ax.plot(tx, ty, label=lbl, color=color, lw=1.8, linestyle=ls)
        if col == 0: ax.set_ylabel("Codebook Dimension (99% Var.)")
        ax.set_xlabel("Training Steps"); ax.set_yscale("log"); ax.set_xlim(0, 105000)
        ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=lfs, loc="best")

    beta_str = " \u2014 including \u03b2=0 baselines (green)" if include_beta0 else ""
    fig.suptitle(f"VQGAN on ImageNet-100 (20k) at 100k total steps {beta_str}: Reconstruction Loss (top) and Codebook Dimension (bottom)", fontsize=14, y=0.955)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    suffix = "_withbeta0" if include_beta0 else ""
    out = OUT / f"ref128_comparison_100k_loss_dim_v3{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def make_plot_with_util(include_beta0=False):
    lfs = 6 if include_beta0 else 7
    nrows = 4 if include_beta0 else 3
    fig, axes = plt.subplots(nrows, 3, figsize=(18, 3.5 * nrows))
    for col, family in enumerate(FAMILIES):
        ax = axes[0, col]
        for lbl, xs, ys, color, ls in get_curves(family, "val/rec_loss", "val", True, include_beta0):
            trunc = [(x, y) for x, y in zip(xs, ys) if x <= 100000]
            if not trunc: continue
            tx, ty = zip(*trunc)
            ax.plot(tx, ty, label=lbl, color=color, lw=1.8, linestyle=ls)
        ax.set_title(FAMILY_TITLES[family], fontsize=12)
        if col == 0: ax.set_ylabel("Val. Rec. Loss (L1 + LPIPS)")
        ax.set_yscale("log"); ax.set_xlim(0, 105000)
        ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=lfs, loc="upper right")

        ax = axes[1, col]
        for lbl, xs, ys, color, ls in get_curves(family, "train/codebook_eff_dim_99", "train", True, include_beta0):
            trunc = [(x, y) for x, y in zip(xs, ys) if x <= 100000]
            if not trunc: continue
            tx, ty = zip(*trunc)
            ax.plot(tx, ty, label=lbl, color=color, lw=1.8, linestyle=ls)
        if col == 0: ax.set_ylabel("Codebook Dimension (99% Var.)")
        ax.set_yscale("log"); ax.set_xlim(0, 105000)
        ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=lfs, loc="best")

        ax = axes[2, col]
        for lbl, xs, ys, color, ls in get_curves_util(family, True, include_beta0):
            trunc = [(x, y) for x, y in zip(xs, ys) if x <= 100000]
            if not trunc: continue
            tx, ty = zip(*trunc)
            ax.plot(tx, ty, label=lbl, color=color, lw=1.8, linestyle=ls)
        if col == 0: ax.set_ylabel("Codebook Utilization (%)")
        ax.set_xlim(0, 105000); ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=lfs, loc="best")

        if include_beta0:
            ax = axes[3, col]
            for lbl, xs, ys, color, ls in get_curves(family, "train/eff_dim_99", "train", True, True):
                trunc = [(x, y) for x, y in zip(xs, ys) if x <= 100000]
                if not trunc: continue
                tx, ty = zip(*trunc)
                ax.plot(tx, ty, label=lbl, color=color, lw=1.8, linestyle=ls)
            if col == 0: ax.set_ylabel("Latent Dimension (99% Var.)")
            ax.set_yscale("log"); ax.set_xlim(0, 105000)
            ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=lfs, loc="best")

        axes[-1, col].set_xlabel("Training Steps")

    beta_str = " incl. \u03b2=0" if include_beta0 else ""
    rows_str = "Reconstruction loss, Codebook dim., Utilization" + (", Latent dim." if include_beta0 else "")
    fig.suptitle(f"VQGAN on ImageNet-100 (20k) {beta_str}: {rows_str}", fontsize=20, y=0.965)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    suffix = "_withbeta0" if include_beta0 else ""
    out = OUT / f"ref128_comparison_100k_loss_dim_util_v3{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta0", action="store_true", help="include beta=0 baselines")
    args = parser.parse_args()

    make_plot(include_beta0=args.beta0)
    make_plot_with_util(include_beta0=args.beta0)
