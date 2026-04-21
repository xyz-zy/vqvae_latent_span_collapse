"""Strip VQ codebook from a checkpoint, producing a clean no-VQ model."""

import argparse
import torch


def strip_vq(input_path: str, output_path: str):
    raw = torch.load(input_path, map_location="cpu")
    sd = raw["state_dict"]

    prefix = "feature_extractor.encodec.quantizer.vq.layers.0._codebook"
    K, D = sd[f"{prefix}.embed"].shape
    print(f"Resetting codebook buffers (K={K}, D={D})")

    sd[f"{prefix}.embed"] = torch.zeros(K, D)
    sd[f"{prefix}.embed_avg"] = torch.zeros(K, D)
    sd[f"{prefix}.cluster_size"] = torch.zeros(K)
    sd[f"{prefix}.inited"] = torch.Tensor([False])

    raw["state_dict"] = sd
    torch.save(raw, output_path)
    print(f"Saved stripped checkpoint to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strip VQ codebook from checkpoint")
    parser.add_argument("input", help="Path to input checkpoint with VQ codebook")
    parser.add_argument("output", help="Path to save stripped checkpoint")
    args = parser.parse_args()
    strip_vq(args.input, args.output)
