import argparse
import math
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a list of absolute paths for all .wav files in a LibriTTS split."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        nargs="+",
        help="One or more LibriTTS split directories (e.g., dev-clean train-clean-100).",
    )
    parser.add_argument(
        "--output_file",
        help=(
            "Path for the combined file list. Required when multiple input_dir values "
            "are provided; optional otherwise."
        ),
    )
    parser.add_argument(
        "--downsample_frac",
        type=float,
        default=1.0,
        help="Fraction (0 < r <= 1) of files to sample from each input directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (0 < args.downsample_frac <= 1):
        raise SystemExit("--downsample_frac must satisfy 0 < r <= 1")

    random.seed(42)

    script_dir = Path(__file__).resolve().parent
    input_paths = []
    for raw_input in args.input_dir:
        resolved = Path(raw_input).expanduser()
        if not resolved.is_absolute():
            resolved = (script_dir / resolved).resolve()
        if not resolved.is_dir():
            raise SystemExit(f"Input directory does not exist: {resolved}")
        input_paths.append(resolved)

    multiple_inputs = len(input_paths) > 1

    if multiple_inputs and not args.output_file:
        raise SystemExit("When providing multiple input_dir values, --output_file is required.")

    if args.output_file:
        output_file = Path(args.output_file).expanduser()
        if not output_file.is_absolute():
            output_file = (script_dir / output_file).resolve()
    else:
        split_name = input_paths[0].name
        output_file = script_dir / f"{split_name}_filelist.txt"

    wav_paths = []
    for input_path in input_paths:
        paths = sorted(p.resolve() for p in input_path.rglob("*.wav"))
        if not paths:
            continue
        sample_size = math.ceil(len(paths) * args.downsample_frac)
        sampled = random.sample(paths, sample_size)
        wav_paths.extend(sampled)

    wav_paths = sorted(wav_paths)
    if not wav_paths:
        raise SystemExit(f"No .wav files found in: {', '.join(str(p) for p in input_paths)}")

    with output_file.open("w", encoding="utf-8") as f:
        for wav_path in wav_paths:
            f.write(str(wav_path) + "\n")

    print(f"Wrote {len(wav_paths)} entries to {output_file}")


if __name__ == "__main__":
    main()
