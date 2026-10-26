"""Download SCRFD (detection) and ArcFace (recognition) ONNX weights from InsightFace.

Usage:
    python scripts/download_weights.py                # buffalo_l: det_10g + w600k_r50 (accurate)
    python scripts/download_weights.py --pack buffalo_s  # det_500m + w600k_mbf (lightweight CPU)

Note: InsightFace pretrained models are released for non-commercial research use.
"""

import argparse
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

RELEASE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/{pack}.zip"
PACK_FILES = {
    "buffalo_l": ("det_10g.onnx", "w600k_r50.onnx"),
    "buffalo_s": ("det_500m.onnx", "w600k_mbf.onnx"),
}
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "app" / "ml" / "weights"
CHUNK_SIZE_BYTES = 1 << 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", choices=sorted(PACK_FILES), default="buffalo_l")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as file:
        total = int(response.headers.get("Content-Length", 0))
        received = 0
        while chunk := response.read(CHUNK_SIZE_BYTES):
            file.write(chunk)
            received += len(chunk)
            if total:
                print(f"\r  {received / total:6.1%} of {total / 1e6:.0f} MB", end="", flush=True)
    print()


def extract_models(archive_path: Path, file_names: tuple[str, ...], output_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        members = {Path(name).name: name for name in archive.namelist()}
        for file_name in file_names:
            if file_name not in members:
                raise FileNotFoundError(f"{file_name} not found in {archive_path.name}")
            (output_dir / file_name).write_bytes(archive.read(members[file_name]))
            print(f"  saved {output_dir / file_name}")


def main() -> int:
    args = parse_args()
    file_names = PACK_FILES[args.pack]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if all((args.output_dir / name).is_file() for name in file_names):
        print(f"Weights already present in {args.output_dir}")
        return 0
    url = RELEASE_URL.format(pack=args.pack)
    print(f"Downloading {url}")
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / f"{args.pack}.zip"
        download(url, archive_path)
        extract_models(archive_path, file_names, args.output_dir)
    if args.pack != "buffalo_l":
        detector, embedder = file_names
        print(f"Set FRE_DETECTOR_MODEL_FILE={detector} and FRE_EMBEDDER_MODEL_FILE={embedder} in .env")
    return 0


if __name__ == "__main__":
    sys.exit(main())
