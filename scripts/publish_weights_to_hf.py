#!/usr/bin/env python3
"""
Publish Frozen Perception Weights to Hugging Face Hub.

Verifies the frozen ResNet34-UNet checkpoint 'best.pt' against the official
cryptographic SHA-256 checksum and uploads it to a specified Hugging Face
model repository using huggingface_hub.HfApi.

Usage:
    # Dry-run validation (verifies SHA-256 without network upload):
    python scripts/publish_weights_to_hf.py --dry-run

    # Upload weights to Hugging Face Hub:
    python scripts/publish_weights_to_hf.py --repo-id <username>/<model-repo-name>
"""

import os
import sys
import hashlib
import argparse
from pathlib import Path

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import EXPECTED_SHA256, DEFAULT_CHECKPOINT_CANDIDATES


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash in 64 KB binary chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def find_weights_file(custom_path: str = None) -> Path:
    """Resolves local path to best.pt."""
    if custom_path:
        p = Path(custom_path)
        if p.exists() and p.is_file():
            return p
        raise FileNotFoundError(f"Specified weights file not found: {custom_path}")

    for cand in DEFAULT_CHECKPOINT_CANDIDATES:
        if cand.exists() and cand.is_file():
            return cand

    raise FileNotFoundError(
        "Could not locate 'best.pt' at default candidate paths: "
        f"{[str(p) for p in DEFAULT_CHECKPOINT_CANDIDATES]}"
    )


def print_hf_space_instructions(repo_id: str):
    """Prints step-by-step instructions for Hugging Face Spaces environment configuration."""
    print("\n" + "=" * 80)
    print("HUGGING FACE SPACES CONFIGURATION INSTRUCTIONS")
    print("=" * 80)
    print("To enable the Hugging Face Space to download these perception weights on startup:")
    print("1. Navigate to your Hugging Face Space repository on huggingface.co.")
    print("2. Click on 'Settings' -> 'Variables and secrets'.")
    print("3. In the 'Variables' section (public environment variables), click 'New variable':")
    print("     Name:  HF_MODEL_REPO")
    print(f"     Value: {repo_id}")
    print("4. Restart the Space. The application pipeline in src/pipeline.py will automatically")
    print("   fetch 'best.pt' from this model repository and verify its SHA-256 checksum.")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Verify and publish frozen perception checkpoint best.pt to Hugging Face Hub."
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=None,
        help="Path to best.pt (defaults to scratch/downloaded_final_weights/best.pt or outputs/models/best.pt)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=os.getenv("HF_MODEL_REPO", ""),
        help="Target Hugging Face model repository ID (e.g. 'username/road-pci-models')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN", None),
        help="Hugging Face API token with write permissions (or set via HF_TOKEN env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify weights file integrity and print instructions without uploading to HF Hub",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("ROAD-PCI-INTELLIGENCE: HUGGING FACE WEIGHTS PUBLISHER")
    print("=" * 80)

    # 1. Locate weights
    weights_path = find_weights_file(args.weights_path)
    file_size_mb = weights_path.stat().st_size / (1024 * 1024)
    print(f"[*] Found checkpoint file : {weights_path} ({file_size_mb:.2f} MB)")

    # 2. Verify SHA-256
    print("[*] Computing cryptographic SHA-256 checksum...")
    actual_sha = compute_sha256(weights_path)
    print(f"    Expected : {EXPECTED_SHA256}")
    print(f"    Actual   : {actual_sha}")

    if actual_sha != EXPECTED_SHA256:
        print("[!] ERROR: Checksum mismatch! Refusing to publish invalid weights.")
        sys.exit(1)

    print("[+] Checksum verified: 100% match with canonical frozen checkpoint.")

    target_repo = args.repo_id if args.repo_id else "<YOUR_HF_USERNAME>/road-pci-perception-weights"

    if args.dry_run:
        print("\n[+] DRY RUN ACTIVE: No network requests made.")
        print(f"    Target Hugging Face Model Repo : {target_repo}")
        print(f"    Remote filename               : best.pt")
        print("    File verified and ready for upload.")
        print_hf_space_instructions(target_repo)
        return

    # 3. Live upload
    if not args.repo_id:
        print("[!] ERROR: --repo-id must be specified (or set via HF_MODEL_REPO environment variable).")
        print_hf_space_instructions("<username>/<model-repo>")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[!] ERROR: 'huggingface_hub' is required. Install via: pip install huggingface-hub")
        sys.exit(1)

    print(f"[*] Uploading 'best.pt' to Hugging Face Model Repo: {args.repo_id}...")
    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(weights_path),
        path_in_repo="best.pt",
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"[+] Successfully published 'best.pt' to https://huggingface.co/{args.repo_id}")
    print_hf_space_instructions(args.repo_id)


if __name__ == "__main__":
    main()
