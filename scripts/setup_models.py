#!/usr/bin/env python3
"""Setup script for HARM model backends.

Usage:
    python scripts/setup_models.py                  # List all backends and status
    python scripts/setup_models.py --install NAME    # Install a specific backend
    python scripts/setup_models.py --install all     # Install all backends
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_CONFIGS: dict[str, dict] = {
    "ela": {
        "description": "ELA + noise analysis (built-in)",
        "gpu": False,
        "vram_mb": 0,
        "deps": [],
        "repo": None,
        "weights": None,
    },
    "mantranet": {
        "description": "ManTraNet pixel-level forgery detection",
        "gpu": False,
        "vram_mb": 70,
        "deps": ["torch", "torchvision"],
        "repo": {
            "url": "https://github.com/RonyAbecidan/ManTraNet-pytorch.git",
            "target": ROOT_DIR / "models" / "mantranet" / "ManTraNet-pytorch",
        },
        "weights": None,  # Weights are included in the repo
    },
    "trufor": {
        "description": "TruFor classification + localization",
        "gpu": True,
        "vram_mb": 500,
        "deps": ["torch", "torchvision"],
        "repo": {
            "url": "https://github.com/grip-unina/TruFor.git",
            "target": ROOT_DIR / "models" / "trufor" / "TruFor",
        },
        "weights": {
            "info": "Download weights from https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip",
            "target_dir": ROOT_DIR / "models" / "trufor" / "TruFor" / "test_docker" / "weights",
        },
    },
    "catnet": {
        "description": "CAT-Net JPEG compression artifact detection (dual-stream RGB + DCT)",
        "gpu": True,
        "vram_mb": 400,
        "deps": ["torch", "torchvision"],
        "repo": {
            "url": "https://github.com/mjkwon2021/CAT-Net.git",
            "target": ROOT_DIR / "models" / "catnet" / "CAT-Net",
        },
        "weights": {
            "info": "Download weights from the CAT-Net repo releases or Google Drive link in README",
            "target_dir": ROOT_DIR / "models" / "catnet" / "CAT-Net" / "Weights",
        },
    },
    "mvssnet": {
        "description": "MVSS-Net (BayarConv + SRM + ResNet50) classification + localization",
        "gpu": True,
        "vram_mb": 600,
        "deps": ["torch", "torchvision"],
        "repo": {
            "url": "https://github.com/dong03/MVSS-Net.git",
            "target": ROOT_DIR / "models" / "mvssnet" / "MVSS-Net",
        },
        "weights": {
            "info": "Download weights from the MVSS-Net repo (Google Drive link in README)",
            "target_dir": ROOT_DIR / "models" / "mvssnet" / "MVSS-Net" / "ckpt",
        },
    },
    "psccnet": {
        "description": "PSCC-Net compact end-to-end classification + localization",
        "gpu": True,
        "vram_mb": 300,
        "deps": ["torch", "torchvision"],
        "repo": {
            "url": "https://github.com/proteus1991/PSCC-Net.git",
            "target": ROOT_DIR / "models" / "psccnet" / "PSCC-Net",
        },
        "weights": {
            "info": "Download weights from the PSCC-Net repo (Google Drive link in README)",
            "target_dir": ROOT_DIR / "models" / "psccnet" / "PSCC-Net" / "checkpoints",
        },
    },
    "focal": {
        "description": "FOCAL best IoU with unsupervised clustering",
        "gpu": True,
        "vram_mb": 800,
        "deps": ["torch", "torchvision"],
        "repo": {
            "url": "https://github.com/HighwayWu/FOCAL.git",
            "target": ROOT_DIR / "models" / "focal" / "FOCAL",
        },
        "weights": {
            "info": "Download weights from the FOCAL repo (links in README)",
            "target_dir": ROOT_DIR / "models" / "focal" / "FOCAL" / "checkpoints",
        },
    },
    "imlvit": {
        "description": "IML-ViT ViT-Base/16 image manipulation localization with edge supervision",
        "gpu": True,
        "vram_mb": 7000,
        "deps": ["torch", "torchvision", "timm"],
        "repo": {
            "url": "https://github.com/SunnyHaze/IML-ViT.git",
            "target": ROOT_DIR / "models" / "imlvit" / "IML-ViT",
        },
        "weights": {
            "info": "Download CASIAv2-trained checkpoint from Google Drive link in repo README",
            "target_dir": ROOT_DIR / "models" / "imlvit" / "IML-ViT" / "checkpoints",
        },
    },
    "mesorch": {
        "description": "Mesorch multi-scale Transformer+CNN forgery localization (AAAI 2025)",
        "gpu": True,
        "vram_mb": 800,
        "deps": ["torch", "torchvision"],
        "repo": {
            "url": "https://github.com/scu-zjz/Mesorch.git",
            "target": ROOT_DIR / "models" / "mesorch" / "Mesorch",
        },
        "weights": {
            "info": "Download mesorch-98.pth or mesorch_p-118.pth from the repo",
            "target_dir": ROOT_DIR / "models" / "mesorch" / "Mesorch" / "checkpoints",
        },
    },
    "loupe": {
        "description": "Loupe adaptive forgery detection with Perception Encoder (IJCAI 2025 1st Place)",
        "gpu": True,
        "vram_mb": 600,
        "deps": ["torch", "torchvision", "transformers"],
        "repo": {
            "url": "https://github.com/Kamichanw/Loupe.git",
            "target": ROOT_DIR / "models" / "loupe" / "Loupe",
        },
        "weights": {
            "info": "Loupe uses Perception Encoder from HuggingFace; see repo README for setup",
            "target_dir": ROOT_DIR / "models" / "loupe" / "Loupe" / "checkpoints",
        },
    },
    "profact": {
        "description": "ProFact progressive feedback transformer for forgery localization (2024)",
        "gpu": True,
        "vram_mb": 600,
        "deps": ["torch", "torchvision"],
        "repo": {
            "url": "https://github.com/multimediaFor/ProFact.git",
            "target": ROOT_DIR / "models" / "profact" / "ProFact",
        },
        "weights": {
            "info": "Download pretrained weights from the ProFact repo",
            "target_dir": ROOT_DIR / "models" / "profact" / "ProFact" / "checkpoints",
        },
    },
}


def _check_python_dep(package: str) -> bool:
    try:
        __import__(package)
        return True
    except ImportError:
        return False


def _check_repo(config: dict) -> bool:
    repo = config.get("repo")
    if repo is None:
        return True
    return Path(repo["target"]).exists()


def _check_weights(config: dict) -> bool:
    weights = config.get("weights")
    if weights is None:
        return True
    target_dir = Path(weights["target_dir"])
    if not target_dir.exists():
        return False
    return any(target_dir.glob("*.pth")) or any(target_dir.glob("*.pt")) or any(target_dir.glob("*.pth.tar"))


def _check_backend(name: str) -> dict[str, bool]:
    config = MODEL_CONFIGS[name]
    deps_ok = all(_check_python_dep(d) for d in config["deps"])
    repo_ok = _check_repo(config)
    weights_ok = _check_weights(config)
    return {
        "deps": deps_ok,
        "repo": repo_ok,
        "weights": weights_ok,
        "ready": deps_ok and repo_ok and weights_ok,
    }


def list_backends():
    print("\nHARM Model Backends:\n")
    for name, config in MODEL_CONFIGS.items():
        status = _check_backend(name)
        icon = "✓" if status["ready"] else "✗"
        gpu_label = "GPU recommended" if config["gpu"] else "CPU only"
        vram = config["vram_mb"]
        vram_str = f"~{vram} MB" if vram > 0 else "no extra deps"

        print(f"  [{icon}] {name:<12} - {gpu_label}, {vram_str}")
        print(f"      {config['description']}")

        if not status["deps"]:
            missing = [d for d in config["deps"] if not _check_python_dep(d)]
            print(f"      → Missing deps: pip install {' '.join(missing)}")
        if not status["repo"]:
            print(f"      → Repo not cloned: python scripts/setup_models.py --install {name}")
        if not status["weights"] and config["weights"]:
            print(f"      → Weights missing: {config['weights']['info']}")
        print()


def install_backend(name: str):
    if name not in MODEL_CONFIGS:
        print(f"Unknown backend: {name}")
        print(f"Available: {', '.join(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    config = MODEL_CONFIGS[name]

    if config["deps"]:
        missing = [d for d in config["deps"] if not _check_python_dep(d)]
        if missing:
            print(f"Installing Python dependencies: {missing}")
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

    repo = config.get("repo")
    if repo is not None:
        target = Path(repo["target"])
        if not target.exists():
            print(f"Cloning {repo['url']} -> {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(["git", "clone", repo["url"], str(target)])
        else:
            print(f"Repo already exists: {target}")

    weights = config.get("weights")
    if weights is not None:
        target_dir = Path(weights["target_dir"])
        if not _check_weights(config):
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nWeights need manual download:")
            print(f"  {weights['info']}")
            print(f"  Place files in: {target_dir}")
        else:
            print(f"Weights already present in: {target_dir}")

    status = _check_backend(name)
    if status["ready"]:
        print(f"\n{name} backend is ready!")
    else:
        print(f"\n{name} backend is partially set up. Check above for remaining steps.")


def main():
    parser = argparse.ArgumentParser(description="HARM model backend setup")
    parser.add_argument("--install", type=str, help="Install a specific backend (or 'all')")
    args = parser.parse_args()

    if args.install:
        if args.install == "all":
            for name in MODEL_CONFIGS:
                if name == "ela":
                    continue
                print(f"\n{'=' * 40}")
                print(f"Setting up: {name}")
                print(f"{'=' * 40}")
                install_backend(name)
        else:
            install_backend(args.install)
    else:
        list_backends()


if __name__ == "__main__":
    main()
