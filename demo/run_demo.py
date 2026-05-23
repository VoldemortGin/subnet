"""
End-to-end demo: run forgery detection on probe images and score results.

Usage: python -m demo.run_demo
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent

PROBES_DIR = ROOT_DIR / "data" / "probes"
CLEAN_DIR = ROOT_DIR / "data" / "clean"


def check_data() -> bool:
    probe_images = list(PROBES_DIR.glob("*.png")) + list(PROBES_DIR.glob("*.jpg"))
    clean_images = list(CLEAN_DIR.glob("*.png")) + list(CLEAN_DIR.glob("*.jpg"))

    if not probe_images and not clean_images:
        print("=" * 60)
        print("  No sample data found.")
        print()
        print("  Generate samples first:")
        print("    python -m src.validator.generate_samples")
        print("=" * 60)
        return False
    return True


def load_ground_truth(probes_dir: Path) -> dict[str, GroundTruth]:
    from src.protocol import ForgeryMethod, GroundTruth, Verdict

    gt_map: dict[str, GroundTruth] = {}
    for meta_path in sorted(probes_dir.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        image_name = meta_path.stem

        mask = None
        mask_path = probes_dir / f"{image_name}_mask.npy"
        if mask_path.exists():
            mask = np.load(str(mask_path))

        gt_map[image_name] = GroundTruth(
            verdict=Verdict(meta["verdict"]),
            method=ForgeryMethod(meta["method"]) if meta.get("method") else None,
            mask=mask,
        )
    return gt_map


def format_table(rows: list[list[str]], headers: list[str]) -> str:
    all_rows = [headers] + rows
    widths = [
        max(len(str(cell)) for cell in col) for col in zip(*all_rows)
    ]

    def fmt_row(row: list[str]) -> str:
        return " | ".join(str(cell).ljust(w) for cell, w in zip(row, widths))

    lines = [fmt_row(headers), "-+-".join("-" * w for w in widths)]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def main() -> None:
    if not check_data():
        sys.exit(1)

    from src.miner.detector import ForgeryDetector
    from src.protocol import Verdict
    from src.validator.scorer import ProbeScorer

    gt_map = load_ground_truth(PROBES_DIR)
    if not gt_map:
        print("No ground truth metadata (.json) found in", PROBES_DIR)
        sys.exit(1)

    detector = ForgeryDetector()
    scorer = ProbeScorer()

    table_rows: list[list[str]] = []
    probe_scores: list[float] = []
    latencies: list[float] = []

    for image_name, gt in gt_map.items():
        for ext in (".png", ".jpg"):
            image_path = PROBES_DIR / f"{image_name}{ext}"
            if image_path.exists():
                break
        else:
            print(f"  [skip] No image file for {image_name}")
            continue

        response = detector.detect_from_path(str(image_path), task_id=image_name)
        score = scorer.score_probe(response, gt)
        iou = scorer.compute_iou(response.mask, gt.mask)

        probe_scores.append(score)
        latencies.append(response.latency_ms)

        table_rows.append([
            image_name,
            gt.verdict.value,
            gt.method.value if gt.method else "-",
            response.verdict.value,
            f"{response.confidence:.2f}",
            f"{iou:.3f}",
            f"{score:.3f}",
        ])

    headers = ["Image", "GT Verdict", "GT Method", "Verdict", "Conf", "IoU", "Score"]
    print()
    print(format_table(table_rows, headers))
    print()

    avg_probe = sum(probe_scores) / len(probe_scores) if probe_scores else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    epoch_result = scorer.score_epoch(
        probe_scores=probe_scores,
        consensus_scores=[],
        avg_latency_ms=avg_latency,
        max_latency_ms=30_000.0,
        miner_id="demo-miner",
    )

    print(f"  Avg probe score:   {avg_probe:.4f}")
    print(f"  Avg latency:       {avg_latency:.0f} ms")
    print(f"  Latency score:     {epoch_result.latency_score:.4f}")
    print(f"  Total epoch score: {epoch_result.total_score:.4f}")
    print()


if __name__ == "__main__":
    main()
