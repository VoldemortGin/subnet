"""
Real forgery detection demo using CASIA 1.0 ground truth dataset.

Runs ELA (built-in) and optionally ManTraNet on real tampered images,
compares against ground truth masks, and saves side-by-side visualizations.

Usage: python -m demo.run_real_demo
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data" / "real_samples"
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"
OUTPUT_DIR = ROOT_DIR / "data" / "demo_output"

# Also support CASIA combine images directly if real_samples not extracted
CASIA_DIR = ROOT_DIR / "data" / "casia1gt"
CASIA_SAMPLES_DIRS = [CASIA_DIR / "Samples" / d for d in ("CM", "Sp")]
CASIA_GT_DIRS = [CASIA_DIR / "CASIA 1.0 groundtruth" / d for d in ("CM", "Sp")]


def discover_images() -> list[tuple[Path, Path]]:
    """Find image-mask pairs from extracted real_samples or CASIA combine images."""
    pairs: list[tuple[Path, Path]] = []

    # Try extracted images first
    if IMAGES_DIR.exists() and MASKS_DIR.exists():
        for img_path in sorted(IMAGES_DIR.glob("*.png")):
            mask_path = MASKS_DIR / img_path.name
            if mask_path.exists():
                pairs.append((img_path, mask_path))

    if pairs:
        return pairs

    # Fallback: extract from CASIA combine images on-the-fly
    for samples_dir, gt_dir in zip(CASIA_SAMPLES_DIRS, CASIA_GT_DIRS):
        if not samples_dir.exists():
            continue
        for combine_path in sorted(samples_dir.glob("*_combine.png")):
            stem = combine_path.stem.replace("_combine", "")
            gt_path = gt_dir / f"{stem}_gt.png"
            if gt_path.exists():
                pairs.append((combine_path, gt_path))

    return pairs


def load_image_and_mask(
    img_path: Path, mask_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Load an image and its GT mask, handling CASIA combine format."""
    if "_combine" in img_path.stem:
        # Extract tampered panel (middle third) from combine image
        combine = cv2.imread(str(img_path))
        w = combine.shape[1] // 3
        image = combine[:, w : 2 * w, :]
    else:
        image = cv2.imread(str(img_path))

    gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Cannot read image: {img_path}")
    if gt_mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")

    # Ensure mask is binary
    _, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)

    return image, gt_mask


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute pixel-level accuracy between two binary masks."""
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    correct = (pred.astype(bool) == gt.astype(bool)).sum()
    return float(correct / gt.size)


def save_visualization(
    name: str,
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    backend_name: str,
    output_dir: Path,
) -> Path:
    """Save a side-by-side visualization: original | detection mask | ground truth."""
    h, w = image.shape[:2]

    # Resize masks to match image dimensions if needed
    if pred_mask.shape[:2] != (h, w):
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if gt_mask.shape[:2] != (h, w):
        gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Convert masks to 3-channel for visualization
    pred_color = cv2.applyColorMap(pred_mask, cv2.COLORMAP_JET)
    gt_color = np.zeros_like(image)
    gt_color[:, :, 2] = gt_mask  # Red channel for GT

    # Create labeled panels
    label_h = 30
    panel_h = h + label_h

    def make_panel(img: np.ndarray, label: str) -> np.ndarray:
        panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
        panel[label_h:, :, :] = img
        cv2.putText(
            panel, label, (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        return panel

    p1 = make_panel(image, "Tampered Image")
    p2 = make_panel(pred_color, f"Detection ({backend_name})")
    p3 = make_panel(gt_color, "Ground Truth")

    combined = cv2.hconcat([p1, p2, p3])
    output_path = output_dir / f"{name}_{backend_name}.png"
    cv2.imwrite(str(output_path), combined)
    return output_path


def format_table(rows: list[list[str]], headers: list[str]) -> str:
    """Format a simple ASCII table."""
    all_rows = [headers] + rows
    widths = [max(len(str(cell)) for cell in col) for col in zip(*all_rows)]

    def fmt_row(row: list[str]) -> str:
        return " | ".join(str(cell).ljust(w) for cell, w in zip(row, widths))

    lines = [fmt_row(headers), "-+-".join("-" * w for w in widths)]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def run_ela_detection(image: np.ndarray, task_id: str):
    """Run the built-in ELA detector."""
    from src.miner.detector import ForgeryDetector

    detector = ForgeryDetector()
    return detector.detect(image, task_id=task_id)


def run_mantranet_detection(image: np.ndarray, task_id: str):
    """Run ManTraNet backend if available."""
    from src.miner.backends.mantranet import ManTraNetBackend
    from src.miner.detector import ForgeryDetector

    if not ManTraNetBackend.is_available():
        return None

    backend = ManTraNetBackend()
    detector = ForgeryDetector(backend=backend)
    return detector.detect(image, task_id=task_id)


def main() -> None:
    pairs = discover_images()
    if not pairs:
        print("=" * 60)
        print("  No real sample images found.")
        print()
        print("  Expected data in:")
        print(f"    {IMAGES_DIR}")
        print(f"    {MASKS_DIR}")
        print("  Or CASIA combine images in:")
        print(f"    {CASIA_DIR}")
        print()
        print("  Run: git clone https://github.com/namtpham/casia1groundtruth.git data/casia1gt")
        print("=" * 60)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Limit to first 10 images for a quick demo
    max_images = 10
    pairs = pairs[:max_images]

    print(f"\nFound {len(pairs)} image-mask pairs (showing up to {max_images})")
    print()

    # Check ManTraNet availability
    mantranet_available = False
    try:
        from src.miner.backends.mantranet import ManTraNetBackend
        mantranet_available = ManTraNetBackend.is_available()
    except ImportError:
        pass

    backends = ["ELA"]
    if mantranet_available:
        backends.append("ManTraNet")
    print(f"Backends: {', '.join(backends)}")
    print()

    # Results storage
    ela_rows: list[list[str]] = []
    mantra_rows: list[list[str]] = []
    ela_ious: list[float] = []
    ela_accs: list[float] = []
    mantra_ious: list[float] = []
    mantra_accs: list[float] = []

    for img_path, mask_path in pairs:
        name = img_path.stem.replace("_combine", "")
        short_name = name if len(name) <= 40 else name[:37] + "..."

        image, gt_mask = load_image_and_mask(img_path, mask_path)

        # ---- ELA ----
        t0 = time.perf_counter()
        ela_result = run_ela_detection(image, task_id=name)
        ela_ms = (time.perf_counter() - t0) * 1000

        ela_pred = ela_result.mask if ela_result.mask is not None else np.zeros_like(gt_mask)
        if ela_pred.shape != gt_mask.shape:
            ela_pred = cv2.resize(ela_pred, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        ela_iou = compute_iou(ela_pred, gt_mask)
        ela_acc = compute_pixel_accuracy(ela_pred, gt_mask)
        ela_ious.append(ela_iou)
        ela_accs.append(ela_acc)

        ela_rows.append([
            short_name,
            ela_result.verdict.value,
            f"{ela_result.confidence:.3f}",
            f"{ela_iou:.3f}",
            f"{ela_acc:.3f}",
            f"{ela_ms:.0f}ms",
        ])

        save_visualization(name, image, ela_pred, gt_mask, "ELA", OUTPUT_DIR)

        # ---- ManTraNet ----
        if mantranet_available:
            t0 = time.perf_counter()
            mantra_result = run_mantranet_detection(image, task_id=name)
            mantra_ms = (time.perf_counter() - t0) * 1000

            if mantra_result is not None:
                mantra_pred = mantra_result.mask if mantra_result.mask is not None else np.zeros_like(gt_mask)
                if mantra_pred.shape != gt_mask.shape:
                    mantra_pred = cv2.resize(
                        mantra_pred, (gt_mask.shape[1], gt_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                mantra_iou = compute_iou(mantra_pred, gt_mask)
                mantra_acc = compute_pixel_accuracy(mantra_pred, gt_mask)
                mantra_ious.append(mantra_iou)
                mantra_accs.append(mantra_acc)

                mantra_rows.append([
                    short_name,
                    mantra_result.verdict.value,
                    f"{mantra_result.confidence:.3f}",
                    f"{mantra_iou:.3f}",
                    f"{mantra_acc:.3f}",
                    f"{mantra_ms:.0f}ms",
                ])

                save_visualization(name, image, mantra_pred, gt_mask, "ManTraNet", OUTPUT_DIR)

    # Print results
    headers = ["Image", "Verdict", "Conf", "IoU", "Acc", "Time"]

    print("=" * 70)
    print("  ELA (Error Level Analysis) Results")
    print("=" * 70)
    print(format_table(ela_rows, headers))
    if ela_ious:
        print(f"\n  Mean IoU:      {sum(ela_ious) / len(ela_ious):.4f}")
        print(f"  Mean Accuracy: {sum(ela_accs) / len(ela_accs):.4f}")
    print()

    if mantra_rows:
        print("=" * 70)
        print("  ManTraNet Results")
        print("=" * 70)
        print(format_table(mantra_rows, headers))
        if mantra_ious:
            print(f"\n  Mean IoU:      {sum(mantra_ious) / len(mantra_ious):.4f}")
            print(f"  Mean Accuracy: {sum(mantra_accs) / len(mantra_accs):.4f}")
        print()

    print(f"Visualizations saved to: {OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()
