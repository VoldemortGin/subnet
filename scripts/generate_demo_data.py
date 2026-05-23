"""Generate demo data for the HARM project presentation.

Creates tampered images using ForgeEngine, synthetic damage overlays,
and a metadata.json file cataloging all demo images.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.protocol import ForgeryMethod
from src.validator.forge import ForgeEngine

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "data" / "demo"
REAL_DIR = DEMO_DIR / "real"
TAMPERED_DIR = DEMO_DIR / "tampered"
AI_GEN_DIR = DEMO_DIR / "ai_generated"

PRODUCT_TYPES = {
    "smartphone": "smartphone",
    "watch": "watch",
    "headphones": "headphones",
    "shoe": "shoe",
    "camera": "camera",
    "sunglasses": "sunglasses",
    "perfume": "perfume",
    "sneaker": "sneaker",
    "bag": "bag",
    "tshirt": "t-shirt",
    "laptop": "laptop",
    "earbuds": "earbuds",
    "keyboard": "keyboard",
    "backpack": "backpack",
    "bottle": "water bottle",
}


def validate_real_images() -> list[Path]:
    """Return list of valid real images (readable by OpenCV)."""
    valid = []
    for p in sorted(REAL_DIR.glob("*.jpg")):
        img = cv2.imread(str(p))
        if img is not None and img.size > 0:
            valid.append(p)
            print(f"  [OK] {p.name} ({img.shape[1]}x{img.shape[0]})")
        else:
            print(f"  [SKIP] {p.name} - cannot read")
    return valid


def generate_tampered(real_images: list[Path]) -> list[dict]:
    """Use ForgeEngine to create tampered versions of real images."""
    engine = ForgeEngine(seed=42)
    records = []

    # Use up to 10 images for tampering, use pairs as donors for splicing
    tamper_images = real_images[:10]

    for i, img_path in enumerate(tamper_images):
        # Pick a method; cycle through available methods for diversity
        methods = [
            ForgeryMethod.COPY_MOVE,
            ForgeryMethod.SPLICING,
            ForgeryMethod.COMPRESSION,
            ForgeryMethod.INPAINTING,
        ]
        method = methods[i % len(methods)]

        # For splicing, provide a donor image from a different product
        donor = None
        if method == ForgeryMethod.SPLICING:
            donor_path = tamper_images[(i + 3) % len(tamper_images)]
            donor = cv2.imread(str(donor_path))

        try:
            probe = engine.generate_probe(
                clean_image_path=img_path,
                output_dir=TAMPERED_DIR,
                method=method,
            )
            print(f"  [OK] {img_path.stem} -> {method.value}")

            # Re-run forge with donor if splicing (generate_probe doesn't pass donor)
            if method == ForgeryMethod.SPLICING and donor is not None:
                img = cv2.imread(str(img_path))
                tampered, mask, used_method = engine.forge(img, method=method, donor_image=donor)
                tampered_path = TAMPERED_DIR / f"{img_path.stem}_tampered.png"
                mask_path = TAMPERED_DIR / f"{img_path.stem}_mask.png"
                cv2.imwrite(str(tampered_path), tampered)
                cv2.imwrite(str(mask_path), mask)

            stem = img_path.stem
            product_type = PRODUCT_TYPES.get(stem, "product")
            records.append({
                "filename": f"{stem}_tampered.png",
                "category": "tampered",
                "method": method.value,
                "original": img_path.name,
                "product_type": product_type,
                "description": f"Tampered via {method.value} - {_method_description(method)}",
                "mask_file": f"{stem}_mask.png",
            })
        except Exception as e:
            print(f"  [FAIL] {img_path.stem}: {e}")

    return records


def _method_description(method: ForgeryMethod) -> str:
    descriptions = {
        ForgeryMethod.COPY_MOVE: "A region was duplicated and moved within the image",
        ForgeryMethod.SPLICING: "A region from another image was spliced in",
        ForgeryMethod.COMPRESSION: "A region has mismatched JPEG compression artifacts",
        ForgeryMethod.INPAINTING: "Noise was injected into a region to simulate inpainting",
    }
    return descriptions.get(method, "Unknown tampering method")


def generate_ai_damage(real_images: list[Path]) -> list[dict]:
    """Create synthetic 'damaged product' images simulating AI-generated fakes."""
    rng = np.random.default_rng(123)
    records = []

    # Pick 8 images for damage simulation
    damage_images = real_images[:8]

    damage_types = [
        "cracks",
        "scratches",
        "stain",
        "missing_part",
        "cracks",
        "scratches",
        "stain",
        "missing_part",
    ]

    for i, img_path in enumerate(damage_images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        damage_type = damage_types[i]
        result = img.copy()
        h, w = result.shape[:2]

        if damage_type == "cracks":
            result = _add_cracks(result, rng, h, w)
        elif damage_type == "scratches":
            result = _add_scratches(result, rng, h, w)
        elif damage_type == "stain":
            result = _add_stains(result, rng, h, w)
        elif damage_type == "missing_part":
            result = _add_missing_part(result, rng, h, w)

        out_name = f"{img_path.stem}_damaged_{damage_type}.png"
        out_path = AI_GEN_DIR / out_name
        cv2.imwrite(str(out_path), result)
        print(f"  [OK] {out_name} ({damage_type})")

        product_type = PRODUCT_TYPES.get(img_path.stem, "product")
        records.append({
            "filename": out_name,
            "category": "ai_generated",
            "damage_type": damage_type,
            "original": img_path.name,
            "product_type": product_type,
            "description": f"Synthetic AI-generated damage ({damage_type}) - simulates fraudulent refund claim",
        })

    return records


def _add_cracks(img: np.ndarray, rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Draw jagged crack lines across the image."""
    for _ in range(rng.integers(3, 7)):
        # Start point in roughly the center area
        start_x = int(rng.integers(w // 4, 3 * w // 4))
        start_y = int(rng.integers(h // 4, 3 * h // 4))

        points = [(start_x, start_y)]
        num_segments = int(rng.integers(8, 20))
        for _ in range(num_segments):
            dx = int(rng.integers(-25, 26))
            dy = int(rng.integers(-25, 26))
            last = points[-1]
            new_x = max(0, min(w - 1, last[0] + dx))
            new_y = max(0, min(h - 1, last[1] + dy))
            points.append((new_x, new_y))

        # Draw the crack as connected line segments
        pts = np.array(points, dtype=np.int32)
        thickness = int(rng.integers(1, 3))
        # Dark color for crack
        color = (int(rng.integers(20, 60)), int(rng.integers(20, 60)), int(rng.integers(20, 60)))
        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)

        # Add slight branching
        if len(points) > 5:
            branch_idx = int(rng.integers(2, len(points) - 2))
            branch_start = points[branch_idx]
            branch_pts = [branch_start]
            for _ in range(int(rng.integers(3, 8))):
                dx = int(rng.integers(-15, 16))
                dy = int(rng.integers(-15, 16))
                last = branch_pts[-1]
                new_x = max(0, min(w - 1, last[0] + dx))
                new_y = max(0, min(h - 1, last[1] + dy))
                branch_pts.append((new_x, new_y))
            bp = np.array(branch_pts, dtype=np.int32)
            cv2.polylines(img, [bp], isClosed=False, color=color, thickness=max(1, thickness - 1))

    return img


def _add_scratches(img: np.ndarray, rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Draw thin curved scratch lines."""
    for _ in range(rng.integers(4, 10)):
        start_x = int(rng.integers(0, w))
        start_y = int(rng.integers(0, h))
        length = int(rng.integers(50, min(200, min(h, w) // 2)))
        angle = rng.uniform(0, 2 * np.pi)

        points = [(start_x, start_y)]
        for j in range(length // 5):
            angle += rng.uniform(-0.3, 0.3)  # slight curve
            dx = int(5 * np.cos(angle))
            dy = int(5 * np.sin(angle))
            last = points[-1]
            new_x = max(0, min(w - 1, last[0] + dx))
            new_y = max(0, min(h - 1, last[1] + dy))
            points.append((new_x, new_y))

        pts = np.array(points, dtype=np.int32)
        # Light whitish scratch color
        brightness = int(rng.integers(180, 240))
        color = (brightness, brightness, brightness)
        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=1)

    return img


def _add_stains(img: np.ndarray, rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Add dark blob stains with gaussian blur."""
    num_stains = int(rng.integers(2, 5))
    for _ in range(num_stains):
        cx = int(rng.integers(w // 6, 5 * w // 6))
        cy = int(rng.integers(h // 6, 5 * h // 6))
        radius = int(rng.integers(15, min(60, min(h, w) // 6)))

        # Create a stain mask
        stain_mask = np.zeros((h, w), dtype=np.uint8)
        # Draw irregular blob using multiple overlapping circles
        for _ in range(int(rng.integers(3, 8))):
            offset_x = int(rng.integers(-radius // 2, radius // 2 + 1))
            offset_y = int(rng.integers(-radius // 2, radius // 2 + 1))
            r = int(rng.integers(radius // 3, radius))
            cv2.circle(stain_mask, (cx + offset_x, cy + offset_y), r, 255, -1)

        # Blur the mask for soft edges
        ksize = radius * 2 + 1
        if ksize % 2 == 0:
            ksize += 1
        stain_mask = cv2.GaussianBlur(stain_mask, (ksize, ksize), 0)

        # Dark brownish stain color
        stain_color = np.array([
            int(rng.integers(20, 60)),
            int(rng.integers(30, 70)),
            int(rng.integers(40, 80)),
        ])

        alpha = stain_mask.astype(np.float32) / 255.0 * 0.6
        for c in range(3):
            img[:, :, c] = (
                img[:, :, c].astype(np.float32) * (1 - alpha)
                + stain_color[c] * alpha
            ).astype(np.uint8)

    return img


def _add_missing_part(img: np.ndarray, rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Black out an irregular region to simulate missing/broken part."""
    # Create an irregular polygon
    cx = int(rng.integers(w // 4, 3 * w // 4))
    cy = int(rng.integers(h // 4, 3 * h // 4))
    num_vertices = int(rng.integers(5, 10))
    max_radius = min(h, w) // 5

    angles = np.sort(rng.uniform(0, 2 * np.pi, size=num_vertices))
    points = []
    for angle in angles:
        r = rng.uniform(max_radius * 0.4, max_radius)
        px = int(cx + r * np.cos(angle))
        py = int(cy + r * np.sin(angle))
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        points.append([px, py])

    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    # Fill with very dark color (simulating missing/broken part)
    dark_color = (
        int(rng.integers(5, 25)),
        int(rng.integers(5, 25)),
        int(rng.integers(5, 25)),
    )
    cv2.fillPoly(img, [pts], dark_color)

    # Add slight edge roughness
    edge_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(edge_mask, [pts], isClosed=True, color=255, thickness=3)
    noise = rng.integers(0, 50, size=(h, w), dtype=np.uint8)
    edge_noise = cv2.bitwise_and(noise, noise, mask=edge_mask)
    img = cv2.add(img, cv2.cvtColor(edge_noise, cv2.COLOR_GRAY2BGR))

    return img


def build_metadata(real_images: list[Path], tampered_records: list[dict], ai_records: list[dict]) -> None:
    """Build and write metadata.json."""
    images = []
    counter = 1

    # Real images
    for p in real_images:
        stem = p.stem
        product_type = PRODUCT_TYPES.get(stem, "product")
        images.append({
            "id": f"demo_{counter:03d}",
            "filename": p.name,
            "category": "real",
            "product_type": product_type,
            "description": "Clean product photo, no tampering",
        })
        counter += 1

    # Tampered images
    for rec in tampered_records:
        images.append({
            "id": f"demo_{counter:03d}",
            "filename": rec["filename"],
            "category": rec["category"],
            "method": rec["method"],
            "original": rec["original"],
            "product_type": rec["product_type"],
            "description": rec["description"],
            "mask_file": rec["mask_file"],
        })
        counter += 1

    # AI-generated images
    for rec in ai_records:
        images.append({
            "id": f"demo_{counter:03d}",
            "filename": rec["filename"],
            "category": rec["category"],
            "damage_type": rec["damage_type"],
            "original": rec["original"],
            "product_type": rec["product_type"],
            "description": rec["description"],
        })
        counter += 1

    metadata = {"images": images, "total_count": len(images)}
    meta_path = DEMO_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nMetadata written to {meta_path} ({len(images)} images)")


def main() -> None:
    print("=== Validating real images ===")
    real_images = validate_real_images()
    print(f"Found {len(real_images)} valid real images\n")

    print("=== Generating tampered images (ForgeEngine) ===")
    tampered_records = generate_tampered(real_images)
    print(f"Generated {len(tampered_records)} tampered images\n")

    print("=== Generating AI-damage images ===")
    ai_records = generate_ai_damage(real_images)
    print(f"Generated {len(ai_records)} AI-damage images\n")

    print("=== Building metadata.json ===")
    build_metadata(real_images, tampered_records, ai_records)

    # Summary
    real_count = len(real_images)
    tampered_count = len(tampered_records)
    ai_count = len(ai_records)
    total = real_count + tampered_count + ai_count
    print(f"\n=== SUMMARY ===")
    print(f"  Real:       {real_count}")
    print(f"  Tampered:   {tampered_count} (+ {tampered_count} masks)")
    print(f"  AI-damage:  {ai_count}")
    print(f"  Total:      {total}")


if __name__ == "__main__":
    main()
