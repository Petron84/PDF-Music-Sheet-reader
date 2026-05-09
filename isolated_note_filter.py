#!/usr/bin/env python3
"""
isolated_note_filter.py

Scans a folder of "isolated note" images and removes any image that does NOT
contain a valid notehead. Uses a three-stage pipeline calibrated on real samples:

  Stage 1 — Width gate:      images narrower than WIDTH_THRESH px are "too thin"
  Stage 2 — Erosion test:    images with no thick ink (erode r=4) have no notehead
  Stage 3 — Blob analysis:   checks the largest ink blob for notehead-like properties
               • minimum total area  (filters out dots, tiny arches)
               • blob height ratio   (stem+notehead spans ~40% of image height)
               • fill ratio          (stem makes it sparse; clefs/time-sigs are denser)

Usage:
    python isolated_note_filter.py --folder /path/to/images [options]

Options:
    --folder          Path to the image folder (required)
    --width-thresh    Width (px) below which image is "too thin" (default: 35)
    --erosion-radius  Erosion kernel radius in px (default: 4)
    --min-area        Min largest-blob area in px² (default: 2000)
    --min-h-ratio     Min blob_height/image_height ratio (default: 0.30)
    --max-fill        Max fill ratio of largest blob bbox (default: 0.45)
    --dry-run         Print what would be removed without deleting
    --log             Path to save a CSV report (default: <folder>/filter_report.csv)
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ─── CALIBRATED DEFAULTS (from sample analysis) ─────────────────────────────
WIDTH_THRESH    = 35     # px — images narrower than this are instantly flagged
EROSION_RADIUS  = 4      # px — erosion kernel radius; kills thin lines/stems
MIN_AREA        = 2000   # px² — minimum largest-blob area to be a note
MIN_H_RATIO     = 0.30   # — blob height must be ≥30% of image height
MAX_FILL        = 0.45   # — blob fill ratio must be <0.45 (clefs/time-sigs are denser)
SUPPORTED_EXTS  = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# ─────────────────────────────────────────────────────────────────────────────


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Grayscale + ensure dark ink on white background."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 128:          # dark background → invert
        gray = cv2.bitwise_not(gray)
    return gray


def classify(img_bgr: np.ndarray,
             width_thresh: int,
             erosion_radius: int,
             min_area: float,
             min_h_ratio: float,
             max_fill: float) -> tuple[str, str, dict]:
    """
    Returns (decision, reason, debug_info).
    decision: "KEEP" | "REMOVE"
    reason:   short string tag
    """
    h_img, w_img = img_bgr.shape[:2]
    debug = {"width": w_img, "height": h_img}

    # ── Stage 1: width gate ───────────────────────────────────────────────────
    if w_img < width_thresh:
        return "REMOVE", "too_thin", debug

    gray = preprocess(img_bgr)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── Stage 2: erosion test ─────────────────────────────────────────────────
    # Erode away thin lines; only "thick" ink (noteheads) survives
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erosion_radius * 2 + 1, erosion_radius * 2 + 1)
    )
    eroded = cv2.erode(binary, k, iterations=1)
    surviving_px = int(np.sum(eroded > 0))
    debug["surviving_px_after_erosion"] = surviving_px

    if surviving_px == 0:
        return "REMOVE", "no_thick_ink", debug

    # ── Stage 3: largest-blob characterisation ────────────────────────────────
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "REMOVE", "no_contours", debug

    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)
    x, y, bw, bh = cv2.boundingRect(largest)
    fill    = area / (bw * bh) if bw * bh > 0 else 0
    h_ratio = bh / h_img

    debug.update({
        "largest_blob_area":     int(area),
        "largest_blob_bbox":     f"{bw}x{bh}",
        "fill_ratio":            round(fill, 3),
        "height_ratio":          round(h_ratio, 3),
    })

    # A valid notehead+stem blob must be large enough, tall enough, and sparse enough
    if area < min_area:
        return "REMOVE", f"blob_too_small", debug
    if h_ratio < min_h_ratio:
        return "REMOVE", f"blob_too_short", debug
    if fill > max_fill:
        return "REMOVE", f"blob_too_dense", debug

    return "KEEP", "notehead_detected", debug


def scan_folder(
    folder:        Path,
    width_thresh:  int,
    erosion_radius:int,
    min_area:      float,
    min_h_ratio:   float,
    max_fill:      float,
    dry_run:       bool,
    log_path:      Path,
) -> None:
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )

    if not images:
        print(f"No supported images found in: {folder}")
        sys.exit(0)

    print(f"\nScanning {len(images)} images in: {folder}")
    print(f"  Width threshold  : < {width_thresh} px")
    print(f"  Erosion radius   : {erosion_radius} px")
    print(f"  Min blob area    : {min_area} px²")
    print(f"  Min height ratio : {min_h_ratio}")
    print(f"  Max fill ratio   : {max_fill}")
    print(f"  Dry-run          : {dry_run}\n")

    rows      = []
    n_removed = 0
    n_kept    = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ?  {img_path.name}  (could not read — skipped)")
            continue

        decision, reason, debug = classify(
            img, width_thresh, erosion_radius, min_area, min_h_ratio, max_fill
        )

        marker = "✓" if decision == "KEEP" else "✗"
        tag    = f"  [{reason}]" if decision == "REMOVE" else ""
        print(f"  {marker}  {img_path.name}{tag}")

        rows.append({
            "file":     img_path.name,
            "decision": decision,
            "reason":   reason,
            **debug,
        })

        if decision == "REMOVE":
            n_removed += 1
            if not dry_run:
                img_path.unlink()
        else:
            n_kept += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"Total scanned : {len(rows)}")
    print(f"Kept          : {n_kept}")
    print(f"Removed       : {n_removed}")
    if dry_run:
        print("(Dry-run: no files were deleted)")

    # ── CSV report ────────────────────────────────────────────────────────────
    fieldnames = ["file", "decision", "reason", "width", "height",
                  "surviving_px_after_erosion", "largest_blob_area",
                  "largest_blob_bbox", "fill_ratio", "height_ratio"]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nReport saved to: {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove isolated-note images that lack a valid notehead."
    )
    parser.add_argument("--folder",           required=True)
    parser.add_argument("--width-thresh",     type=int,   default=WIDTH_THRESH,
                        help=f"Images narrower than this (px) are removed (default: {WIDTH_THRESH})")
    parser.add_argument("--erosion-radius",   type=int,   default=EROSION_RADIUS,
                        help=f"Erosion kernel radius in px (default: {EROSION_RADIUS})")
    parser.add_argument("--min-area",         type=float, default=MIN_AREA,
                        help=f"Min largest-blob area px² (default: {MIN_AREA})")
    parser.add_argument("--min-h-ratio",      type=float, default=MIN_H_RATIO,
                        help=f"Min blob_h/img_h ratio (default: {MIN_H_RATIO})")
    parser.add_argument("--max-fill",         type=float, default=MAX_FILL,
                        help=f"Max blob fill ratio (default: {MAX_FILL})")
    parser.add_argument("--dry-run",          action="store_true",
                        help="Simulate without deleting files")
    parser.add_argument("--log",              default=None,
                        help="CSV report path (default: <folder>/filter_report.csv)")

    args   = parser.parse_args()
    folder = Path(args.folder).expanduser().resolve()

    if not folder.is_dir():
        print(f"Error: '{folder}' is not a valid directory.")
        sys.exit(1)

    log_path = Path(args.log) if args.log else folder / "filter_report.csv"

    scan_folder(
        folder         = folder,
        width_thresh   = args.width_thresh,
        erosion_radius = args.erosion_radius,
        min_area       = args.min_area,
        min_h_ratio    = args.min_h_ratio,
        max_fill       = args.max_fill,
        dry_run        = args.dry_run,
        log_path       = log_path,
    )


if __name__ == "__main__":
    main()