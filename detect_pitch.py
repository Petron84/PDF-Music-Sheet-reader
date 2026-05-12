#!/usr/bin/env python3
"""
detect_pitch.py

Detects the musical pitch of an isolated note image.

The image must follow the naming convention:
    <staff_idx>_<note_idx>_<staff_type>_<song_name>.ext
    e.g.  1_3_bass_2png(2)_v14.png  →  bass clef

Core algorithm:
    1. Detect staff lines from the image (anchored on the topmost line,
       which is always unobscured by the notehead).
    2. Remove staff lines from the binary image to isolate note blobs.
    3. Detect the notehead centre y-coordinate:
         - For FILLED noteheads (quarter/8th): centroid of the largest round blob.
         - For HOLLOW noteheads (half/whole): the staff line that crosses the
           notehead splits it in two; average the centroids of both halves.
    4. Map the y-coordinate to a pitch slot index using the known staff spacing,
       then look up the pitch name from the clef's slot table.

Usage (as a library):
    from detect_pitch import get_pitch
    pitch = get_pitch("1_3_bass_<song_name>.png")   # → "F3"

Usage (as a script):
    python detect_pitch.py path/to/image.png [path/to/image2.png ...]
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# ─── STAFF LINE POSITIONS (canonical, in pixels) ─────────────────────────────
# These are the y-pixel positions of the 5 staff lines (top to bottom in image,
# i.e. line 5 first → line 1 last in music notation) for images of this dataset.
# Adjust if your images use a different resolution or layout.
TREBLE_TOP_LINE_Y = 72   # y of line 5 (top staff line, highest pitch)
BASS_TOP_LINE_Y   = 171  # y of line 5 (top staff line, highest pitch)
STAFF_SPACING     = 43   # pixels between adjacent staff lines (inter-line gap)

# ─── PITCH SLOT TABLES ───────────────────────────────────────────────────────
# Slot index: each integer = one diatonic step.
# Slot 9 = top staff line (line 5), slot 1 = bottom staff line (line 1).
# Even slots = spaces, odd slots = lines.
# Slots < 1 and > 9 = ledger line territory.

TREBLE_SLOTS: dict[int, str] = {
    -3: "A3",
    -2: "B3",
    -1: "C4",   # ledger line below staff (middle C)
     0: "D4",   # space below staff
     1: "E4",   # line 1 (bottom)
     2: "F4",
     3: "G4",   # line 2
     4: "A4",
     5: "B4",   # line 3 (middle)
     6: "C5",
     7: "D5",   # line 4
     8: "E5",
     9: "F5",   # line 5 (top)
    10: "G5",
    11: "A5",
}

BASS_SLOTS: dict[int, str] = {
    -2: "D2",
    -1: "E2",
     0: "F2",   # space below staff
     1: "G2",   # line 1 (bottom)
     2: "A2",
     3: "B2",   # line 2
     4: "C3",
     5: "D3",   # line 3 (middle)
     6: "E3",
     7: "F3",   # line 4
     8: "G3",
     9: "A3",   # line 5 (top)
    10: "B3",
    11: "C4",
}

# ─── INTERNALS ───────────────────────────────────────────────────────────────

def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Grayscale + ensure dark ink on white background."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 128:
        gray = cv2.bitwise_not(gray)
    return gray


def _get_staff_lines(gray: np.ndarray, img_w: int, clef: str) -> tuple[np.ndarray, list[int]]:
    """
    Detect 5 staff line y-positions.
    Anchors on the topmost detected staff line (always unobscured) and uses
    the inter-line spacing estimated from detected lines. Falls back to
    canonical pixel positions if detection is unreliable.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_sums  = np.sum(binary > 0, axis=1)

    # Rows with ink spanning ≥45% of image width → likely staff lines
    staff_rows = np.where(row_sums > img_w * 0.45)[0]

    canon_top = TREBLE_TOP_LINE_Y if clef == "treble" else BASS_TOP_LINE_Y

    if len(staff_rows) == 0:
        lines5 = [canon_top + STAFF_SPACING * i for i in range(5)]
        return binary, lines5

    # Cluster adjacent rows into single lines
    clusters, current = [], [staff_rows[0]]
    for r in staff_rows[1:]:
        if r - current[-1] <= 4:
            current.append(r)
        else:
            clusters.append(int(np.mean(current)))
            current = [r]
    clusters.append(int(np.mean(current)))

    # Estimate spacing from detected inter-line gaps
    if len(clusters) >= 2:
        gaps = [clusters[i + 1] - clusters[i] for i in range(len(clusters) - 1)]
        valid = [g for g in gaps if 30 <= g <= 55]
        spacing = int(np.median(valid)) if valid else STAFF_SPACING
    else:
        spacing = STAFF_SPACING

    # Use topmost detected line as anchor; fall back to canonical if too far off
    top_line = min(clusters)
    if abs(top_line - canon_top) > 10:
        top_line = canon_top

    lines5 = [top_line + spacing * i for i in range(5)]
    return binary, lines5


def _remove_staff_lines(binary: np.ndarray, lines5: list[int], thickness: int = 10) -> np.ndarray:
    """Zero out pixels around each staff line y-position."""
    clean = binary.copy()
    for ly in lines5:
        y0 = max(0, ly - thickness // 2)
        y1 = min(binary.shape[0], ly + thickness // 2 + 1)
        clean[y0:y1, :] = 0
    return clean


def _find_notehead_y(binary: np.ndarray, lines5: list[int],
                     img_h: int, img_w: int) -> float | None:
    """
    Return the y-coordinate of the notehead centre.

    After removing staff lines:
      - Filled noteheads (quarter / 8th) appear as a single elliptical blob.
      - Hollow noteheads (half / whole) are split into two arcs by the staff
        line that was removed; the algorithm detects both halves and averages
        their centroids.
    """
    THICKNESS = 10
    clean = _remove_staff_lines(binary, lines5, thickness=THICKNESS)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # ── Collect candidates (exclude pure stems and residual staff lines) ──────
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 40:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / bh if bh > 0 else 0
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cy = M["m01"] / M["m00"]

        # Reject thin vertical stems
        if aspect < 0.25 and bh > img_h * 0.1:
            continue
        # Reject near-full-width horizontal slivers (residual staff lines)
        if bw >= img_w * 0.85 and bh < 12:
            continue

        candidates.append({
            "area": area, "x": x, "y": y,
            "w": bw, "h": bh, "aspect": aspect, "cy": cy,
        })

    if not candidates:
        return None

    candidates.sort(key=lambda b: b["area"], reverse=True)

    if len(candidates) == 1:
        return candidates[0]["cy"]

    # ── Check for split hollow notehead (two adjacent wide blobs) ────────────
    b0, b1   = candidates[0], candidates[1]
    top_b    = b0 if b0["cy"] < b1["cy"] else b1
    bot_b    = b1 if b0["cy"] < b1["cy"] else b0

    vert_gap     = bot_b["y"] - (top_b["y"] + top_b["h"])
    both_wide    = top_b["aspect"] > 0.8 and bot_b["aspect"] > 0.8
    area_similar = min(b0["area"], b1["area"]) / max(b0["area"], b1["area"]) > 0.3
    adjacent     = vert_gap <= THICKNESS * 2 + 5

    if both_wide and area_similar and adjacent:
        # Hollow notehead: weighted average of both halves
        total = b0["area"] + b1["area"]
        return (b0["cy"] * b0["area"] + b1["cy"] * b1["area"]) / total

    # Filled notehead: largest blob centroid
    return candidates[0]["cy"]


def _y_to_pitch(notehead_y: float, lines5: list[int],
                slot_map: dict[int, str]) -> str:
    """
    Convert a notehead y-pixel position to a pitch string.

    `lines5` runs top-to-bottom in pixel space (lines5[0] = top staff line = line 5
    = highest pitch).  Half the inter-line spacing equals one diatonic step.
    """
    spacings   = [lines5[i + 1] - lines5[i] for i in range(len(lines5) - 1)]
    half_space = np.mean(spacings) / 2   # one diatonic step in pixels

    # Use the middle staff line as a stable reference
    ref_y    = lines5[2]  # line 3 (middle)
    ref_slot = 5          # slot 5 = line 3

    delta_y    = notehead_y - ref_y
    delta_slot = -delta_y / half_space   # y↑ = pitch↑ = slot↑
    slot_float = ref_slot + delta_slot
    slot_int   = round(slot_float)

    return slot_map.get(slot_int, f"unknown(slot={slot_int})")


# ─── PUBLIC API ──────────────────────────────────────────────────────────────

def get_pitch(image_path: str) -> str:
    """
    Detect the pitch of a note in an isolated-note image.

    Parameters
    ----------
    image_path : str | Path
        Path to the image file.  The filename must contain either "treble" or
        "bass" to indicate the clef (e.g. "1_3_bass_2png(2)_v14.png").

    Returns
    -------
    str
        Pitch name such as "F3", "A4", "C5", etc.
        Returns "unknown(slot=N)" if the note falls outside the slot table.
        Raises ValueError on unreadable images or failed detection.
    """
    path = Path(image_path)
    img  = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    # ── Parse clef from filename ──────────────────────────────────────────────
    name_lower = path.name.lower()
    if "treble" in name_lower:
        clef     = "treble"
        slot_map = TREBLE_SLOTS
    elif "bass" in name_lower:
        clef     = "bass"
        slot_map = BASS_SLOTS
    else:
        raise ValueError(
            f"Cannot determine clef from filename '{path.name}'. "
            "Filename must contain 'treble' or 'bass'."
        )

    h, w = img.shape[:2]
    gray = _preprocess(img)
    binary, lines5 = _get_staff_lines(gray, w, clef)
    notehead_y     = _find_notehead_y(binary, lines5, h, w)

    if notehead_y is None:
        raise ValueError(f"No notehead detected in '{path.name}'.")

    return _y_to_pitch(notehead_y, lines5, slot_map)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python detect_pitch.py <image> [image2 ...]")
        sys.exit(1)

    for arg in sys.argv[1:]:
        try:
            pitch = get_pitch(arg)
        except:
            print(f"{arg}: ERROR - {sys.exc_info()[1]}")