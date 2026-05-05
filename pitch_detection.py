import cv2
from matplotlib import lines
import numpy as np
import json
import torch


def detect_noteheads(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 1) Threshold: black music symbols -> white foreground
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2) Detect staff lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    staff_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horizontal_kernel)

    # 3) Estimate line y-positions from row sums
    row_sum = np.sum(staff_lines > 0, axis=1)
    staff_y = np.where(row_sum > 0.5 * row_sum.max())[0]

    # compress nearby rows into single line centers
    lines = []
    if len(staff_y) > 0:
        group = [staff_y[0]]
        for y in staff_y[1:]:
            if y - group[-1] <= 2:
                group.append(y)
            else:
                lines.append(int(np.mean(group)))
                group = [y]
        lines.append(int(np.mean(group)))

    print("staff lines:", lines)

    # 4) Remove staff lines from binary image
    notes_only = cv2.subtract(bw, staff_lines)


    # 5) Clean up notehead blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    notes_only = cv2.morphologyEx(notes_only, cv2.MORPH_OPEN, kernel)
    notes_only = cv2.morphologyEx(notes_only, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("notes_only.png", notes_only)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 6) Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(notes_only, connectivity=8)

    best_blob = None
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        # heuristic filter for notehead-like blobs
        aspect = w / float(h)
        if 50 < area < 2000 and 0.5 < aspect < 2.0:
            cx, cy = centroids[i]
            best_blob = (int(cx), int(cy), x, y, w, h, area)

    if best_blob:
        cx, cy, x, y, w, h, area = best_blob
        print("notehead center:", (cx, cy))

        # find nearest staff position
        if len(lines) >= 5:
            positions = []
            for ly in lines:
                positions.append(("line", ly))
            for i in range(len(lines) - 1):
                positions.append(("space", (lines[i] + lines[i + 1]) / 2))

            kind, nearest_y = min(positions, key=lambda p: abs(cy - p[1]))
            print("nearest staff position:", kind, nearest_y)

            debug = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(debug, (cx, cy), 4, (0, 0, 255), -1)

            for ly in lines:
                cv2.line(debug, (0, ly), (debug.shape[1], ly), (255, 0, 0), 1)

            cv2.imwrite("debug_note_detection.png", debug)
    else:
        print("no notehead-like blob found")

    return best_blob, lines


# pitch detection (splitting cases into treble/bass clef)
# after determining the pitch, we can have this data to be assigned to NoteImage (self.pitch)
# for later use (we should store them somewhere too)

# the default clef will be treble
def pitch_from_position(cy, lines, clef_type="treble"):
    if len(lines) < 5:
        return None

    lines = sorted(lines)
    spaces = [(lines[i] + lines[i + 1]) / 2 for i in range(len(lines) - 1)]

    if clef_type == "treble":
        line_map  = ["E4", "G4", "B4", "D5", "F5"]
        space_map = ["F4", "A4", "C5", "E5"]
    elif clef_type == "bass":
        line_map  = ["G2", "B2", "D3", "F3", "A3"]
        space_map = ["A2", "C3", "E3", "G3"]
    else:
        return None

    line_distances = [abs(cy - y) for y in lines]
    space_distances = [abs(cy - y) for y in spaces]

    min_line_idx = int(np.argmin(line_distances))
    min_space_idx = int(np.argmin(space_distances))

    if line_distances[min_line_idx] <= space_distances[min_space_idx]:
        return line_map[min_line_idx]
    else:
        return space_map[min_space_idx]
    
# access the data from a place we have stored them
# json example structure:
# [
#     {
#         "image_name": "test1.png",
#         "image_path": "testnotes\\test1.png",
#         "type": "16th Notes",
#         "confidence": 0.7855,
#         "pitch": null,
#         "staff_type": null,
#         "staff_idx": null,
#         "note_idx": null
#     },
#     ...
# ]
# the plan is to access the data from the json file, and then assign the pitch
# to the element in the json file, the pitch is determined by the <staff_type> that is from the json file.

location = "note_data/note_images.json"
with open(location, "r") as f:
    note_images_data = json.load(f)

for note_data in note_images_data:
    image_path = note_data["image_path"]
    best_blob, lines = detect_noteheads(image_path)
    if best_blob and lines:
        cx, cy, _, _, _, _, _ = best_blob
        pitch = pitch_from_position(cy, lines)
        note_data["pitch"] = pitch
    print(f"Processed {note_data['image_name']}: pitch = {note_data['pitch']}\n")

# save the updated data back to the json file
with open(location, "w") as f:
    json.dump(note_images_data, f, indent=4)

print("\nUpdated note images data with pitch information saved to:", location)