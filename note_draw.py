import cv2
import numpy as np
import random

random.seed(42)  # For reproducible results

IMG_SIZE = 64

def draw_note(pitch_index=0, note_type="quarter"):
    img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255

    line_spacing = 6
    staff_center = 32
    bottom_line_y = staff_center + 2 * line_spacing

    # Draw staff lines
    for i in range(5):
        y = int(bottom_line_y - i * line_spacing)
        cv2.line(img, (0, y), (IMG_SIZE-1, y), 0, 1)

    # Compute note position
    note_y = int(bottom_line_y - pitch_index * (line_spacing / 2))
    note_x = 32 + random.randint(-2, 2)  # small horizontal jitter

    # Slight vertical jitter
    note_y += random.randint(-1, 1)

    # Determine note head fill
    filled = note_type in ["quarter", "eighth", "sixteenth"]

    thickness = -1 if filled else 1

    # Draw head
    cv2.ellipse(img, (note_x, note_y),
                (5, 4), 0, 0, 360, 0, thickness)

    # Stem logic
    if note_type != "whole":
        stem_height = 18

        # Auto stem direction: below middle line → up
        middle_line_y = int(bottom_line_y - 2 * line_spacing)
        stem_up = note_y > middle_line_y

        if stem_up:
            stem_x = note_x + 5
            cv2.line(img,
                     (stem_x, note_y),
                     (stem_x, note_y - stem_height),
                     0, 1)
            flag_base_y = note_y - stem_height
        else:
            stem_x = note_x - 5
            cv2.line(img,
                     (stem_x, note_y),
                     (stem_x, note_y + stem_height),
                     0, 1)
            flag_base_y = note_y + stem_height

        # Flags
        flag_count = 0
        if note_type == "eighth":
            flag_count = 1
        elif note_type == "sixteenth":
            flag_count = 2

        for i in range(flag_count):
            offset = i * 4
            if stem_up:
                cv2.ellipse(img,
                            (stem_x + 3, flag_base_y + offset),
                            (4, 2),
                            0, 0, 180,
                            0, -1)
            else:
                cv2.ellipse(img,
                            (stem_x - 3, flag_base_y - offset),
                            (4, 2),
                            0, 180, 360,
                            0, -1)

    return img

##########################################################################

import os

note_types = [
    "whole",
    "half",
    "quarter",
    "eighth",
    "sixteenth"
]

base_path = "dataset"

for note_type in note_types:
    os.makedirs(f"{base_path}/{note_type}", exist_ok=True)

count_per_class = 2000

for note_type in note_types:
    for i in range(count_per_class):

        pitch = random.randint(-4, 10)

        img = draw_note(
            pitch_index=pitch,
            note_type=note_type
        )

        filename = f"{base_path}/{note_type}/{note_type}_{i}.png"
        cv2.imwrite(filename, img)

print("Dataset generation complete.")

