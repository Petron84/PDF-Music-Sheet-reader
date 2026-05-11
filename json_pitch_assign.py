"""
This program will be the next step in the pipeline after we have the nearly done NoteImage objects
in the json file. 
We will use the get_pitch function to predict the pitch of each note and assign the pitch determined
to the object in the json file:

Fomat of the json file:
[
    {
        "image_name": "1_2_treble_2png(2)_v1.png",
        "image_path": "test_linenotes\\1_2_treble_2png(2)_v1.png",
        "type": "Quarter Note",
        "confidence": 1.0,
        "pitch": null,
        "staff_type": "treble",
        "staff_idx": 1,
        "note_idx": 2
    },
    {
        "image_name": "1_3_bass_2png(2)_v14.png",
        "image_path": "test_linenotes\\1_3_bass_2png(2)_v14.png",
        "type": "Half Notes",
        "confidence": 1.0,
        "pitch": null,
        "staff_type": "bass",
        "staff_idx": 1,
        "note_idx": 3
    },
    ...
]
The program will access the image_path of each note, use the get_pitch function to predict the pitch, 
and then assign the predicted pitch to the "pitch" field of the NoteImage object in the json file. 
Finally, we will save the updated json file with the predicted pitches.

Return format of json file after pitch prediction:
[
    {
        "image_name": "1_2_treble_2png(2)_v1.png",
        "image_path": "test_linenotes\\1_2_treble_2png(2)_v1.png",
        "type": "Quarter Note",
        "confidence": 1.0,
        "pitch": "F4",
        "staff_type": "treble",
        "staff_idx": 1,
        "note_idx": 2
    },
    {
        "image_name": "1_3_bass_2png(2)_v14.png",
        "image_path": "test_linenotes\\1_3_bass_2png(2)_v14.png",
        "type": "Half Notes",
        "confidence": 1.0,
        "pitch": "A3",
        "staff_type": "bass",
        "staff_idx": 1,
        "note_idx": 3
    },
    ...
]
"""

# from detect_pitch import get_pitch
# pitch = get_pitch("1_3_bass_<song_name>.png")   # → "F3"

from detect_pitch import get_pitch
import json
import os

# access the json file and access the image path of each note, use the get_pitch function to predict the pitch, 
# and then assign the predicted pitch
file_path = "note_data/note_images.json"
with open(file_path, "r") as f:
    note_images = json.load(f)

for note in note_images:
    image_path = note["image_path"]
    note["pitch"] = get_pitch(image_path)


with open(file_path, "w") as f:
    json.dump(note_images, f, indent=4)

print("Pitch prediction completed and saved to JSON file.")