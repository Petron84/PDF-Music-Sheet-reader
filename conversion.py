# Below is the structure of the json file that will undergo conversion to a different kind of json file.
# [
#     {
#         "image_name": "test1.png",
#         "image_path": "testnotes\\test1.png",
#         "type": "16th Notes",
#         "confidence": 0.7855,
#         "pitch": "C5",
#         "staff_type": null,
#         "staff_idx": null,
#         "note_idx": null
#     },
#     {
#         "image_name": "test2.png",
#         "image_path": "testnotes\\test2.png",
#         "type": "Quarter Note",
#         "confidence": 0.999,
#         "pitch": "E5",
#         "staff_type": null,
#         "staff_idx": null,
#         "note_idx": null
#     },
#     ...
# ]
# This will be convereted to the following structure:
# song name = <info extracted from the file name>
# bpm = <assumed info>
# [
#     {
#         "note": "C5",
#         "start": 0.0, (this is determined by the order of the notes and the bpm, the order of the note will be taken from the <image_name>)
#         "duration": 0.25 (this will be determined by the <type> of the note, and the bpm)
#     },
#     {
#         "note": "E5",
#         "start": 0.25,
#         "duration": 1.0 (the last note if 16th and duration is 0.25, then the next note start at 0.25, and last 1.0 because it is a quarter note)
#     }
# ]
# Though for now, we will just focus on the <duration> and <note> fields, because we dont have the complete json file yet.
import json
import os
from typing import List, Dict, Any


def get_duration(note_type: str, bpm: int) -> float:
    # Define the duration of each note type in terms of beats
    note_durations = {
        "Whole Note": 4.0,
        "Half Note": 2.0,
        "Quarter Note": 1.0,
        "Eighth Note": 0.5,
        "16th Notes": 0.25
    }
    
    # Get the duration in beats for the given note type
    beats = note_durations.get(note_type, 0)
    
    # Convert beats to seconds based on the bpm
    seconds_per_beat = 60.0 / bpm
    return beats * seconds_per_beat


def convert_json(input_file: str, output_file: str, bpm: int = 120) -> None:
    with open(input_file, 'r') as f:
        data = json.load(f)

    converted_data = []
    current_time = 0.0

    for item in data:
        note_info = {
            "note": item["pitch"],
            "start": current_time,
            "duration": get_duration(item["type"], bpm)
        }
        converted_data.append(note_info)
        current_time += note_info["duration"]

    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=4)

if __name__ == "__main__":
    input_file = "input.json"  # Replace with your input file path
    output_file = "output.json"  # Replace with your desired output file path
    bpm = 120  # You can adjust the bpm as needed
    convert_json(input_file, output_file, bpm)