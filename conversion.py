"""
Below is the structure of the json file that will undergo conversion to a different kind of json file.
[
    {
        "image_name": "1_2_treble_2png(2)_v1.png",
        "image_path": "linenotes\\1_2_treble_2png(2)_v1.png",
        "type": "Quarter Note",
        "confidence": 1.0,
        "pitch": "A4",
        "staff_type": "treble",
        "staff_idx": 1,
        "note_idx": 2
    },
    {
        "image_name": "1_3_bass_2png(2)_v14.png",
        "image_path": "linenotes\\1_3_bass_2png(2)_v14.png",
        "type": "Half Notes",
        "confidence": 1.0,
        "pitch": "F3",
        "staff_type": "bass",
        "staff_idx": 1,
        "note_idx": 3
    },
    ...
]
The json file has been sorted by staff_idx and note_idx before.
This will help us constructing the new json file in the correct order.

This will be convereted to the following structure:
song name = <info extracted from the file name>
bpm = <assumed info>
[
    {
        "note": "A4",
        "start": 0.0, (this is determined by the order of the notes and the bpm, the order of the note will be taken from the <image_name>)
        "duration": 0.25 (this will be determined by the <type> of the note, and the bpm)
    },
    {
        "note": "F3",
        "start": 0.25,
        "duration": 0.5 (if quarter note is 0.25, then half note is 0.5, whole note is 1.0, etc.)
    },
    ...
]

To construct in the correct order, we will process the notes that are in the same staff idx,
then, based on the note idx, we will determine the start time of the note, 
and based on the type of the note, we will determine the duration of the note.

There will be 2 parralel processes, one for the treble staff and one for the bass staff, 
and then we will merge the two processes together based on the start time of the notes.
"""

import json
from typing import List, Dict, Any


NOTE_DURATIONS = {
    "Whole Note":    4.0,
    "Half Note":     2.0,
    "Half Notes":    2.0,   # handle plural variants
    "Quarter Note":  1.0,
    "Quarter Notes": 1.0,
    "Eighth Note":   0.5,
    "Eighth Notes":  0.5,
    "16th Note":     0.25,
    "16th Notes":    0.25,
}


def get_duration(note_type: str, bpm: int) -> float:
    """Return duration in seconds for a given note type at the specified BPM."""
    beats = NOTE_DURATIONS.get(note_type, 0)
    seconds_per_beat = 60.0 / bpm
    return beats * seconds_per_beat


def process_staff(notes: List[Dict[str, Any]], bpm: int) -> List[Dict[str, Any]]:
    """
    Process notes belonging to a single staff type (treble or bass).

    Notes are expected to be pre-sorted by staff_idx then note_idx.
    Returns a list of note dicts with 'note', 'start', and 'duration',
    grouped by staff_idx (each staff_idx timeline restarts from 0.0).
    """
    result = []

    # Group by staff_idx so each measure/staff block gets its own timeline
    staff_groups: Dict[int, List[Dict[str, Any]]] = {}
    for item in notes:
        idx = item["staff_idx"]
        staff_groups.setdefault(idx, []).append(item)

    for staff_idx in sorted(staff_groups.keys()):
        current_time = 0.0
        for item in sorted(staff_groups[staff_idx], key=lambda x: x["note_idx"]):
            duration = get_duration(item["type"], bpm)
            result.append({
                "note":     item["pitch"],
                "start":    round(current_time, 6),
                "duration": round(duration, 6),
                "staff_idx": staff_idx,
            })
            current_time += duration

    return result


def merge_staves(
    treble: List[Dict[str, Any]],
    bass: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge treble and bass note lists into a single timeline.

    Within each staff_idx block, treble and bass play simultaneously,
    so their start times are offset together: the bass track's start
    times are anchored to the same staff_idx block as the treble track.
    Notes from both staves within the same staff_idx are interleaved by
    start time, then sorted globally.
    """
    # Collect the cumulative start offset per staff_idx from the treble track
    # (use treble as the reference timeline; if treble is missing a block, fall back to bass)
    block_offsets: Dict[int, float] = {}
    cumulative = 0.0

    all_staff_idxs = sorted(
        set(n["staff_idx"] for n in treble) | set(n["staff_idx"] for n in bass)
    )

    # Build per-block durations from treble (or bass as fallback)
    def block_duration(notes_list, sidx):
        items = [n for n in notes_list if n["staff_idx"] == sidx]
        if not items:
            return 0.0
        return max(n["start"] + n["duration"] for n in items)

    for sidx in all_staff_idxs:
        block_offsets[sidx] = cumulative
        treble_dur = block_duration(treble, sidx)
        bass_dur = block_duration(bass, sidx)
        cumulative += max(treble_dur, bass_dur)

    merged = []
    for note in treble + bass:
        offset = block_offsets[note["staff_idx"]]
        merged.append({
            "note":     note["note"],
            "start":    round(note["start"] + offset, 6),
            "duration": note["duration"],
        })

    merged.sort(key=lambda n: (n["start"], n["note"]))
    return merged


def convert_json(input_file: str, output_file: str, bpm: int = 120) -> None:
    """
    Convert the detected-note JSON into a timing-based note list JSON.

    Reads `input_file`, splits notes by staff_type into treble and bass
    tracks, processes each track independently (start times restart per
    staff_idx block), then merges both tracks into a single sorted list
    and writes the result to `output_file`.

    Args:
        input_file:  Path to the input JSON file.
        output_file: Path to write the output JSON file.
        bpm:         Beats per minute used to convert beat durations to seconds.
    """
    with open(input_file, "r") as f:
        data: List[Dict[str, Any]] = json.load(f)

    # Split by staff type
    treble_notes = [n for n in data if n.get("staff_type") == "treble"]
    bass_notes   = [n for n in data if n.get("staff_type") == "bass"]

    # Process each staff independently
    treble_timeline = process_staff(treble_notes, bpm)
    bass_timeline   = process_staff(bass_notes, bpm)

    # Merge into one sorted timeline
    merged = merge_staves(treble_timeline, bass_timeline)

    with open(output_file, "w") as f:
        json.dump(merged, f, indent=4)

    print(f"Converted {len(data)} notes → {len(merged)} entries written to '{output_file}'")


if __name__ == "__main__":
    input_file  = "note_data/note_images.json"   # Replace with your input file path
    output_file = "note_data/output.json"  # Replace with your desired output file path
    bpm         = 120            # Adjust as needed
    convert_json(input_file, output_file, bpm)