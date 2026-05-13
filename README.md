# PDF-Music-Sheet-reader
A PDF Music Sheet reader, with basic implementation and imperfect performance


# System Demonstration

## Setup

Create a new folder on your computer and initialize Git inside it. Then open a terminal and install the required dependencies: [file:1]

```bash
pip install opencv-python
pip install matplotlib
pip install torch torchvision
pip install bs4
pip install requests
pip install pypdfium2
```

## Running the Program

Clone or download the project repository into your folder. After that, run the main program from the terminal:

```bash
python main.py
```

You can also open `main.py` directly in VS Code and run it there. When the program starts, it will ask you to select a file to render; alternatively, you may choose a music file stored on your computer as a PNG input.

Intermediate processed images are stored in the following folders:

- `media\\lines`
- `media\\linenotes`

## Post-processing Steps

After `main.py` finishes running, execute the following scripts in the VS Code terminal to complete the pipeline:

### 1. Evaluate detected notes

```bash
python evaluation.py
```

Expected terminal output:

```text
Saved <number> note images to note_data/note_images.json
```

### 2. Assign pitch predictions

```bash
python json_pitch_assign.py
```

Expected terminal output:

```text
Pitch prediction completed and saved to JSON file.
```

### 3. Convert results to final JSON output

```bash
python conversion.py
```

Expected terminal output:

```text
Converted <number> notes → <number> entries written to 'note_data/output.json'
```

## Output

The final processed output is written to:

```text
note_data/output.json
```