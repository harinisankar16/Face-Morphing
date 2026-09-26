# Face Morphing Tutorial

Learn face morphing step by step in `face_morphing.ipynb`, or use `morph.py` to
create a sequence from two or more images.

## Installation

You need Git and Python. The project has been run with Python 3.12.
Run these commands in a terminal:

```bash
git clone https://github.com/CyrusYang-cy/Face-Morphing-Tutorial.git
cd Face-Morphing-Tutorial
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the Python dependencies for both the notebook and script:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your system uses `python3` instead of `python`, use that command to create
the environment. If installing `dlib` requires a source build, install CMake
and a C++ compiler first. On macOS, the compiler is available through
`xcode-select --install`; on Windows, use Visual Studio Build Tools with the
C++ workload. These system tools are not installed by `requirements.txt`.

### Download the model

The dlib landmark model, `models/shape_predictor_68_face_landmarks.dat`, is
included in the repository. The face-parsing ONNX model is **not included**;
download it once before running either the notebook or script:

```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx', 'models/face_parsing_resnet18.onnx')"
```

Both files must be in `models/`. Image processing runs locally.

## Run the Jupyter notebook

From the repository folder, with the environment activated:

```bash
python -m jupyter lab face_morphing.ipynb
```

1. Select the two image filenames using `IMAGE_A` and `IMAGE_B` in the first
   code cell. You can also change the canvas size and frame count there.
2. Choose **Run → Run All Cells**.
3. View the figures and animation in the notebook. Saved results are in
   `output/`, including `face_morph_tutorial.gif`.

Launch Jupyter from the repository folder so relative input/model paths work.
If you use another Jupyter installation, select a kernel containing the
packages in `requirements.txt`.

## Run the Python script

The script morphs adjacent images in order: 1 → 2 → 3 → … → N.
For the included eleven-image sequence on macOS/Linux:

```bash
python morph.py input/DPD_1_BM12_p5b_*.png \
  --frames 61 --gif --output-dir output/my_sequence
```

Alternatively, pass each filename explicitly in the desired order (also works
in Windows PowerShell):

```bash
python morph.py input/DPD_1_BM12_p5b_01.png input/DPD_1_BM12_p5b_06.png input/DPD_1_BM12_p5b_11.png --frames 61 --gif --output-dir output/my_three_images
```

Without filenames, the script reads supported images from `input/` in natural
filename order. The included folder contains multiple face sets, so select a
sequence explicitly or use `--input-dir` with a folder containing just that set.

- `--frames X`: save exactly X PNGs in `<output-dir>/morph/`. X must be at least
  the number of inputs. Transitions receive as equal a number of intervals as
  possible, without duplicating shared keyframes.
- `--gif` / `--video`: optionally save `morph.gif` / `morph.mp4`.
- `--fps 10`: animation playback speed.
- `--output-dir PATH`: output folder; default is `output/` next to the script.
- `--input-dir PATH` / `--model-dir PATH`: override the default `input/` / `models/`.
- `--help`: list all options.

Use a new output folder for each run; the script refuses a nonempty `morph/`
folder to avoid mixing frames. Landmark previews are saved in `diagnostics/`,
and run details are saved in `morph_manifest.json`.

## Credits

Adapted from [a-tabaza/face-morphing](https://github.com/a-tabaza/face-morphing),
which credits [Azmarie/Face-Morphing](https://github.com/Azmarie/Face-Morphing).
The face-parsing model comes from
[yakhyo/face-parsing](https://github.com/yakhyo/face-parsing).
