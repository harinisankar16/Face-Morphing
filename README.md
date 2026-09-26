# Face Morphing with Python
Face morphing is a technique that involves blending two or more images of faces together in order to create a new image that appears to be a hybrid of the original faces. [Code Source](https://github.com/Azmarie/Face-Morphing/)
### Libraries Used:
 1. glob 
 2. os 
 3. sys 
 4. PIL 
 5. OpenCV 
 6. imutils 
 7. dlib 
 8. skimage 
 9. matplotlib 
 10. numpy  
 11. random
 12. math
 
 #### Concepts & Ideas used:
  1. Basic Image data manipulation and processing using OpenCV and PIL
  2. Facial Landmarks Recognition using pretrained models from dlib library
  3. Delaunay triangulation using OpenCV
  4. Advanced morphology for blending and warping using OpenCV

[Example 1](https://i.imgur.com/FwNK3YW.mp4)
[Example 2](https://i.imgur.com/fLD62Ez.mp4)
[Example 3](https://i.imgur.com/gJzfA89.mp4)

##### Make sure to change the paths to the images used, all files are linked on [Google Drive](https://drive.google.com/drive/folders/1n7iRsWiVVOQiN9m8oxz8yjI2mHFY4ZeC?usp=share_link). Images are uploaded on this repo.






## Local beginner tutorial

`face_morphing.ipynb` has been reorganized as a visual, locally runnable tutorial. It includes landmark overlays, a shared triangle mesh, alpha examples, a cross-fade comparison, and an exported GIF.

```bash
cd ~/Development/face-morphing-tutorial
python -m jupyter lab face_morphing.ipynb
```

Use an environment with NumPy, OpenCV, dlib, Pillow, Matplotlib, and Jupyter. The existing Anaconda environment was used to execute the default example successfully. Original upstream files remain available in Git history. Outputs go to `tutorial_output/`.

### Ear and neck extension (section 3.3)

Section 3 now compares 76 base points, 84 points with forehead controls, and up to 96 points with ear/neck contours derived from BiSeNet face parsing. The final figure shows both images at all three stages. These contour samples are approximate correspondences; inspect masks, missing regions, and crop warnings before experimental use.

The ONNX model is already downloaded locally. For a fresh checkout:

```bash
python -m pip install onnxruntime
mkdir -p models
curl -L --fail https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx -o models/face_parsing_resnet18.onnx
```

Model source and inference conventions: https://github.com/yakhyo/face-parsing (MIT-licensed repository). The source URL and local weight hash are recorded in `models/face_parsing_model.json`. No user photographs are uploaded for inference.

Outputs include `tutorial_output/landmark_stages_comparison.png`, raw label maps, `extended_landmarks.json`, and the updated animation. The example notebook ran end to end with 96 points. Across 101 alpha samples, its triangles did not flip or collapse; this is a geometry check, not validation of perceived realism. Both sample necks reach the bottom of the crop, so only their visible portions are represented.

### Hairline refinement

Section 3.2 now traces the predicted frontal skin–hair interface separately in each image, samples eight controls along it, and shows the former geometric offsets as gray crosses. Missing contours are omitted from both meshes. The current selected DPD pair executes with 90 points: 68 face, 8 hairline, 8 canvas, and 6 ear controls; no neck region is visible/detected. See the selected output folder for the regenerated comparison.

## Standalone N-image script

`morph.py` repeats the current notebook's pairwise procedure for each adjacent
pair: 1 → 2 → 3 → … → N. Each pair independently detects its matching hairline,
outer-hair, ear/neck controls, retains eight canvas points, and triangulates the
average landmarks. It does not align to a central face or apply a fixed hull.
The notebook itself is unchanged.

```bash
cd ~/Development/face-morphing-tutorial
/Users/cyrusyang/anaconda3/bin/python morph.py \
  input/DPD_1_BM12_p5b_*.png --frames 61 --gif --output-dir output/my_sequence
```

Explicit paths retain their supplied order. Without paths, supported images in
`input/` are naturally sorted; the current folder contains multiple face sets,
so select your sequence explicitly. Defaults are relative to the script:
`input/`, `models/`, and `output/`. Requires NumPy, OpenCV, dlib, Pillow,
onnxruntime, and the existing two model files. No Jupyter dependency.

- `--frames X`: exactly X PNGs in `<output-dir>/morph/`; X must be at least N.
  Distribute X−1 intervals across N−1 transitions, with any remainder assigned
  to earlier transitions. Shared keyframes appear once. Eleven images and
  61 frames give six intervals per transition.
- `--gif` and `--video`: optional `morph.gif` and `morph.mp4` in the output root.
- `--fps 10`: playback speed; `--size 400`: square center-cropped canvas.
- `--input-dir`, `--model-dir`, `--output-dir`: override default folders.
- `--show-mesh`: draw triangles on exports.
- `--no-hairline`, `--no-hair-outline`, `--no-ear-neck`: disable extensions.
- `--overrides FILE`: JSON mapping zero-based image indices to detected
  ear/neck point names and corrected `[x,y]` canvas coordinates.

Nonempty frame folders are refused to avoid mixing runs. `diagnostics/` holds
pairwise landmark comparisons; `morph_manifest.json` records points, triangles,
settings, model hashes, and frame mapping. Missing controls are omitted per
pair, so adjacent segments can use different meshes. Inspect transitions and
contours before experimental use.

Corrected 61-frame example: `output/eleven_notebook_pairwise/morph.gif`.
The older `output/eleven_example/` is retained for comparison and uses the
superseded central-anchor method.
