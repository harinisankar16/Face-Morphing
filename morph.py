#!/usr/bin/env python3
"""Ordered N-image face morph; independent of Jupyter. See README for usage."""
from pathlib import Path
import argparse
import json
import math
import re
import hashlib
import cv2
import dlib
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent
CANVAS_SIZE = 400
HAIR_OUTLINE_FRACTIONS = np.linspace(.06, .94, 9)

def load_on_canvas(filename, side=CANVAS_SIZE):
    image = cv2.imread(str(ROOT / filename))
    if image is None:
        raise FileNotFoundError(f"Could not read {filename}")
    h, w = image.shape[:2]
    scale = side / min(h, w)
    image = cv2.resize(image, (math.ceil(w * scale), math.ceil(h * scale)),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    h, w = image.shape[:2]
    top, left = (h - side) // 2, (w - side) // 2
    return image[top:top + side, left:left + side].copy()

def landmarks_with_boundary(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) != 1:
        raise ValueError(f"Expected one face; detected {len(faces)}. Try another crop/photo.")
    shape = predictor(gray, faces[0])
    points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)
    h, w = image.shape[:2]
    if np.any(points < 0) or np.any(points[:, 0] >= w) or np.any(points[:, 1] >= h):
        raise ValueError("A predicted landmark is outside the canvas; use a better-framed photo.")
    boundary = np.array([(0, 0), ((w-1)/2, 0), (w-1, 0),
                         (w-1, (h-1)/2), (w-1, h-1), ((w-1)/2, h-1),
                         (0, h-1), (0, (h-1)/2)], dtype=np.float32)
    return np.vstack([points, boundary])

def parse_face_regions(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = (tensor - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])
    logits = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    labels = logits[0].argmax(axis=0).astype(np.uint8)
    return cv2.resize(labels, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

def trace_visible_hairline(labels, facial_points):
    # Largest skin component excludes isolated false-positive patches.
    count, cc, stats, _ = cv2.connectedComponentsWithStats((labels == 1).astype(np.uint8), 8)
    if count < 2:
        return np.empty((0, 2), np.float32), 'No usable skin region.'
    skin = cc == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    hair_nearby = cv2.dilate((labels == 17).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    face_width = float(np.ptp(facial_points[:17, 0]))
    x0 = max(1, int(facial_points[17:27, 0].min() - .15*face_width))
    x1 = min(labels.shape[1]-2, int(facial_points[17:27, 0].max() + .15*face_width))
    brow_limit = int(facial_points[17:27, 1].max())
    candidate = skin & hair_nearby
    # Ignore beard and lower-face boundaries; find the top skin edge at each column.
    runs, run = [], []
    for x in range(x0, x1+1):
        ys = np.flatnonzero(candidate[:brow_limit, x])
        if len(ys):
            run.append((x, int(ys[0])))
        elif run:
            runs.append(run); run = []
    if run:
        runs.append(run)
    if not runs:
        return np.empty((0, 2), np.float32), 'No continuous visible skin–hair interface above the brows.'
    curve = np.asarray(max(runs, key=len), np.float32)
    if len(curve) < max(20, .35*face_width):
        return np.empty((0, 2), np.float32), 'Visible interface too short; inspect hair, crop, and segmentation.'
    return curve, 'Detected frontal skin–hair interface; sideburns and hidden contours excluded.'

def sample_hairline(curve, count=8):
    length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    targets = np.linspace(.03, .97, count) * length[-1]
    return np.column_stack([np.interp(targets, length, curve[:, d]) for d in range(2)]).astype(np.float32)

def outer_hair_controls(labels, face_points):
    count, component, stats, _ = cv2.connectedComponentsWithStats((labels == 17).astype(np.uint8), 8)
    if count < 2:
        return {}, np.empty((0, 2), np.float32), 'No usable hair region.'
    mask = component == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.where(mask)
    near_background = cv2.dilate((labels == 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    brow_limit = int(face_points[17:27, 1].min())
    contour = []
    for x in range(int(xs.min()), int(xs.max())+1):
        rows = np.flatnonzero(mask[:, x])
        y = int(rows[0]) if len(rows) else -1
        valid = 1 < y < brow_limit and 1 < x < labels.shape[1]-2 and near_background[y, x]
        contour.append((x, float(y) if valid else np.nan))
    contour = np.asarray(contour, np.float32)
    result = {}
    for i, fraction in enumerate(HAIR_OUTLINE_FRACTIONS, 1):
        x = int(round(xs.min() + fraction*(xs.max()-xs.min())))
        y = contour[x-int(xs.min()), 1]
        if np.isfinite(y):
            result[f'T{i}'] = (float(x), float(y))
    message = 'Visible upper hair–background boundary sampled.'
    if ys.min() <= 1:
        message += ' Hair touches the top crop: clipped portions are not treated as a detected contour.'
    return result, contour, message

def largest_region(mask):
    count, component, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count < 2:
        return None
    index = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    if stats[index, cv2.CC_STAT_AREA] < 30:
        return None
    return component == index

def extract_region_controls(labels, face_landmarks):
    controls, masks, messages = {}, {}, []
    mid_x = float(np.median(face_landmarks[:, 0]))
    # Combine the model's two ear classes, then use image-side naming to avoid
    # confusing the subject's anatomical left with the viewer's left.
    for side in ['left', 'right']:
        ear = np.isin(labels, [7, 8])
        cols = np.arange(labels.shape[1])[None, :]
        ear &= cols < mid_x if side == 'left' else cols >= mid_x
        mask = largest_region(ear)
        region = 'ear_image_' + side
        if mask is None:
            messages.append(region + ': missing or too small; no points invented')
            continue
        masks[region] = mask
        ys, xs = np.where(mask)
        if xs.min() == 0 or xs.max() == labels.shape[1] - 1 or ys.min() == 0 or ys.max() == labels.shape[0] - 1:
            messages.append(region + ': touches crop boundary; review visibility')
        # Three points down the visible OUTER ear contour, not helix/tragus landmarks.
        for number, fraction in enumerate([0.15, 0.50, 0.85], 1):
            target_y = ys.min() + fraction * (ys.max() - ys.min())
            row = np.unique(ys)[np.argmin(abs(np.unique(ys) - target_y))]
            xx = xs[ys == row]
            x = xx.min() if side == 'left' else xx.max()
            controls[f'{region}_{number}'] = (float(x), float(row))

    mask = largest_region(labels == 14)  # neck; class 15 is necklace, not neck
    if mask is None:
        messages.append('neck: missing or too small; no points invented')
    else:
        masks['neck'] = mask
        ys, xs = np.where(mask)
        if ys.max() == labels.shape[0] - 1:
            messages.append('neck: cropped at bottom; points describe only the visible portion')
        # Only sample rows with neck pixels on both sides of the face center.
        # Near the jaw, the neck may appear as just one thin side sliver.
        valid_rows = np.array([row for row in np.unique(ys)
                               if xs[ys == row].min() < mid_x - 2
                               and xs[ys == row].max() > mid_x + 2])
        if len(valid_rows) < 6:
            messages.append('neck: insufficient bilateral visible extent; no points added')
            return controls, masks, messages
        for number, fraction in enumerate([0.25, 0.50, 0.75], 1):
            target_y = valid_rows.min() + fraction * (valid_rows.max() - valid_rows.min())
            row = valid_rows[np.argmin(abs(valid_rows - target_y))]
            xx = xs[ys == row]
            controls[f'neck_left_{number}'] = (float(xx.min()), float(row))
            controls[f'neck_right_{number}'] = (float(xx.max()), float(row))
    return controls, masks, messages

def triangulate(points, image_shape):
    h, w = image_shape[:2]
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for point in points:
        subdiv.insert(tuple(map(float, point)))
    triangles = []
    seen = set()
    for raw in subdiv.getTriangleList():
        vertices = raw.reshape(3, 2)
        if np.any(vertices < 0) or np.any(vertices[:, 0] >= w) or np.any(vertices[:, 1] >= h):
            continue
        distances = np.linalg.norm(vertices[:, None, :] - points[None, :, :], axis=2)
        indices = np.argmin(distances, axis=1)
        if np.max(distances[np.arange(3), indices]) > 0.1 or len(set(indices)) != 3:
            raise ValueError("Could not match a triangle to three distinct landmarks.")
        key = tuple(sorted(map(int, indices)))
        if key not in seen:
            seen.add(key)
            triangles.append(tuple(map(int, indices)))
    if not triangles:
        raise ValueError("No triangles found.")
    return triangles

def apply_affine_transform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst

def morph_triangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = apply_affine_transform(img1Rect, t1Rect, tRect, size)
    warpImage2 = apply_affine_transform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

def frame_schedule(n, total):
    """Include every keyframe exactly once; interval counts differ by at most one."""
    if n < 2 or total < n:
        raise ValueError('Need at least two images and --frames >= number of images.')
    q, r = divmod(total - 1, n - 1)
    for segment in range(n - 1):
        steps = q + (segment < r)
        for step in range(0 if segment == 0 else 1, steps + 1):
            yield segment, step / steps


def prepare_pair(image_a, image_b, args, segment):
    MODEL = args.model_dir
    PREDICTOR_PATH = MODEL / "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(PREDICTOR_PATH))

    def landmarks_with_boundary(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        if len(faces) != 1:
            raise ValueError(f'Expected one face; detected {len(faces)}. Try another crop/photo.')
        shape = predictor(gray, faces[0])
        points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)
        h, w = image.shape[:2]
        if np.any(points < 0) or np.any(points[:, 0] >= w) or np.any(points[:, 1] >= h):
            raise ValueError('A predicted landmark is outside the canvas; use a better-framed photo.')
        boundary = np.array([(0, 0), ((w - 1) / 2, 0), (w - 1, 0), (w - 1, (h - 1) / 2), (w - 1, h - 1), ((w - 1) / 2, h - 1), (0, h - 1), (0, (h - 1) / 2)], dtype=np.float32)
        return np.vstack([points, boundary])
    points_a = landmarks_with_boundary(image_a)
    points_b = landmarks_with_boundary(image_b)
    import onnxruntime as ort
    import warnings
    import hashlib
    PARSING_MODEL = MODEL / 'face_parsing_resnet18.onnx'
    PARSING_URL = 'https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx'
    if not PARSING_MODEL.exists():
        raise FileNotFoundError(f'Download {PARSING_URL} into {PARSING_MODEL}; see README.')
    session = ort.InferenceSession(str(PARSING_MODEL), providers=['CPUExecutionProvider'])

    def parse_face_regions(image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = (tensor - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])
        logits = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        labels = logits[0].argmax(axis=0).astype(np.uint8)
        return cv2.resize(labels, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    FOREHEAD_ROW_CONFIG = ((0.1, -0.03), (0.13, -0.05), (0.15, -0.08), (0.33, -0.12))

    def forehead_controls(reference_landmarks, image_shape):
        x, y, w, h = cv2.boundingRect(np.round(reference_landmarks).astype(np.int32))
        controls = []
        for inset, height_offset in FOREHEAD_ROW_CONFIG:
            dx, dy = (int(w * inset), int(h * height_offset))
            controls.extend([(x + dx, y + dy), (x + w - dx, y + dy)])
        controls = np.asarray(controls, dtype=np.float32)
        image_h, image_w = image_shape[:2]
        controls[:, 0] = np.clip(controls[:, 0], 1, image_w - 1)
        controls[:, 1] = np.clip(controls[:, 1], 1, image_h - 1)
        return controls
    base_points_a = landmarks_with_boundary(image_a)
    base_points_b = landmarks_with_boundary(image_b)
    USE_FOREHEAD_CONTROLS = not args.no_hairline
    labels_a, labels_b = (parse_face_regions(image_a), parse_face_regions(image_b))
    legacy_forehead_points = forehead_controls((base_points_a[:68] + base_points_b[:68]) / 2, image_a.shape)

    def trace_visible_hairline(labels, facial_points):
        count, cc, stats, _ = cv2.connectedComponentsWithStats((labels == 1).astype(np.uint8), 8)
        if count < 2:
            return (np.empty((0, 2), np.float32), 'No usable skin region.')
        skin = cc == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        hair_nearby = cv2.dilate((labels == 17).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        face_width = float(np.ptp(facial_points[:17, 0]))
        x0 = max(1, int(facial_points[17:27, 0].min() - 0.15 * face_width))
        x1 = min(labels.shape[1] - 2, int(facial_points[17:27, 0].max() + 0.15 * face_width))
        brow_limit = int(facial_points[17:27, 1].max())
        candidate = skin & hair_nearby
        runs, run = ([], [])
        for x in range(x0, x1 + 1):
            ys = np.flatnonzero(candidate[:brow_limit, x])
            if len(ys):
                run.append((x, int(ys[0])))
            elif run:
                runs.append(run)
                run = []
        if run:
            runs.append(run)
        if not runs:
            return (np.empty((0, 2), np.float32), 'No continuous visible skin–hair interface above the brows.')
        curve = np.asarray(max(runs, key=len), np.float32)
        if len(curve) < max(20, 0.35 * face_width):
            return (np.empty((0, 2), np.float32), 'Visible interface too short; inspect hair, crop, and segmentation.')
        return (curve, 'Detected frontal skin–hair interface; sideburns and hidden contours excluded.')

    def sample_hairline(curve, count=8):
        length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        targets = np.linspace(0.03, 0.97, count) * length[-1]
        return np.column_stack([np.interp(targets, length, curve[:, d]) for d in range(2)]).astype(np.float32)
    hairline_a, hairline_status_a = trace_visible_hairline(labels_a, base_points_a[:68])
    hairline_b, hairline_status_b = trace_visible_hairline(labels_b, base_points_b[:68])
    hairline_pair_available = len(hairline_a) > 0 and len(hairline_b) > 0
    forehead_points_a = sample_hairline(hairline_a) if hairline_pair_available else np.empty((0, 2), np.float32)
    forehead_points_b = sample_hairline(hairline_b) if hairline_pair_available else np.empty((0, 2), np.float32)
    points_a = np.vstack([base_points_a[:68], forehead_points_a, base_points_a[68:]]) if USE_FOREHEAD_CONTROLS else base_points_a.copy()
    points_b = np.vstack([base_points_b[:68], forehead_points_b, base_points_b[68:]]) if USE_FOREHEAD_CONTROLS else base_points_b.copy()
    if not hairline_pair_available:
        print('Omitting hairline controls from BOTH images: no guessed fallback.')
    USE_HAIR_OUTLINE_CONTROLS = not args.no_hair_outline
    HAIR_OUTLINE_FRACTIONS = np.linspace(0.06, 0.94, 9)

    def outer_hair_controls(labels, face_points):
        count, component, stats, _ = cv2.connectedComponentsWithStats((labels == 17).astype(np.uint8), 8)
        if count < 2:
            return ({}, np.empty((0, 2), np.float32), 'No usable hair region.')
        mask = component == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        ys, xs = np.where(mask)
        near_background = cv2.dilate((labels == 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        brow_limit = int(face_points[17:27, 1].min())
        contour = []
        for x in range(int(xs.min()), int(xs.max()) + 1):
            rows = np.flatnonzero(mask[:, x])
            y = int(rows[0]) if len(rows) else -1
            valid = 1 < y < brow_limit and 1 < x < labels.shape[1] - 2 and near_background[y, x]
            contour.append((x, float(y) if valid else np.nan))
        contour = np.asarray(contour, np.float32)
        result = {}
        for i, fraction in enumerate(HAIR_OUTLINE_FRACTIONS, 1):
            x = int(round(xs.min() + fraction * (xs.max() - xs.min())))
            y = contour[x - int(xs.min()), 1]
            if np.isfinite(y):
                result[f'T{i}'] = (float(x), float(y))
        message = 'Visible upper hair–background boundary sampled.'
        if ys.min() <= 1:
            message += ' Hair touches the top crop: clipped portions are not treated as a detected contour.'
        return (result, contour, message)
    hair_controls_a, hair_outline_a, hair_outline_status_a = outer_hair_controls(labels_a, base_points_a[:68])
    hair_controls_b, hair_outline_b, hair_outline_status_b = outer_hair_controls(labels_b, base_points_b[:68])
    hair_point_names = []
    if USE_HAIR_OUTLINE_CONTROLS:
        for name in hair_controls_a:
            if name not in hair_controls_b:
                continue
            a, b = (np.asarray(hair_controls_a[name]), np.asarray(hair_controls_b[name]))
            if min(np.linalg.norm(points_a - a, axis=1).min(), np.linalg.norm(points_b - b, axis=1).min(), np.linalg.norm((points_a + points_b) / 2 - (a + b) / 2, axis=1).min()) < 2:
                continue
            points_a = np.vstack([points_a[:-8], a, points_a[-8:]]).astype(np.float32)
            points_b = np.vstack([points_b[:-8], b, points_b[-8:]]).astype(np.float32)
            hair_point_names.append(name)
    stage2_points_a, stage2_points_b = (points_a.copy(), points_b.copy())
    USE_EAR_NECK_CONTROLS = not args.no_ear_neck
    all_overrides = json.loads(args.overrides.read_text()) if args.overrides else {}
    REVIEWED_POINT_OVERRIDES = {'A': all_overrides.get(str(segment), {}), 'B': all_overrides.get(str(segment + 1), {})}

    def largest_region(mask):
        count, component, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        if count < 2:
            return None
        index = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        if stats[index, cv2.CC_STAT_AREA] < 30:
            return None
        return component == index

    def extract_region_controls(labels, face_landmarks):
        controls, masks, messages = ({}, {}, [])
        mid_x = float(np.median(face_landmarks[:, 0]))
        for side in ['left', 'right']:
            ear = np.isin(labels, [7, 8])
            cols = np.arange(labels.shape[1])[None, :]
            ear &= cols < mid_x if side == 'left' else cols >= mid_x
            mask = largest_region(ear)
            region = 'ear_image_' + side
            if mask is None:
                messages.append(region + ': missing or too small; no points invented')
                continue
            masks[region] = mask
            ys, xs = np.where(mask)
            if xs.min() == 0 or xs.max() == labels.shape[1] - 1 or ys.min() == 0 or (ys.max() == labels.shape[0] - 1):
                messages.append(region + ': touches crop boundary; review visibility')
            for number, fraction in enumerate([0.15, 0.5, 0.85], 1):
                target_y = ys.min() + fraction * (ys.max() - ys.min())
                row = np.unique(ys)[np.argmin(abs(np.unique(ys) - target_y))]
                xx = xs[ys == row]
                x = xx.min() if side == 'left' else xx.max()
                controls[f'{region}_{number}'] = (float(x), float(row))
        mask = largest_region(labels == 14)
        if mask is None:
            messages.append('neck: missing or too small; no points invented')
        else:
            masks['neck'] = mask
            ys, xs = np.where(mask)
            if ys.max() == labels.shape[0] - 1:
                messages.append('neck: cropped at bottom; points describe only the visible portion')
            valid_rows = np.array([row for row in np.unique(ys) if xs[ys == row].min() < mid_x - 2 and xs[ys == row].max() > mid_x + 2])
            if len(valid_rows) < 6:
                messages.append('neck: insufficient bilateral visible extent; no points added')
                return (controls, masks, messages)
            for number, fraction in enumerate([0.25, 0.5, 0.75], 1):
                target_y = valid_rows.min() + fraction * (valid_rows.max() - valid_rows.min())
                row = valid_rows[np.argmin(abs(valid_rows - target_y))]
                xx = xs[ys == row]
                controls[f'neck_left_{number}'] = (float(xx.min()), float(row))
                controls[f'neck_right_{number}'] = (float(xx.max()), float(row))
        return (controls, masks, messages)
    controls_a, masks_a, warnings_a = extract_region_controls(labels_a, base_points_a[:68])
    controls_b, masks_b, warnings_b = extract_region_controls(labels_b, base_points_b[:68])
    for controls, overrides in [(controls_a, REVIEWED_POINT_OVERRIDES['A']), (controls_b, REVIEWED_POINT_OVERRIDES['B'])]:
        for name, xy in overrides.items():
            if name not in controls:
                raise ValueError('Review existing detections only; missing region: ' + name)
            x, y = map(float, xy)
            if not (0 <= x < image_a.shape[1] and 0 <= y < image_a.shape[0]):
                raise ValueError('Override outside canvas: ' + name)
            controls[name] = (x, y)
    common_names = []
    for region in ['ear_image_left', 'ear_image_right', 'neck']:
        names = [name for name in controls_a if name.startswith(region + '_')]
        if names and all((name in controls_b for name in names)):
            common_names.extend(names)
        else:
            print(f'Omitting unmatched region from morph: {region}')

    def stage_points(base, forehead, include_forehead):
        return np.vstack([base[:68], forehead, base[68:]]) if include_forehead else base.copy()
    points_a = stage2_points_a.copy()
    points_b = stage2_points_b.copy()
    accepted_names = []
    if USE_EAR_NECK_CONTROLS:
        for name in common_names:
            a, b = (np.asarray(controls_a[name]), np.asarray(controls_b[name]))
            distances = [np.min(np.linalg.norm(points_a - a, axis=1)), np.min(np.linalg.norm(points_b - b, axis=1)), np.min(np.linalg.norm((points_a + points_b) / 2 - (a + b) / 2, axis=1))]
            if min(distances) < 2.0:
                print('Skipping near-duplicate control in both images:', name)
                continue
            points_a = np.vstack([points_a[:-8], a, points_a[-8:]]).astype(np.float32)
            points_b = np.vstack([points_b[:-8], b, points_b[-8:]]).astype(np.float32)
            accepted_names.append(name)
    for prefix, messages in [('A', warnings_a), ('B', warnings_b)]:
        for message in messages:
            print(f'Review {prefix}: {message}')
    return points_a, points_b, dict(hair_controls=hair_point_names, region_controls=accepted_names, notes=warnings_a+warnings_b)


def render_pair(a, b, pa, pb, triangles, alpha):
    target = (1-alpha)*pa + alpha*pb
    result = np.zeros(a.shape, np.float32)
    for tri in triangles:
        idx = list(tri)
        morph_triangle(a, b, result, pa[idx], pb[idx], target[idx], alpha)
    return np.clip(result,0,255).astype(np.uint8), target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', nargs='*', type=Path, help='Ordered image paths; otherwise scan input folder in natural filename order.')
    parser.add_argument('--input-dir', type=Path, default=ROOT/'input')
    parser.add_argument('--model-dir', type=Path, default=ROOT/'models')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'output')
    parser.add_argument('--frames', type=int, default=61)
    parser.add_argument('--fps', type=float, default=10)
    parser.add_argument('--size', type=int, default=400)
    parser.add_argument('--gif', action='store_true')
    parser.add_argument('--video', action='store_true', help='Also export MP4.')
    parser.add_argument('--show-mesh', action='store_true')
    parser.add_argument('--no-hairline', action='store_true')
    parser.add_argument('--no-hair-outline', action='store_true')
    parser.add_argument('--no-ear-neck', action='store_true')
    parser.add_argument('--overrides', type=Path, help='JSON: image index -> detected ear/neck point name -> [x,y], on the cropped canvas.')
    args = parser.parse_args(argv)
    paths = args.images or sorted((p for p in args.input_dir.iterdir()
        if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}),
        key=lambda p: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)',p.name)])
    schedule = list(frame_schedule(len(paths), args.frames))
    if args.size < 64 or not np.isfinite(args.fps) or args.fps <= 0:
        parser.error('Invalid size (minimum 64), or fps.')
    # A dedicated directory prevents silently mixing frames from different runs.
    destination = args.output_dir/'morph'
    if destination.exists() and any(destination.iterdir()):
        parser.error(f'{destination} is not empty. Choose a new --output-dir or move the prior run.')
    images = [load_on_canvas(p.resolve(), args.size) for p in paths]
    bundles = []
    destination.mkdir(parents=True,exist_ok=True)
    diagnostics = args.output_dir/'diagnostics'
    diagnostics.mkdir(exist_ok=True)
    for segment in range(len(images)-1):
        pa,pb,meta = prepare_pair(images[segment],images[segment+1],args,segment)
        triangles = triangulate((pa+pb)/2,images[segment].shape)
        bundles.append((pa,pb,triangles,meta))
        previews = []
        for im,points in zip(images[segment:segment+2],[pa,pb]):
            preview = im.copy()
            for j,(x,y) in enumerate(points):
                cv2.circle(preview,(round(float(x)),round(float(y))),2,(0,255,255) if j<68 else (255,100,0),-1)
            previews.append(preview)
        cv2.imwrite(str(diagnostics/f'pair_{segment:02d}.png'),np.hstack(previews))
    video = None
    if args.video:
        video = cv2.VideoWriter(str(args.output_dir/'morph.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),args.fps,(args.size,args.size))
        if not video.isOpened():
            raise RuntimeError('MP4 encoder could not open.')
    gif_frames = []
    float_images = [im.astype(np.float32) for im in images]
    try:
        for index,(segment,alpha) in enumerate(schedule):
            pa,pb,triangles,_ = bundles[segment]
            frame,target = render_pair(float_images[segment],float_images[segment+1],pa,pb,triangles,alpha)
            if args.show_mesh:
                for tri in triangles:
                    cv2.polylines(frame,[target[list(tri)].astype(np.int32)],True,(255,255,255),1)
            if not cv2.imwrite(str(destination/f'frame_{index:04d}.png'),frame):
                raise IOError('Failed to write frame.')
            if video is not None:
                video.write(frame)
            if args.gif:
                gif_frames.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
    finally:
        if video is not None:
            video.release()
    if args.gif:
        gif_frames[0].save(args.output_dir/'morph.gif',save_all=True,append_images=gif_frames[1:],
            duration=max(10,round(1000/args.fps)),loop=0,optimize=False)
    manifest = dict(images=[str(p.resolve()) for p in paths],mode="notebook_pairwise",
        frames=args.frames, fps=args.fps,
        pairs=[dict(points_a=pa.tolist(),points_b=pb.tolist(),triangles=tri,**meta)
               for pa,pb,tri,meta in bundles],
        schedule=[dict(segment=s,alpha=a) for s,a in schedule],
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='images'},
        models={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [args.model_dir/'shape_predictor_68_face_landmarks.dat',args.model_dir/'face_parsing_resnet18.onnx']})
    (args.output_dir/'morph_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Wrote {len(schedule)} notebook-pairwise frames to {destination}.')


if __name__ == '__main__':
    main()
