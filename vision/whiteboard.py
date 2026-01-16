#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

COS_22_5 = math.cos(math.pi / 8.0)  # cos(22.5°)

# Optional SVG rasterization deps
try:
    import cairosvg  # type: ignore
    from PIL import Image  # type: ignore
except Exception:
    cairosvg = None
    Image = None


@dataclass
class Det:
    x: int
    y: int
    r: int
    score: float
    delta: float
    edge: float
    ring_std: float
    ink_frac: float

    octagon: Optional[np.ndarray] = None  # Nx1x2 int32 in image coords

    # debug
    cy: Optional[float] = None
    scale: Optional[float] = None

    # icon match
    icon_label: Optional[str] = None
    icon_conf: Optional[float] = None
    icon_score: Optional[float] = None
    icon_label2: Optional[str] = None
    icon_score2: Optional[float] = None


@dataclass
class IconTemplate:
    label: str
    edges: np.ndarray          # uint8 edge image (template_size x template_size)
    dt_bank: List[np.ndarray]  # list of DTs per orientation bin


# -----------------------------
# Basic helpers
# -----------------------------

def clahe(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def annulus_mask(shape_hw: Tuple[int, int], x: int, y: int, r_in: int, r_out: int) -> np.ndarray:
    h, w = shape_hw
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (x, y), r_out, 255, -1)
    cv2.circle(m, (x, y), r_in, 0, -1)
    return m


def build_ink_masks(bgr: np.ndarray, dark_v_thresh: int, green_s_thresh: int) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, green_s_thresh, 40), (90, 255, 255))
    v = hsv[:, :, 2]
    dark = cv2.inRange(v, 0, dark_v_thresh)
    return green, dark


# -----------------------------
# Hole detection scoring
# -----------------------------

def circle_delta(gray: np.ndarray, x: int, y: int, r: int) -> float:
    h, w = gray.shape[:2]
    if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
        return -1.0

    mask_in = np.zeros_like(gray, np.uint8)
    cv2.circle(mask_in, (x, y), int(r * 0.55), 255, -1)
    inner = float(cv2.mean(gray, mask=mask_in)[0])

    mask_ring = np.zeros_like(gray, np.uint8)
    cv2.circle(mask_ring, (x, y), int(r * 1.35), 255, -1)
    cv2.circle(mask_ring, (x, y), int(r * 0.85), 0, -1)
    outer = float(cv2.mean(gray, mask=mask_ring)[0])

    return outer - inner


def ring_stddev(gray: np.ndarray, x: int, y: int, r: int) -> float:
    h, w = gray.shape[:2]
    r_out = int(r * 1.9)
    r_in = int(r * 1.1)
    if x - r_out < 0 or y - r_out < 0 or x + r_out >= w or y + r_out >= h:
        return 1e9
    m = annulus_mask((h, w), x, y, r_in, r_out)
    _, std = cv2.meanStdDev(gray, mask=m)
    return float(std[0, 0])


def edge_ring_score(gray: np.ndarray, x: int, y: int, r: int) -> float:
    h, w = gray.shape[:2]
    if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
        return 0.0

    g = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    n = 72
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = (x + np.cos(angles) * r).astype(np.int32)
    ys = (y + np.sin(angles) * r).astype(np.int32)
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    return float(np.mean(mag[ys, xs]))


def ink_fraction_in_ring(
    green_mask: np.ndarray,
    dark_mask: np.ndarray,
    x: int,
    y: int,
    r: int,
    r_in_mul: float = 1.1,
    r_out_mul: float = 2.0,
) -> float:
    h, w = green_mask.shape[:2]
    r_in = int(r * r_in_mul)
    r_out = int(r * r_out_mul)
    if x - r_out < 0 or y - r_out < 0 or x + r_out >= w or y + r_out >= h:
        return 1.0
    ring = annulus_mask((h, w), x, y, r_in, r_out)
    ink = cv2.bitwise_or(green_mask, dark_mask)
    ink_in_ring = cv2.countNonZero(cv2.bitwise_and(ink, ring))
    ring_area = cv2.countNonZero(ring)
    return float(ink_in_ring) / float(ring_area) if ring_area > 0 else 1.0


def nms_by_distance(dets: List[Det], min_dist_px: int) -> List[Det]:
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: List[Det] = []
    for d in dets:
        if all((d.x - k.x) ** 2 + (d.y - k.y) ** 2 >= min_dist_px**2 for k in kept):
            kept.append(d)
    kept.sort(key=lambda d: (d.x, d.y))
    return kept


def detect_holes(
    bgr: np.ndarray,
    dp: float,
    min_dist: float,
    param1: float,
    param2_list: List[float],
    min_radius: int,
    max_radius: int,
    delta_thresh: float,
    edge_thresh: float,
    ring_std_max: float,
    ink_frac_max: float,
    dark_v_thresh: int,
    green_s_thresh: int,
    nms_min_dist: int,
    max_detections: int,
    ignore_top_frac: float,
) -> List[Det]:
    gray0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, _ = gray0.shape[:2]
    green_mask, dark_mask = build_ink_masks(bgr, dark_v_thresh=dark_v_thresh, green_s_thresh=green_s_thresh)

    gray = gray0.copy()
    if ignore_top_frac > 0:
        y_cut = int(h * ignore_top_frac)
        gray[:y_cut, :] = 255

    g = clahe(gray)
    g = cv2.GaussianBlur(g, (9, 9), 2)

    candidates: List[Tuple[int, int, int]] = []
    for p2 in param2_list:
        circles = cv2.HoughCircles(
            g,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=p2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            continue
        for xf, yf, rf in circles[0]:
            candidates.append((int(round(float(xf))), int(round(float(yf))), int(round(float(rf)))))

    dets: List[Det] = []
    for x, y, r in candidates:
        d = circle_delta(gray0, x, y, r)
        if d < delta_thresh:
            continue

        e = edge_ring_score(gray0, x, y, r)
        if e < edge_thresh:
            continue

        s = ring_stddev(gray0, x, y, r)
        if s > ring_std_max:
            continue

        inkf = ink_fraction_in_ring(green_mask, dark_mask, x, y, r)
        if inkf > ink_frac_max:
            continue

        score = (d * 1.0) + (e * 0.02) - (s * 0.05) - (inkf * 25.0)

        dets.append(
            Det(
                x=x, y=y, r=r,
                score=float(score),
                delta=float(d),
                edge=float(e),
                ring_std=float(s),
                ink_frac=float(inkf),
            )
        )

    dets = nms_by_distance(dets, min_dist_px=nms_min_dist)

    if max_detections > 0:
        dets = sorted(dets, key=lambda d: d.score, reverse=True)[:max_detections]
        dets.sort(key=lambda d: (d.x, d.y))

    return dets


# -----------------------------
# Regular octagon model + scoring
# -----------------------------

def regular_octagon_points(cx: float, cy: float, apothem: float) -> np.ndarray:
    R = apothem / COS_22_5
    pts = []
    for k in range(8):
        ang = math.radians(22.5 + 45.0 * k)
        pts.append([cx + R * math.cos(ang), cy + R * math.sin(ang)])
    return np.array(pts, dtype=np.float32)  # (8,2)


def precompute_unit_octagon_samples(samples_per_edge: int = 20) -> np.ndarray:
    poly = regular_octagon_points(0.0, 0.0, 1.0)  # (8,2)
    out = []
    ts = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False).astype(np.float32)
    for i in range(8):
        a = poly[i]
        b = poly[(i + 1) % 8]
        seg = a[None, :] * (1.0 - ts[:, None]) + b[None, :] * ts[:, None]
        out.append(seg)
    return np.vstack(out).astype(np.float32)


def score_dt_fast(dt: np.ndarray, cx: float, cy: float, apothem: float, base_samples: np.ndarray) -> float:
    H, W = dt.shape[:2]
    pts = base_samples * float(apothem) + np.array([cx, cy], dtype=np.float32)
    xs = np.clip(np.rint(pts[:, 0]).astype(np.int32), 0, W - 1)
    ys = np.clip(np.rint(pts[:, 1]).astype(np.int32), 0, H - 1)
    return float(np.mean(dt[ys, xs]))


def dt_band_roi(
    gray_roi: np.ndarray,
    green_roi: np.ndarray,
    dark_roi: np.ndarray,
    cx_roi: float,
    cy_roi: float,
    a0: float,
    canny1: int,
    canny2: int,
    ink_dilate: int,
    scale_min: float,
    scale_max: float,
    band_inner: float,
    band_outer: float,
) -> np.ndarray:
    g = clahe(gray_roi)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    edges = cv2.Canny(g, canny1, canny2)

    ink = cv2.bitwise_or(green_roi, dark_roi)
    if ink_dilate > 0:
        ink = cv2.dilate(ink, np.ones((ink_dilate, ink_dilate), np.uint8), iterations=1)
    edges[ink > 0] = 0

    R0 = a0 / COS_22_5
    inner = band_inner * (R0 * scale_min)
    outer = band_outer * (R0 * scale_max)

    yy, xx = np.indices(edges.shape, dtype=np.float32)
    rr = np.sqrt((xx - float(cx_roi)) ** 2 + (yy - float(cy_roi)) ** 2)
    band = (rr >= inner) & (rr <= outer)

    edges = np.where(band, edges, 0).astype(np.uint8)

    inv = (edges == 0).astype(np.uint8) * 255
    return cv2.distanceTransform(inv, cv2.DIST_L2, 3)


def refine_cy_and_scale(
    dt: np.ndarray,
    cx: float,
    cy0: float,
    a0: float,
    base_samples: np.ndarray,
    cy_pix: int,
    cy_step: int,
    scale_min: float,
    scale_max: float,
    scale_steps: int,
) -> Tuple[float, float]:
    scales = np.linspace(scale_min, scale_max, max(5, scale_steps)).astype(np.float32)

    best_score = 1e18
    best_cy = cy0
    best_s = 1.0

    for oy in range(-cy_pix, cy_pix + 1, max(1, cy_step)):
        cy = cy0 + oy
        for s in scales:
            sc = score_dt_fast(dt, cx, cy, a0 * float(s), base_samples)
            if sc < best_score:
                best_score = sc
                best_cy = float(cy)
                best_s = float(s)

    fine_scales = np.linspace(
        max(scale_min, best_s - 0.04),
        min(scale_max, best_s + 0.04),
        7,
    ).astype(np.float32)
    for oy in range(-max(1, cy_step), max(1, cy_step) + 1):
        cy = best_cy + oy
        for s in fine_scales:
            sc = score_dt_fast(dt, cx, cy, a0 * float(s), base_samples)
            if sc < best_score:
                best_score = sc
                best_cy = float(cy)
                best_s = float(s)

    return best_cy, best_s


def to_int_contour(poly8: np.ndarray) -> np.ndarray:
    return np.rint(poly8).astype(np.int32).reshape(-1, 1, 2)


# -----------------------------
# Icon extraction (mask) + purple overlay
# -----------------------------
def extract_icon_edges_from_gray_roi(
    roi_gray: np.ndarray,
    mask: np.ndarray,
    pctl: float = 92.0,
) -> np.ndarray:
    """
    Robust emboss edge extractor:
      - CLAHE
      - Sobel gradient magnitude
      - normalize within mask
      - keep top pctl% gradient pixels within mask
    Returns uint8 0/255.
    """
    g = clahe(roi_gray)
    g = cv2.GaussianBlur(g, (5, 5), 0)

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # normalize magnitude to 0..255 using masked percentiles
    vals = mag[mask > 0]
    if vals.size < 50:
        return np.zeros_like(mask)

    p5, p95 = np.percentile(vals, [5, 95])
    denom = max(1e-6, float(p95 - p5))
    magn = ((mag - float(p5)) * (255.0 / denom))
    magn = np.clip(magn, 0, 255).astype(np.uint8)

    valsn = magn[mask > 0]
    thr = int(np.percentile(valsn, pctl))

    edges = np.zeros_like(mask)
    edges[(magn >= thr) & (mask > 0)] = 255

    # connect small gaps a bit, but don't erase thin lines
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    return edges


def overlay_mask_purple(
    debug_bgr_roi: np.ndarray,
    mask: np.ndarray,
    purple_bgr: Tuple[int, int, int],
    alpha: float,
) -> None:
    if cv2.countNonZero(mask) == 0:
        return
    overlay = debug_bgr_roi.copy()
    overlay[mask > 0] = np.array(purple_bgr, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, debug_bgr_roi, 1.0 - alpha, 0, dst=debug_bgr_roi)


def overlay_icon_purple_and_return_mask(
    args,
    debug_bgr: np.ndarray,
    gray: np.ndarray,
    det: Det,
    apothem: float,
    purple_bgr: Tuple[int, int, int],
    alpha: float,
    center_radius_mul: float,
    dilate_px: int,
    pctl_hi: float,
    pctl_lo: float,
    hyst_iter: int,
    core_radius_mul: float,
    core_grow_iter: int,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], Optional[Tuple[float, float]]]:
    """
    Overlays purple and returns (keep_mask, roi_xywh, (cx_roi, cy_roi)) for matching.
    """
    if det.octagon is None:
        return None, None, None

    H, W = gray.shape[:2]
    poly = det.octagon

    x, y, w, h = cv2.boundingRect(poly)
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None, None, None

    roi_gray = gray[y0:y1, x0:x1]
    roi_dbg = debug_bgr[y0:y1, x0:x1]

    # octagon mask in ROI
    mask_oct = np.zeros((y1 - y0, x1 - x0), np.uint8)
    poly_roi = poly.copy()
    poly_roi[:, 0, 0] -= x0
    poly_roi[:, 0, 1] -= y0
    cv2.fillPoly(mask_oct, [poly_roi], 255)

    cx_roi = float(det.x) - x0
    cy_roi = float(det.cy if det.cy is not None else det.y) - y0
    rad = max(10, int(center_radius_mul * float(apothem)))

    mask_center = np.zeros_like(mask_oct)
    cv2.circle(mask_center, (int(round(cx_roi)), int(round(cy_roi))), rad, 255, -1)
    mask = cv2.bitwise_and(mask_oct, mask_center)
    print("mask:", np.count_nonzero(mask))

    keep = extract_icon_edges_from_gray_roi(
        roi_gray,
        mask,
        pctl=args.icon_match_grad_pctl
    )

    overlay_mask_purple(roi_dbg, keep, purple_bgr=purple_bgr, alpha=alpha)
    return keep, (x0, y0, x1 - x0, y1 - y0), (cx_roi, cy_roi)


# -----------------------------
# Icon template loading + matching
# -----------------------------

def rasterize_svg_to_gray(svg_path: str, size: int) -> np.ndarray:
    if cairosvg is None or Image is None:
        raise RuntimeError("SVG support requires cairosvg + pillow. Install: uv add cairosvg pillow")
    png_bytes = cairosvg.svg2png(url=svg_path, output_width=size, output_height=size)
    pil = Image.open(io.BytesIO(png_bytes)).convert("L")
    return np.array(pil, dtype=np.uint8)


def template_from_gray(gray: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Convert grayscale template to edge DT. Returns (dist_transform, edge_count).
    """
    # Normalize: binarize, then edges
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure foreground lines are white-ish; invert if needed
    if float(np.mean(bw)) > 127.0:
        bw = 255 - bw

    edges = cv2.Canny(bw, 40, 120)
    edge_count = int(np.count_nonzero(edges))

    inv = (edges == 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)  # float32
    return dist, edge_count


def load_icon_templates(svg_dir: str, size: int, allowlist: Optional[set[str]] = None, n_bins: int = 8) -> List[IconTemplate]:
    svgs = [os.path.join(svg_dir, fn) for fn in sorted(os.listdir(svg_dir)) if fn.lower().endswith(".svg")]
    templates: List[IconTemplate] = []
    for path in svgs:
        label = os.path.splitext(os.path.basename(path))[0]
        if allowlist is not None and label not in allowlist:
            continue

        g = rasterize_svg_to_gray(path, size=size)
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if float(np.mean(bw)) > 127.0:
            bw = 255 - bw

        t_edges = cv2.Canny(bw, 40, 120)
        t_dt_bank = oriented_dt_bank(t_edges, n_bins=n_bins)

        templates.append(IconTemplate(label=label, edges=t_edges, dt_bank=t_dt_bank))
    return templates


def icon_query_to_canvas(
    keep_mask: np.ndarray,
    cx_roi: float,
    cy_roi: float,
    size: int,
    crop_radius: int,
) -> np.ndarray:
    # Start from your center crop, but then tighten to content bbox.
    h, w = keep_mask.shape[:2]
    cx = int(round(cx_roi))
    cy = int(round(cy_roi))

    x0 = max(0, cx - crop_radius)
    y0 = max(0, cy - crop_radius)
    x1 = min(w, cx + crop_radius)
    y1 = min(h, cy + crop_radius)
    patch = keep_mask[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros((size, size), np.uint8)

    ys, xs = np.where(patch > 0)
    if xs.size < 8:
        return np.zeros((size,size), np.uint8)

    # Tight bbox around content + small padding
    pad = 6
    bx0 = max(0, int(xs.min()) - pad)
    by0 = max(0, int(ys.min()) - pad)
    bx1 = min(patch.shape[1], int(xs.max()) + pad + 1)
    by1 = min(patch.shape[0], int(ys.max()) + pad + 1)
    patch = patch[by0:by1, bx0:bx1]

    ph, pw = patch.shape[:2]
    side = max(ph, pw)
    canvas = np.zeros((side, side), np.uint8)
    oy = (side - ph) // 2
    ox = (side - pw) // 2
    canvas[oy:oy + ph, ox:ox + pw] = patch

    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_NEAREST)


def chamfer_score(query_edges: np.ndarray, template_dist: np.ndarray) -> float:
    ys, xs = np.where(query_edges > 0)
    if xs.size < 25:
        return float("inf")
    vals = template_dist[ys, xs]
    return float(np.mean(vals))


def match_icon(
    args,
    roi_gray: np.ndarray,
    mask: np.ndarray,
    cx_roi: float,
    cy_roi: float,
    apothem: float,
    templates: List[IconTemplate],
    template_size: int,
    crop_radius_mul: float,
    n_bins: int = 8,
) -> Tuple[str, float, float, str, float]:
    # Extract query emboss edges inside mask
    q_edges_roi = extract_icon_edges_from_gray_roi(
        roi_gray,
        mask,
        pctl=float(args.icon_match_grad_pctl),
    )

    crop_radius = max(20, int(float(crop_radius_mul) * float(apothem)))
    q_canvas = icon_query_to_canvas(q_edges_roi, cx_roi, cy_roi, size=template_size, crop_radius=crop_radius)

    print("q_edges_roi:", np.count_nonzero(q_edges_roi), "q_canvas:", np.count_nonzero(q_canvas))
    if int(np.count_nonzero(q_canvas)) < 15:
        return "unknown", 0.0, float("inf"), "unknown", float("inf")

    # Build query edges (already edges), just ensure binary
    q_edges = (q_canvas > 0).astype(np.uint8) * 255

    scores: List[Tuple[float, str]] = []
    for t in templates:
        s = symmetric_oriented_chamfer(q_edges, t.edges, t.dt_bank, n_bins=n_bins)
        scores.append((s, t.label))

    scores.sort(key=lambda x: x[0])
    best_score, best_label = scores[0]
    second_score, second_label = scores[1] if len(scores) > 1 else (float("inf"), "unknown")

    # Confidence: scale to typical chamfer scores; smaller is better
    gap = float(second_score - best_score)
    scale = max(1e-6, 0.25 * (1.0 + best_score))
    conf = float(1.0 / (1.0 + math.exp(-(gap / scale))))

    return best_label, conf, float(best_score), second_label, float(second_score)


# -----------------------------
# Output / debug drawing
# -----------------------------

def contour_to_points(contour: Optional[np.ndarray]) -> Optional[List[List[int]]]:
    if contour is None:
        return None
    pts = contour.reshape(-1, 2)
    return [[int(px), int(py)] for px, py in pts]


def draw_debug_base(bgr: np.ndarray, dets: List[Det]) -> np.ndarray:
    dbg = bgr.copy()
    for i, d in enumerate(dets):
        cv2.circle(dbg, (d.x, d.y), d.r, (0, 0, 255), 3)
        cv2.circle(dbg, (d.x, d.y), 3, (0, 255, 0), -1)
        if d.octagon is not None:
            cv2.polylines(dbg, [d.octagon], True, (255, 0, 0), 3)
        cv2.putText(
            dbg,
            str(i),
            (d.x + 10, d.y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return dbg


def draw_icon_label(dbg: np.ndarray, det: Det) -> None:
    if det.octagon is None:
        return
    if not det.icon_label:
        return

    x, y, w, h = cv2.boundingRect(det.octagon)
    label = det.icon_label
    conf = det.icon_conf if det.icon_conf is not None else 0.0
    text = f"{label} ({conf:.2f})"

    # Place near top-left of octagon bbox, but inside the tile area a bit
    tx = x + 10
    ty = y + 30

    # background for readability
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(dbg, (tx - 4, ty - th - 6), (tx + tw + 4, ty + 6), (255, 255, 255), -1)
    cv2.putText(dbg, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

def edge_orientation_bins(edges: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """
    Given edges (0/255), compute orientation bin [0..n_bins-1] per pixel.
    Non-edge pixels are set to -1.
    """
    g = edges.astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    ang = (np.arctan2(gy, gx) + np.pi)  # [0..2pi]
    bins = (ang * (n_bins / (2.0 * np.pi))).astype(np.int32) % n_bins

    out = np.full(edges.shape, -1, dtype=np.int32)
    out[edges > 0] = bins[edges > 0]
    return out


def oriented_dt_bank(edges: np.ndarray, n_bins: int = 8) -> List[np.ndarray]:
    """
    For each orientation bin, compute distance transform to nearest edge pixel in that bin.
    Returns list length n_bins of float32 DT images.
    """
    bins = edge_orientation_bins(edges, n_bins=n_bins)
    banks: List[np.ndarray] = []
    for b in range(n_bins):
        m = (bins == b).astype(np.uint8) * 255
        inv = (m == 0).astype(np.uint8) * 255
        dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        banks.append(dt)
    return banks

def oriented_chamfer_score(
    q_edges: np.ndarray,
    q_bins: np.ndarray,
    t_dt_bank: List[np.ndarray],
) -> float:
    ys, xs = np.where(q_edges > 0)
    if xs.size < 30:
        return float("inf")
    # sample DT of matching orientation bin
    vals = []
    for y, x in zip(ys, xs):
        b = q_bins[y, x]
        if b < 0:
            continue
        vals.append(t_dt_bank[b][y, x])
    if not vals:
        return float("inf")
    return float(np.mean(vals))


def symmetric_oriented_chamfer(
    q_edges: np.ndarray,
    t_edges: np.ndarray,
    t_dt_bank: List[np.ndarray],
    n_bins: int = 8,
) -> float:
    """
    Score = oriented chamfer(query -> template) + oriented chamfer(template -> query)
    """
    q_bins = edge_orientation_bins(q_edges, n_bins=n_bins)
    # forward
    fwd = oriented_chamfer_score(q_edges, q_bins, t_dt_bank)

    # backward: build query dt bank once
    q_dt_bank = oriented_dt_bank(q_edges, n_bins=n_bins)
    t_bins = edge_orientation_bins(t_edges, n_bins=n_bins)
    bwd = oriented_chamfer_score(t_edges, t_bins, q_dt_bank)

    return float(fwd + bwd)

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Detect tag holes + rigid octagon + purple icon overlay + SVG matching.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_json", default="detections.json")
    ap.add_argument("--out_debug", default="debug_overlay.png")

    # Hole detect params
    ap.add_argument("--dp", type=float, default=1.2)
    ap.add_argument("--min_dist", type=float, default=140.0)
    ap.add_argument("--param1", type=float, default=100.0)
    ap.add_argument("--param2_list", default="18,22,26,30")
    ap.add_argument("--min_radius", type=int, default=24)
    ap.add_argument("--max_radius", type=int, default=70)

    ap.add_argument("--delta_thresh", type=float, default=12.0)
    ap.add_argument("--edge_thresh", type=float, default=16.0)
    ap.add_argument("--ring_std_max", type=float, default=40.0)

    ap.add_argument("--ink_frac_max", type=float, default=0.025)
    ap.add_argument("--dark_v_thresh", type=int, default=80)
    ap.add_argument("--green_s_thresh", type=int, default=70)

    ap.add_argument("--nms_min_dist", type=int, default=190)
    ap.add_argument("--max_detections", type=int, default=0)
    ap.add_argument("--ignore_top_frac", type=float, default=0.03)

    # Octagon model: apothem = beta * r, center_y = hole_y + dy_mul * r, center_x = hole_x
    ap.add_argument("--oct_beta", type=float, default=5.0)
    ap.add_argument("--oct_center_dy_mul", type=float, default=4.6)

    ap.add_argument("--oct_canny1", type=int, default=30)
    ap.add_argument("--oct_canny2", type=int, default=90)
    ap.add_argument("--oct_ink_dilate", type=int, default=5)

    ap.add_argument("--oct_band_inner", type=float, default=0.92)
    ap.add_argument("--oct_band_outer", type=float, default=1.10)
    ap.add_argument("--oct_pad_mul", type=float, default=6.0)

    ap.add_argument("--oct_refine_cy_pix", type=int, default=14)
    ap.add_argument("--oct_refine_cy_step", type=int, default=2)

    ap.add_argument("--oct_scale_min", type=float, default=0.96)
    ap.add_argument("--oct_scale_max", type=float, default=1.12)
    ap.add_argument("--oct_scale_steps", type=int, default=11)
    ap.add_argument("--oct_samples_per_edge", type=int, default=20)
    ap.add_argument(
        "--oct_hole_gap_mul",
        type=float,
        default=0.70,
        help="Gap from top of hole to top octagon vertex, in units of hole radius r.",
    )

    # Icon overlay params
    ap.add_argument("--icon_alpha", type=float, default=0.65)
    ap.add_argument("--icon_center_radius_mul", type=float, default=0.62)
    ap.add_argument("--icon_dilate_px", type=int, default=3)

    ap.add_argument("--icon_pctl_hi", type=float, default=90.0)
    ap.add_argument("--icon_pctl_lo", type=float, default=75.0)
    ap.add_argument("--icon_hyst_iter", type=int, default=3)

    ap.add_argument("--icon_core_radius_mul", type=float, default=0.55)
    ap.add_argument("--icon_core_grow_iter", type=int, default=18)

    # Icon matching
    ap.add_argument("--icon_library_dir", default=None,
                    help="Directory containing reference .svg icons. Filename stem becomes label.")
    ap.add_argument("--icon_template_size", type=int, default=256)
    ap.add_argument("--icon_crop_radius_mul", type=float, default=0.70,
                    help="Crop radius (around icon center) as fraction of apothem for matching.")
    ap.add_argument("--icon_match_min_conf", type=float, default=0.0,
                    help="If confidence below this, label becomes 'unknown'.")
    ap.add_argument(
        "--icon_allowlist",
        default=None,
        help="Comma-separated list of icon labels (filename stems) to consider. "
             "Example: aws-lambda,aws-dynamodb,aws-users",
    )
    ap.add_argument("--icon_match_center_radius_mul", type=float, default=0.85)    
    ap.add_argument("--icon_match_canny1", type=int, default=10)
    ap.add_argument("--icon_match_canny2", type=int, default=40)

    ap.add_argument("--icon_match_grad_pctl", type=float, default=92.0,
                    help="Percentile threshold on gradient magnitude for query emboss edges.")

    args = ap.parse_args()

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    param2_list = [float(x.strip()) for x in args.param2_list.split(",") if x.strip()]

    templates: List[IconTemplate] = []
    if args.icon_library_dir:
        if not os.path.isdir(args.icon_library_dir):
            raise SystemExit(f"--icon_library_dir is not a directory: {args.icon_library_dir}")
        allow = None
        if args.icon_allowlist:
            allow = {s.strip() for s in args.icon_allowlist.split(",") if s.strip()}
        templates = load_icon_templates(args.icon_library_dir, size=args.icon_template_size, allowlist=allow)
        if not templates:
            raise SystemExit(f"No .svg files found in {args.icon_library_dir}")

    dets = detect_holes(
        bgr=bgr,
        dp=args.dp,
        min_dist=args.min_dist,
        param1=args.param1,
        param2_list=param2_list,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        delta_thresh=args.delta_thresh,
        edge_thresh=args.edge_thresh,
        ring_std_max=args.ring_std_max,
        ink_frac_max=args.ink_frac_max,
        dark_v_thresh=args.dark_v_thresh,
        green_s_thresh=args.green_s_thresh,
        nms_min_dist=args.nms_min_dist,
        max_detections=args.max_detections,
        ignore_top_frac=args.ignore_top_frac,
    )

    green_mask, dark_mask = build_ink_masks(bgr, args.dark_v_thresh, args.green_s_thresh)
    base_samples = precompute_unit_octagon_samples(samples_per_edge=args.oct_samples_per_edge)

    # Build octagons
    for d in dets:
        cx = float(d.x)  # hole is exact horizontal center
        a0 = float(args.oct_beta) * float(d.r)
        R0 = a0 / COS_22_5

        gap_px = float(args.oct_hole_gap_mul) * float(d.r)

        # Place octagon center so the top vertex is 'gap_px' above the top of the hole
        # hole top = y - r ; top vertex = cy - R0
        cy0 = (float(d.y) - float(d.r)) + R0 - gap_px

        pad = int(float(args.oct_pad_mul) * float(d.r))
        x0 = max(0, int(cx) - pad)
        x1 = min(W, int(cx) + pad)
        y0 = max(0, int(cy0) - pad)
        y1 = min(H, int(cy0) + pad)

        roi_gray = gray[y0:y1, x0:x1]
        roi_green = green_mask[y0:y1, x0:x1]
        roi_dark = dark_mask[y0:y1, x0:x1]
        if roi_gray.size == 0:
            continue

        cx_roi = cx - x0
        cy_roi0 = cy0 - y0

        dt = dt_band_roi(
            gray_roi=roi_gray,
            green_roi=roi_green,
            dark_roi=roi_dark,
            cx_roi=cx_roi,
            cy_roi=cy_roi0,
            a0=a0,
            canny1=args.oct_canny1,
            canny2=args.oct_canny2,
            ink_dilate=args.oct_ink_dilate,
            scale_min=args.oct_scale_min,
            scale_max=args.oct_scale_max,
            band_inner=args.oct_band_inner,
            band_outer=args.oct_band_outer,
        )

        best_cy_roi, best_s = refine_cy_and_scale(
            dt=dt,
            cx=cx_roi,
            cy0=cy_roi0,
            a0=a0,
            base_samples=base_samples,
            cy_pix=args.oct_refine_cy_pix,
            cy_step=args.oct_refine_cy_step,
            scale_min=args.oct_scale_min,
            scale_max=args.oct_scale_max,
            scale_steps=args.oct_scale_steps,
        )

        best_cy = best_cy_roi + y0
        poly = regular_octagon_points(cx, best_cy, a0 * best_s)

        d.octagon = to_int_contour(poly)
        d.cy = float(best_cy)
        d.scale = float(best_s)

    dbg = draw_debug_base(bgr, dets)

    # Overlay icon + match
    PURPLE_BGR = (255, 0, 255)

    for d in dets:
        if d.octagon is None or d.scale is None or d.cy is None:
            continue

        apothem = float(args.oct_beta) * float(d.r) * float(d.scale)

        keep_mask, roi_xywh, center = overlay_icon_purple_and_return_mask(
            args,
            debug_bgr=dbg,
            gray=gray,
            det=d,
            apothem=apothem,
            purple_bgr=PURPLE_BGR,
            alpha=float(args.icon_alpha),
            center_radius_mul=float(args.icon_center_radius_mul),
            dilate_px=int(args.icon_dilate_px),
            pctl_hi=float(args.icon_pctl_hi),
            pctl_lo=float(args.icon_pctl_lo),
            hyst_iter=int(args.icon_hyst_iter),
            core_radius_mul=float(args.icon_core_radius_mul),
            core_grow_iter=int(args.icon_core_grow_iter),
        )

        if templates and roi_xywh is not None and center is not None:
            x0, y0, rw, rh = roi_xywh
            roi_gray = gray[y0:y0 + rh, x0:x0 + rw]

            cx_roi, cy_roi = center

            # Rebuild the SAME mask used for icon extraction:
            # octagon mask ∩ center circle mask
            mask_oct = np.zeros((rh, rw), np.uint8)
            poly_roi = d.octagon.copy()
            poly_roi[:, 0, 0] -= x0
            poly_roi[:, 0, 1] -= y0
            cv2.fillPoly(mask_oct, [poly_roi], 255)

            rad_match = max(10, int(float(args.icon_match_center_radius_mul) * float(apothem)))
            mask_center = np.zeros_like(mask_oct)
            cv2.circle(mask_center, (int(round(cx_roi)), int(round(cy_roi))), rad_match, 255, -1)
            mask = cv2.bitwise_and(mask_oct, mask_center)

            label, conf, score, label2, score2 = match_icon(
                args,
                roi_gray=roi_gray,
                mask=mask,
                cx_roi=cx_roi,
                cy_roi=cy_roi,
                apothem=apothem,
                templates=templates,
                template_size=int(args.icon_template_size),
                crop_radius_mul=float(args.icon_crop_radius_mul),
                n_bins=8,
            )

            if conf < float(args.icon_match_min_conf):
                label = "unknown"

            d.icon_label = label
            d.icon_conf = conf
            d.icon_score = score
            d.icon_label2 = label2
            d.icon_score2 = score2

            draw_icon_label(dbg, d)

    out = []
    for d in dets:
        item = {
            "circle": {"x": d.x, "y": d.y, "r": d.r},
            "octagon": contour_to_points(d.octagon),
            "model": {
                "cx": float(d.x),
                "cy": d.cy,
                "scale": d.scale,
                "oct_beta": float(args.oct_beta),
                "oct_center_dy_mul": float(args.oct_center_dy_mul),
            },
            "metrics": {
                "score": d.score,
                "delta": d.delta,
                "edge": d.edge,
                "ring_std": d.ring_std,
                "ink_frac": d.ink_frac,
            },
        }
        if d.icon_label is not None:
            item["icon_match"] = {
                "label": d.icon_label,
                "confidence": d.icon_conf,
                "score": d.icon_score,
                "second_label": d.icon_label2,
                "second_score": d.icon_score2,
            }
        out.append(item)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    cv2.imwrite(args.out_debug, dbg)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
