from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Small utilities
def imread_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def find_file_with_ext(stem_path: str, exts=(".png", ".jpg", ".jpeg")) -> str:
    for e in exts:
        p = stem_path + e
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Missing file: {stem_path}{{{','.join(exts)}}}")

# Matching visualization

def detect_and_match(img1_bgr: np.ndarray, img2_bgr: np.ndarray, feature: str = "SIFT",
                     max_features: int = 1200, ratio: float = 0.75) -> Tuple[int, int, np.ndarray]:
    """Return (good_matches, inliers, vis_image)."""
    g1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
    if feature.upper() == "ORB":
        det = cv2.ORB_create(nfeatures=max_features)
        norm = cv2.NORM_HAMMING
    else:
        det = cv2.SIFT_create(nfeatures=max_features)
        norm = cv2.NORM_L2
    k1, d1 = det.detectAndCompute(g1, None)
    k2, d2 = det.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        vis = np.hstack([img1_bgr, img2_bgr])
        return 0, 0, vis
    bf = cv2.BFMatcher(norm)
    knn =bf.knnMatch(d1, d2, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    # Mutual cross-check (simple): keep matches that are best in both directions
    knn2 = bf.knnMatch(d2, d1, k=1)
    best21 ={m.queryIdx: m.trainIdx for [m] in knn2}
    good_cc =[m for m in good if best21.get(m.trainIdx, -1) == m.queryIdx]
    if len(good_cc) < 8:
        vis =cv2.drawMatches(img1_bgr, k1, img2_bgr, k2, good_cc, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return len(good_cc), 0, vis
    pts1 = np.float32([k1[m.queryIdx].pt for m in good_cc])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good_cc])
    H, inl = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    inliers = int(inl.sum()) if inl is not None else 0

    # Draw only inlier matches for clarity
    inlier_matches = [m for m, keep in zip(good_cc, inl.ravel().tolist() if inl is not None else []) if keep]
    vis = cv2.drawMatches(img1_bgr, k1, img2_bgr, k2, inlier_matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good_cc), inliers, vis

# Figure generators

def fig_matches_orb_vs_sift(data_dir: str, out_dir: str) -> str:
    p1 = os.path.join(data_dir, "STB_0032.JPG")
    p2 = os.path.join(data_dir, "STC_0033.JPG")
    img1 = imread_bgr(p1)
    img2 = imread_bgr(p2)
    # ORB
    g_orb, i_orb, vis_orb = detect_and_match(img1, img2, feature="ORB", max_features=1000, ratio=0.80)
    # SIFT
    g_sift, i_sift, vis_sift = detect_and_match(img1, img2, feature="SIFT", max_features=1200, ratio=0.75)
    # Stack
    vis_orb = cv2.putText(vis_orb, f"ORB (good={g_orb}, inliers={i_orb})", (20, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    vis_sift = cv2.putText(vis_sift, f"SIFT (good={g_sift}, inliers={i_sift})", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    # Make same width
    w = max(vis_orb.shape[1], vis_sift.shape[1])
    def pad_to(img, w):
        if img.shape[1] == w:
            return img
        pad = w - img.shape[1]
        return cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    panel = np.vstack([pad_to(vis_orb, w), pad_to(vis_sift, w)])
    out_path = os.path.join(out_dir, "fig_matches_orb_vs_sift.png")
    cv2.imwrite(out_path, panel)
    return out_path

def fig_inliers_per_pair(results_dir: str, out_dir: str) -> str:
    orb_meta = json.load(open(find_file_with_ext(os.path.join(results_dir, "orb_seammb_meta"), exts=(".json",))))
    sift_meta = json.load(open(find_file_with_ext(os.path.join(results_dir, "sift_seammb_meta"), exts=(".json",))))
    orb = orb_meta.get("inliers_counts", [])
    sift = sift_meta.get("inliers_counts", [])
    x = np.arange(1, max(len(orb), len(sift)) + 1)
    plt.figure(figsize=(6.2, 3.2))
    if orb:
        plt.plot(np.arange(1, len(orb) + 1), orb, marker='o', label='ORB')
    if sift:
        plt.plot(np.arange(1, len(sift) + 1), sift, marker='o', label='SIFT')
    plt.title("RANSAC Inliers per Adjacent Pair (Higher is Better)")
    plt.xlabel("Adjacent pair index (k → k+1)")
    plt.ylabel("Inliers")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = os.path.join(out_dir, "fig_inliers_per_pair.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

def fig_seam_energy(results_dir: str, out_dir: str) -> str:
    csv_path = os.path.join(results_dir, "metrics_summary_with_valid_ratio.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(results_dir, "metrics_summary.csv")
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(6.2, 3.2))
    plt.bar(df["Setting"], df["Seam Energy"])
    plt.title("Seam Energy Comparison (Lower is Better)")
    plt.ylabel("Seam Energy")
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_seam_energy.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path
    
def _best_diff_patch(a: np.ndarray, b: np.ndarray, patch: int = 220, stride: int = 40) -> Tuple[int, int, np.ndarray]:
    """Find a patch with the largest mean abs difference (rough but effective)."""
    h, w = a.shape[:2]
    h2, w2 = b.shape[:2]
    h = min(h, h2)
    w = min(w, w2)
    a = a[:h, :w]
    b = b[:h, :w]
    gray = cv2.cvtColor(cv2.absdiff(a, b), cv2.COLOR_BGR2GRAY).astype(np.float32)
    best = (-1.0, 0, 0)
    for y in range(0, max(1, h - patch), stride):
        for x in range(0, max(1, w - patch), stride):
            m = float(gray[y:y+patch, x:x+patch].mean())
            if m > best[0]:
                best = (m, y, x)
    _, y, x = best
    return y, x, gray

def fig_zoom_sift_blend_diff(results_dir: str, out_dir: str) -> str:
    a = imread_bgr(find_file_with_ext(os.path.join(results_dir, "sift_feather_pano")))
    b = imread_bgr(find_file_with_ext(os.path.join(results_dir, "sift_seammb_pano")))
    y, x, gray = _best_diff_patch(a, b)
    patch = 240
    A = a[y:y+patch, x:x+patch]
    B = b[y:y+patch, x:x+patch]
    D = cv2.absdiff(A, B)
    Dg = cv2.cvtColor(D, cv2.COLOR_BGR2GRAY)
    # Normalize diff for visualization
    Dn = cv2.normalize(Dg, None, 0, 255, cv2.NORM_MINMAX)
    Dn = cv2.applyColorMap(Dn.astype(np.uint8), cv2.COLORMAP_MAGMA)
    # Compose 3 panels
    gap = 10
    h = patch
    panel = np.full((h, patch*3 + gap*2, 3), 255, np.uint8)
    panel[:, 0:patch] = A
    panel[:, patch+gap:2*patch+gap] = B
    panel[:, 2*patch+2*gap:3*patch+2*gap] = Dn
    cv2.putText(panel, "Feather", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(panel, "Seam+MB", (patch+gap+10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(panel, "|Diff|", (2*patch+2*gap+10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
    out_path = os.path.join(out_dir, "fig_zoom_sift_blend_diff.png")
    cv2.imwrite(out_path, panel)
    return out_path

def fig_orb_failure(results_dir: str, out_dir: str) -> str:
    pano = imread_bgr(find_file_with_ext(os.path.join(results_dir, "orb_seammb_pano")))
    mask = cv2.imread(find_file_with_ext(os.path.join(results_dir, "orb_seammb_mask")), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError("orb_seammb_mask")
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Resize to same height
    H = min(pano.shape[0], mask_vis.shape[0])
    def resize_h(img, H):
        h,w = img.shape[:2]
        s = H / float(h)
        return cv2.resize(img, (int(w*s), H))
    pano_r = resize_h(pano, H)
    mask_r = resize_h(mask_vis, H)
    gap = 10
    out = np.full((H, pano_r.shape[1] + gap + mask_r.shape[1], 3), 0, np.uint8)
    out[:, 0:pano_r.shape[1]] = pano_r
    out[:, pano_r.shape[1]+gap:pano_r.shape[1]+gap+mask_r.shape[1]] = mask_r
    cv2.putText(out, "ORB panorama", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(out, "Valid mask", (pano_r.shape[1]+gap+20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
    out_path = os.path.join(out_dir, "fig_orb_failure.png")
    cv2.imwrite(out_path, out)
    return out_path

def fig_orb_blend_diff(results_dir: str, out_dir: str) -> str:
    a = imread_bgr(find_file_with_ext(os.path.join(results_dir, "orb_feather_pano")))
    b = imread_bgr(find_file_with_ext(os.path.join(results_dir, "orb_seammb_pano")))
    D = cv2.absdiff(a, b)
    Dg = cv2.cvtColor(D, cv2.COLOR_BGR2GRAY)
    Dn = cv2.normalize(Dg, None, 0, 255, cv2.NORM_MINMAX)
    Dn = cv2.applyColorMap(Dn.astype(np.uint8), cv2.COLORMAP_MAGMA)
    # stack
    w = max(a.shape[1], b.shape[1], Dn.shape[1])
    def padw(img):
        if img.shape[1]==w: return img
        return cv2.copyMakeBorder(img, 0, 0, 0, w-img.shape[1], cv2.BORDER_CONSTANT, value=(0,0,0))
    panel = np.vstack([padw(a), padw(b), padw(Dn)])
    cv2.putText(panel, "ORB Feather", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(panel, "ORB Seam+MB", (20, a.shape[0]+40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(panel, "|Diff|", (20, a.shape[0]+b.shape[0]+40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
    out_path = os.path.join(out_dir, "fig_orb_blend_diff.png")
    cv2.imwrite(out_path, panel)
    return out_path

def generate_all_figures(data_dir: str, results_dir: str, out_dir: str) -> None:
    ensure_dir(out_dir)
    fig_matches_orb_vs_sift(data_dir, out_dir)
    fig_inliers_per_pair(results_dir, out_dir)
    fig_seam_energy(results_dir, out_dir)
    fig_zoom_sift_blend_diff(results_dir, out_dir)
    fig_orb_failure(results_dir, out_dir)
    fig_orb_blend_diff(results_dir, out_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_dir", default="figures")
    args = ap.parse_args()
    generate_all_figures(args.data_dir, args.results_dir, args.out_dir)
    print("[OK] Figures saved to", args.out_dir)

if __name__ == "__main__":
    main()
