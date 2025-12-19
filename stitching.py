from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
import os, json, math, time
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def resize_image(img: np.ndarray, scale: float) -> np.ndarray:
    if scale==1.0:
        return img
    h,w=img.shape[:2]
    return cv2.resize(img, (int(w*scale),int(h*scale)),interpolation=cv2.INTER_AREA)

def detect_and_describe(img:np.ndarray,method:str,max_features:int)->Tuple[List[cv2.KeyPoint],np.ndarray,int]:
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    method=method.upper()
    if method=="ORB":
        orb=cv2.ORB_create(nfeatures=max_features,scaleFactor=1.2,nlevels=8,edgeThreshold=31,fastThreshold=20)
        kps,desc=orb.detectAndCompute(gray, None)
        norm=cv2.NORM_HAMMING
    elif method=="SIFT":
        sift=cv2.SIFT_create(nfeatures=max_features)
        kps,desc = sift.detectAndCompute(gray, None)
        norm=cv2.NORM_L2
    else:
        raise ValueError("method must be ORB or SIFT")
    return kps, desc, norm

def match_descriptors(desc1:np.ndarray,desc2:np.ndarray,norm:int,ratio:float=0.8,cross_check:bool=True):
    """KNN + Lowe ratio test (+ optional mutual check)."""
    if desc1 is None or desc2 is None or len(desc1)==0 or len(desc2)==0:
        return []
    bf=cv2.BFMatcher(norm, crossCheck=False)
    m12=bf.knnMatch(desc1, desc2, k=2)
    good12=[]
    for m,n in m12:
        if m.distance<ratio*n.distance:
            good12.append(m)
    if not cross_check:
        return good12
    # mutual check
    m21=bf.knnMatch(desc2, desc1, k=2)
    good21={}
    for m,n in m21:
        if m.distance<ratio*n.distance:
            good21[m.queryIdx]=m.trainIdx
    mutual=[]
    for m in good12:
        if m.trainIdx in good21 and good21[m.trainIdx]==m.queryIdx:
            mutual.append(m)
    return mutual

def compute_homography(kps1,kps2,matches,ransac_thresh=3.0,max_iters=800,confidence=0.999):
    if len(matches)<4:
        return None, None
    pts1=np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2=np.float32([kps2[m.trainIdx].pt for m in matches])
    H,mask=cv2.findHomography(
        pts1, pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=max_iters,
        confidence=confidence
    )
    if mask is None:
        return H, None
    return H.astype(np.float32), mask.ravel().astype(bool)

def compute_global_transforms(H_pair: List[np.ndarray]) -> Tuple[List[np.ndarray], int]:
    """
    H_pair[i] maps points from image i -> image i+1.
    Build transforms that map each image into the middle reference frame to reduce drift.
    """
    N=len(H_pair)+1
    T_to0 =[np.eye(3,dtype=np.float32)]
    for i in range(1,N):
        T_to0.append(T_to0[i - 1] @ np.linalg.inv(H_pair[i -1]))
    ref=N // 2
    T_ref_inv=np.linalg.inv(T_to0[ref])
    T_to_ref=[T_ref_inv @ T for T in T_to0]
    return T_to_ref, ref


def compute_canvas(transforms: List[np.ndarray], img_shape) -> Tuple[int, int, np.ndarray]:
    h,w=img_shape[:2]
    corners=np.array([[0, 0, 1],
                        [w,0, 1],
                        [w, h,1],
                        [0, h, 1]],dtype=np.float32).T
    all_pts =[]
    for H in transforms:
        pts =H @ corners
        pts= pts[:2] / pts[2]
        all_pts.append(pts.T)
    all_pts =np.vstack(all_pts)
    min_x, min_y =np.min(all_pts, axis=0)
    max_x, max_y =np.max(all_pts, axis=0)
    tx, ty= -min_x, -min_y
    out_w= int(math.ceil(max_x - min_x))
    out_h= int(math.ceil(max_y - min_y))
    T= np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]], dtype=np.float32)
    return out_h, out_w, T

def warp_all_images(imgs, masks,transforms_to_ref):
    out_h, out_w, T=compute_canvas(transforms_to_ref, imgs[0].shape)
    wimgs, wmasks =[],[]
    for img, m, H in zip(imgs, masks, transforms_to_ref):
        Ht =T @ H
        wimg= cv2.warpPerspective(img, Ht, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        wmask= cv2.warpPerspective(m, Ht, (out_w, out_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        wmask =(wmask>0).astype(np.uint8)*255
        wimgs.append(wimg)
        wmasks.append(wmask)
    return wimgs, wmasks, (out_h, out_w)

def grad_mag(gray: np.ndarray) -> np.ndarray:
    gx= cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    gy= cv2.Sobel(gray,cv2.CV_32F,0,1, ksize=3)
    return cv2.magnitude(gx, gy)

def estimate_gain(img, ref, overlap_mask, use_gradient_weight=True) -> float:
    """Scalar gain α so that α*img matches ref in the overlap region."""
    ig =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rg= cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m= overlap_mask > 0
    if np.count_nonzero(m)< 500:
        return 1.0
    w =grad_mag(rg) if use_gradient_weight else np.ones_like(rg)
    w =w[m]
    a=ig[m]
    b =rg[m]
    alpha =float(np.sum(w* a *b) /(np.sum(w * a* a) +1e-6))
    return float(np.clip(alpha, 0.5, 2.0))

def apply_gain(img, alpha: float)->np.ndarray:
    return np.clip(img.astype(np.float32) *alpha, 0, 255).astype(np.uint8)

def exposure_compensate(wimgs, wmasks, ref_idx: int):
    """Chain gains from the reference outwards (left and right)."""
    N =len(wimgs)
    corrected =[None] *N
    corrected[ref_idx] =wimgs[ref_idx]
    gains = [1.0] *N
    for i in range(ref_idx -1, -1, -1):
        ov=((wmasks[i] > 0) & (wmasks[i + 1] > 0)).astype(np.uint8) * 255
        a =estimate_gain(wimgs[i], corrected[i +1],ov, use_gradient_weight=True)
        gains[i] =gains[i + 1] *a
        corrected[i] = apply_gain(wimgs[i],a)
    for i in range(ref_idx + 1, N):
        ov = ((wmasks[i] > 0) & (wmasks[i - 1] > 0)).astype(np.uint8) * 255
        a = estimate_gain(wimgs[i], corrected[i - 1], ov, use_gradient_weight=True)
        gains[i] = gains[i - 1] * a
        corrected[i] = apply_gain(wimgs[i], a)
    return corrected, gains

def distance_transform_weights(mask: np.ndarray, downscale: float = 0.3, mask_size: int = 3) -> np.ndarray:
    """Distance-transform weights (computed on a downsampled mask for speed)."""
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    if downscale < 1.0:
        hs =max(1, int(h * downscale))
        ws = max(1, int(w * downscale))
        m_small =cv2.resize(m, (ws, hs), interpolation=cv2.INTER_NEAREST)
        dist_small = cv2.distanceTransform(m_small, cv2.DIST_L2, mask_size)
        dist = cv2.resize(dist_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    else:
        dist =cv2.distanceTransform(m, cv2.DIST_L2, mask_size).astype(np.float32)
    dist[m == 0] = 0.0
    return dist

def blend_feather(imgs, masks, dt_downscale: float = 0.3):
    """Feather blending (multi-image) using distance-transform weights."""
    h, w =imgs[0].shape[:2]
    acc = np.zeros((h, w, 3), np.float32)
    wsum =np.zeros((h, w), np.float32)
    for img, m in zip(imgs, masks):
        wt = distance_transform_weights(m, downscale=dt_downscale)
        acc += img.astype(np.float32) * wt[..., None]
        wsum +=wt
    out =acc / (wsum[..., None] + 1e-6)
    out =np.clip(out, 0, 255).astype(np.uint8)
    out_mask= (wsum > 0).astype(np.uint8) * 255
    return out, out_mask


def center_seam_energy(pano_img, pano_mask, new_img, new_mask, lamb_c=1.0, lamb_g=0.2) -> float:
    overlap = (pano_mask > 0) & (new_mask > 0)
    if np.count_nonzero(overlap) < 1000:
        return 0.0
    ys, xs = np.where(overlap)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    A = pano_img[y0:y1 + 1, x0:x1 + 1]
    B = new_img[y0:y1 + 1, x0:x1 + 1]
    ov = overlap[y0:y1 + 1, x0:x1 + 1]
    Ag = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY).astype(np.float32)
    Bg = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY).astype(np.float32)
    c = np.abs(Ag - Bg)
    g = np.abs(grad_mag(Ag) - grad_mag(Bg))
    cost = lamb_c * c + lamb_g * g
    cost = cost + (~ov) * 1e6
    Hc, _= cost.shape
    seam_cost = 0.0
    n = 0
    for i in range(Hc):
        cols = np.where(ov[i])[0]
        if cols.size == 0:
            continue
        j = int(cols[cols.size // 2])
        seam_cost += float(cost[i, j])
        n += 1
    return seam_cost / (max(n, 1) + 1e-6)

def blend_feather_incremental(imgs, masks, order, dt_downscale: float = 0.3, lamb_c: float = 1.0, lamb_g: float = 0.2):
    """Incremental feather blending (also computes a seam-energy baseline)."""
    idx0 = order[0]
    pano = imgs[idx0].copy()
    pano_mask = masks[idx0].copy()
    seam_energies = []
    for idx in order[1:]:
        new = imgs[idx]
        new_mask = masks[idx]
        seam_energies.append(center_seam_energy(pano, pano_mask, new, new_mask, lamb_c=lamb_c, lamb_g=lamb_g))
        wA = distance_transform_weights(pano_mask, downscale=dt_downscale)
        wB = distance_transform_weights(new_mask, downscale=dt_downscale)
        acc = pano.astype(np.float32) * wA[..., None] + new.astype(np.float32) * wB[..., None]
        wsum = wA + wB
        pano = (acc / (wsum[..., None] + 1e-6))
        pano = np.clip(pano, 0, 255).astype(np.uint8)
        pano_mask = ((pano_mask > 0) | (new_mask > 0)).astype(np.uint8) * 255
    return pano, pano_mask, float(np.mean(seam_energies)) if seam_energies else 0.0
def build_gaussian_pyramid(img, levels: int):
    gp = [img]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        gp.append(img)
    return gp

def build_laplacian_pyramid(img, levels: int):
    gp = build_gaussian_pyramid(img, levels)
    lp = []
    for i in range(levels):
        size = (gp[i].shape[1], gp[i].shape[0])
        ge = cv2.pyrUp(gp[i + 1], dstsize=size)
        lp.append(gp[i] - ge)
    lp.append(gp[-1])
    return lp

def multiband_blend_two(A, B, maskB, levels: int = 4):
    """Two-image multi-band blending."""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    m = maskB.astype(np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    m = np.clip(m, 0, 1)
    LA = build_laplacian_pyramid(A, levels)
    LB = build_laplacian_pyramid(B, levels)
    GM = build_gaussian_pyramid(m, levels)
    LS = []
    for la, lb, gm in zip(LA, LB, GM):
        gm3 = gm[..., None]
        LS.append(la * (1 - gm3) + lb * gm3)
    out = LS[-1]
    for i in range(levels - 1, -1, -1):
        size = (LS[i].shape[1], LS[i].shape[0])
        out = cv2.pyrUp(out, dstsize=size)
        out = out + LS[i]
    return np.clip(out, 0, 255).astype(np.uint8)

def dp_vertical_seam(cost: np.ndarray):
    """Minimum-cost top-to-bottom seam via DP (educational)."""
    H, W = cost.shape
    dp = np.zeros_like(cost, dtype=np.float32)
    back = np.zeros_like(cost, dtype=np.int32)
    dp[0] = cost[0]
    back[0] = -1
    for i in range(1, H):
        for j in range(W):
            j0 = max(j - 1, 0)
            j1 = min(j + 1, W - 1)
            prev = dp[i - 1, j0:j1 + 1]
            idx = int(np.argmin(prev))
            dp[i, j] = cost[i, j] + prev[idx]
            back[i, j] = j0 + idx
    seam = np.zeros(H, dtype=np.int32)
    seam[H - 1] = int(np.argmin(dp[H - 1]))
    for i in range(H - 2, -1, -1):
        seam[i] = back[i + 1, seam[i + 1]]
    return seam, float(dp[H - 1, seam[H - 1]])

def seam_mask(pano_img, pano_mask, new_img, new_mask, orientation="right", lamb_c=1.0, lamb_g=0.2):
    """Compute a binary-ish mask for the new image using an optimal seam in overlap."""
    overlap = (pano_mask > 0) & (new_mask > 0)
    if np.count_nonzero(overlap) < 1000:
        return (new_mask > 0).astype(np.float32), 0.0

    ys, xs = np.where(overlap)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    A = pano_img[y0:y1 + 1, x0:x1 + 1]
    B = new_img[y0:y1 + 1, x0:x1 + 1]
    ov = overlap[y0:y1 + 1, x0:x1 + 1]
    Ag = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY).astype(np.float32)
    Bg = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY).astype(np.float32)
    c = np.abs(Ag - Bg)
    g = np.abs(grad_mag(Ag) - grad_mag(Bg))
    cost = lamb_c * c + lamb_g * g
    cost = cost + (~ov) * 1e6
    seam, seam_cost = dp_vertical_seam(cost)
    Hc, Wc = cost.shape
    local = np.zeros((Hc, Wc), np.float32)
    for i in range(Hc):
        j = seam[i]
        if orientation == "right":
            local[i, j:] = 1.0
        else:
            local[i, :j + 1] = 1.0
    mask_new = (new_mask > 0).astype(np.float32)
    region = mask_new[y0:y1 + 1, x0:x1 + 1]
    region[ov] = local[ov]
    mask_new[y0:y1 + 1, x0:x1 + 1] = region

    return mask_new, seam_cost / (Hc + 1e-6)

def blend_seam_multiband(imgs, masks, order, levels=4, lamb_c=1.0, lamb_g=0.2):
    """Incremental seam + multi-band blending."""
    idx0 = order[0]
    pano = imgs[idx0].copy()
    pano_mask = masks[idx0].copy()
    seam_energies = []
    def cx(mask):
        ys, xs = np.where(mask > 0)
        return float(xs.mean()) if xs.size else 0.0

    for idx in order[1:]:
        new = imgs[idx]
        new_mask = masks[idx]
        orient = "right" if cx(new_mask) >= cx(pano_mask) else "left"
        m_new, e = seam_mask(pano, pano_mask, new, new_mask, orientation=orient, lamb_c=lamb_c, lamb_g=lamb_g)
        seam_energies.append(e)
        pano = multiband_blend_two(pano, new, m_new, levels=levels)
        pano_mask = ((pano_mask > 0) | (new_mask > 0)).astype(np.uint8) * 255

    return pano, pano_mask, float(np.mean(seam_energies)) if seam_energies else 0.0

def compute_overlap_metrics(imgs, masks, order):
    """Mean SSIM/PSNR over overlaps of adjacent images (warped)."""
    ssims, psnrs = [], []
    for a, b in zip(order[:-1], order[1:]):
        A = imgs[a]
        B = imgs[b]
        m = (masks[a] > 0) & (masks[b] > 0)
        if np.count_nonzero(m) < 3000:
            continue
        Ag = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY).astype(np.float32)
        Bg = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ys, xs = np.where(m)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        Agc = Ag[y0:y1 + 1, x0:x1 + 1]
        Bgc = Bg[y0:y1 + 1, x0:x1 + 1]
        mc = m[y0:y1 + 1, x0:x1 + 1]
        a_vec = Agc[mc]
        b_vec = Bgc[mc]
        mse = float(np.mean((a_vec - b_vec) ** 2))
        psnr_val = 99.0 if mse == 0 else 10 * math.log10((255 ** 2) / mse)
        psnrs.append(psnr_val)
        meanA = float(a_vec.mean())
        meanB = float(b_vec.mean())
        Ag_tmp = Agc.copy()
        Bg_tmp = Bgc.copy()
        Ag_tmp[~mc] = meanA
        Bg_tmp[~mc] = meanB
        ssims.append(float(ssim(Ag_tmp, Bg_tmp, data_range=255)))

    return float(np.mean(ssims)) if ssims else 0.0, float(np.mean(psnrs)) if psnrs else 0.0

@dataclass
class StitchConfig:
    feature: str = "ORB" # ORB or SIFT
    blend: str = "feather" # feather or seam_multiband
    resize_scale: float = 0.15
    max_features: int = 1200
    orb_ratio: float = 0.8
    sift_ratio: float = 0.75
    ransac_thresh: float = 3.0
    ransac_max_iters: int = 800
    dt_downscale: float = 0.3
    multiband_levels: int = 4
    seam_lamb_c: float = 1.0
    seam_lamb_g: float = 0.2


def stitch_images(image_paths: List[str], cfg: StitchConfig, out_dir: str, tag: str = "run") -> Dict:
    t0 = time.time()
    imgs, masks = [], []
    for p in image_paths:
        im = cv2.imread(p)
        if im is None:
            raise FileNotFoundError(p)
        im = resize_image(im, cfg.resize_scale)
        imgs.append(im)
        masks.append(np.ones(im.shape[:2], np.uint8) * 255)
    # Pairwise homography estimation
    H_pair = []
    inlier_counts = []
    for i in range(len(imgs) - 1):
        k1, d1, norm = detect_and_describe(imgs[i], cfg.feature, cfg.max_features)
        k2, d2, _ = detect_and_describe(imgs[i + 1], cfg.feature, cfg.max_features)
        ratio = cfg.orb_ratio if cfg.feature.upper() == "ORB" else cfg.sift_ratio
        matches = match_descriptors(d1, d2, norm, ratio=ratio, cross_check=True)
        H, inliers = compute_homography(k1, k2, matches, ransac_thresh=cfg.ransac_thresh, max_iters=cfg.ransac_max_iters)
        if H is None or inliers is None:
            raise RuntimeError(f"Homography failed for pair {i}-{i + 1}. Try SIFT or increase max_features.")
        H_pair.append(H)
        inlier_counts.append(int(inliers.sum()))
    T_to_ref, ref_idx = compute_global_transforms(H_pair)
    wimgs, wmasks, out_shape = warp_all_images(imgs, masks, T_to_ref)
    cimgs, gains = exposure_compensate(wimgs, wmasks, ref_idx)
    # blend order by x-center
    def cx(mask):
        ys, xs = np.where(mask > 0)
        return float(xs.mean()) if xs.size else 0.0
    centers = [cx(m) for m in wmasks]
    order = sorted(range(len(imgs)), key=lambda i: centers[i])
    mean_ssim, mean_psnr = compute_overlap_metrics(cimgs, wmasks, order)
    if cfg.blend == "feather":
        pano, pano_mask, seam_energy = blend_feather_incremental(
            cimgs, wmasks, order,
            dt_downscale=cfg.dt_downscale,
            lamb_c=cfg.seam_lamb_c,
            lamb_g=cfg.seam_lamb_g
        )
    elif cfg.blend == "seam_multiband":
        pano, pano_mask, seam_energy = blend_seam_multiband(
            cimgs, wmasks, order,
            levels=cfg.multiband_levels,
            lamb_c=cfg.seam_lamb_c,
            lamb_g=cfg.seam_lamb_g
        )
    else:
        raise ValueError("blend must be feather or seam_multiband")

    runtime = time.time() - t0
    os.makedirs(out_dir, exist_ok=True)
    pano_path = os.path.join(out_dir, f"{tag}_pano.jpg")
    mask_path = os.path.join(out_dir, f"{tag}_mask.png")
    meta_path = os.path.join(out_dir, f"{tag}_meta.json")
    cv2.imwrite(pano_path, pano)
    cv2.imwrite(mask_path, pano_mask)
    meta = {
        "tag": tag,
        "config": asdict(cfg),
        "ref_index": ref_idx,
        "order": order,
        "centers": centers,
        "out_shape": out_shape,
        "inliers_counts": inlier_counts,
        "gains": gains,
        "metrics": {
            "mean_ssim": mean_ssim,
            "mean_psnr": mean_psnr,
            "mean_seam_energy": seam_energy,
            "runtime_sec": runtime,
            "mean_inliers": float(np.mean(inlier_counts)),
            "num_images": len(imgs),
        }
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta
