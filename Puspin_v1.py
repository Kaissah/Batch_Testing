import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class FADResult:
    risk_level: str
    redness_score: float
    bald_area_ratio: float
    lesion_circularity: float
    affected_pixel_count: int
    overlay_image: np.ndarray
    metrics: dict


K_SENSITIVITY = 2.0  
MIN_AREA_SIG = 0.005 

def preprocess(image_bgr: np.ndarray) -> dict:
    h, w = image_bgr.shape[:2]
    scale = 700 / max(h, w)
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))

    denoised = cv2.bilateralFilter(image_bgr, d=9, sigmaColor=75, sigmaSpace=75)

    return {
        "bgr":          denoised,
        "hsv":          cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV),
        "lab":          cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB),
        "gray":         cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY),
        "original_bgr": image_bgr,
    }


def get_cat_mask(data: dict) -> np.ndarray:
    hsv  = data["hsv"]
    gray = data["gray"]
    s, v = hsv[:, :, 1], hsv[:, :, 2]

    gray_f  = gray.astype(np.float32)
    mean_sq = cv2.blur(gray_f ** 2, (15, 15))
    mean    = cv2.blur(gray_f,      (15, 15))
    texture = np.sqrt((mean_sq - mean ** 2).clip(0))

    bg = ((s < 30) & (v > 200)) | ((texture < 8) & (v > 180))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bg = cv2.dilate(bg.astype(np.uint8), kernel, iterations=2)

    return ((1 - bg) * 255).astype(np.uint8)


def compute_edge_anomaly(gray: np.ndarray, cat_mask: np.ndarray) -> np.ndarray:
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sx**2 + sy**2)

    cat_vals = sobel_mag[cat_mask > 0]
    sobel_norm = (sobel_mag / cat_vals.max()).clip(0, 1) if cat_vals.max() > 0 else sobel_mag * 0

    sobel_f    = sobel_norm.astype(np.float32)
    local_mean = cv2.blur(sobel_f, (25, 25))
    local_msq  = cv2.blur(sobel_f ** 2, (25, 25))
    local_var  = (local_msq - local_mean ** 2).clip(0)

    anomaly = local_var / (local_var.max() + 1e-6)
    return anomaly.astype(np.float32)

def compute_texture_map(gray: np.ndarray, cat_mask: np.ndarray) -> np.ndarray:
    gray_f  = gray.astype(np.float32)
    mean_sq = cv2.blur(gray_f ** 2, (9, 9))
    mean    = cv2.blur(gray_f,      (9, 9))
    texture = np.sqrt((mean_sq - mean ** 2).clip(0))

    max_tex = texture[cat_mask > 0].max() if cat_mask.any() else 1.0
    smooth_map = 1.0 - (texture / (max_tex + 1e-6)).clip(0, 1)
    return smooth_map.astype(np.float32)

def get_adaptive_candidate_mask(edge_anomaly, smooth_map, cat_mask, gray) -> np.ndarray:

    edge_complexity = np.std(edge_anomaly[cat_mask > 0]) if cat_mask.any() else 0.5
    texture_w = np.clip(edge_complexity * 2, 0.4, 0.8)
    edge_w = 1.0 - texture_w

    fused = (edge_w * edge_anomaly + texture_w * smooth_map).clip(0, 1)
    fused_u8 = cv2.bitwise_and((fused * 255).astype(np.uint8), (fused * 255).astype(np.uint8), mask=cat_mask)
    
    _, candidate = cv2.threshold(fused_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(candidate, cv2.MORPH_OPEN, k)

def confirm_lesions_statistically(data: dict, candidate_mask: np.ndarray, cat_mask: np.ndarray) -> tuple:
    a = data["lab"][:, :, 1].astype(np.float32)
    cat_a_vals = a[cat_mask > 0]
    
    baseline_a = float(np.mean(cat_a_vals)) if len(cat_a_vals) > 0 else 128.0
    std_a      = float(np.std(cat_a_vals))  if len(cat_a_vals) > 0 else 5.0

    threshold_a = baseline_a + (K_SENSITIVITY * std_a)
    
    inflamed = (a > threshold_a).astype(np.uint8) * 255
    lesion_mask = cv2.bitwise_and(candidate_mask, inflamed)
    
    return lesion_mask, baseline_a, std_a


def filter_blobs(mask: np.ndarray, max_frac=0.6) -> np.ndarray:
    total = mask.shape[0] * mask.shape[1]
    conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    for cnt in conts:
        area = cv2.contourArea(cnt)
        if 100 < area < (total * max_frac):
            cv2.drawContours(out, [cnt], -1, 255, -1)
    return out

def analyze_shape(lesion_mask: np.ndarray, cat_mask: np.ndarray) -> dict:
    cat_pixels = np.sum(cat_mask > 0)
    conts, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not conts:
        return {"area_ratio": 0.0, "circularity": 0.0, "count": 0, "conts": []}

    largest = max(conts, key=cv2.contourArea)
    p = cv2.arcLength(largest, True)
    a = cv2.contourArea(largest)
    circ = (4 * np.pi * a) / (p**2) if p > 0 else 0
    
    return {
        "area_ratio": float(np.sum(lesion_mask > 0) / (cat_pixels + 1)),
        "circularity": min(circ, 1.0),
        "count": len(conts),
        "conts": conts
    }

def detect(image_bgr: np.ndarray) -> FADResult:
    data = preprocess(image_bgr)
    cat_mask = get_cat_mask(data)
    
    
    edge_map = compute_edge_anomaly(data["gray"], cat_mask)
    smooth_map = compute_texture_map(data["gray"], cat_mask)
    
    
    candidates = get_adaptive_candidate_mask(edge_map, smooth_map, cat_mask, data["gray"])
    
    
    lesion_mask, base_a, std_a = confirm_lesions_statistically(data, candidates, cat_mask)
    
   
    lesion_mask = filter_blobs(lesion_mask)
    
    
    shape = analyze_shape(lesion_mask, cat_mask)
    

    is_healthy = True
    if shape["area_ratio"] > MIN_AREA_SIG and shape["count"] > 0:
        is_healthy = False

    overlay = data["original_bgr"].copy()
    if not is_healthy:
        mask_color = np.zeros_like(overlay)
        mask_color[lesion_mask > 0] = [30, 30, 220]
        cv2.addWeighted(mask_color, 0.4, overlay, 0.6, 0, overlay)
        cv2.drawContours(overlay, shape["conts"], -1, (0, 255, 255), 2)
        status = "Abnormality Detected"
    else:
        status = "No Abnormality Detected"

    cv2.putText(overlay, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4)
    cv2.putText(overlay, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    return FADResult(
        risk_level = "High" if not is_healthy else "Low",
        redness_score = float((base_a - 128) / 128),
        bald_area_ratio = shape["area_ratio"],
        lesion_circularity = shape["circularity"],
        affected_pixel_count = int(np.sum(lesion_mask > 0)),
        overlay_image = overlay,
        metrics = {"stat_base_redness": base_a, "stat_std_redness": std_a}
    )

if __name__ == "__main__":
    img = cv2.imread(r"..\Dataset\Allergy dermatitis\AD_Opulencia_1.png")
    if img is not None:
        res = detect(img)
        cv2.imshow("Adaptive FAD Detection", res.overlay_image)
        cv2.waitKey(0)