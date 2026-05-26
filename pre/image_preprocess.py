"""
Image preprocessing for OCR quality improvement.

Provides enhancement features applied BEFORE VLM parsing:
  1. CLAHE adaptive contrast  (OpenCV — uneven lighting correction)
  2. Simple contrast           (PIL — uniform boost)
  3. Deskew                    (OpenCV — straighten rotated scans)
  4. Denoise                   (OpenCV fast NL-means)
  5. Contrast enhancement      (CLAHE on L-channel via OpenCV)
  6. Sharpness                 (PIL)

Only raster image formats are pre-processed (PNG, JPEG, TIFF, BMP, WebP).
PDF files are passed through untouched — MinerU handles them natively.

Author: Phạm Văn Khánh - 2026-05-08
"""
import logging
import math
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Raster formats that can be pre-processed
PREPROCESSABLE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

def is_preprocessable_image(file_path: str) -> bool:
    """Return True if the file is a raster image we can preprocess."""
    return Path(file_path).suffix.lower() in PREPROCESSABLE_EXTS

def rotate_image(
    image: np.ndarray,
    angle: float,
    background: int | tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Rotate *image* by *angle* degrees, expanding the canvas to avoid cropping."""
    if abs(angle) < 0.1:
        return image

    old_h, old_w = image.shape[:2]
    angle_rad = math.radians(angle)

    new_w = abs(math.sin(angle_rad) * old_h) + abs(math.cos(angle_rad) * old_w)
    new_h = abs(math.sin(angle_rad) * old_w) + abs(math.cos(angle_rad) * old_h)

    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot_mat[0, 2] += (new_w - old_w) / 2
    rot_mat[1, 2] += (new_h - old_h) / 2

    return cv2.warpAffine(
        image,
        rot_mat,
        (int(round(new_w)), int(round(new_h))),
        borderValue=background,
    )


def _find_skew_angle(gray: np.ndarray, delta: float = 0.2, limit: float = 5.0) -> float:
    """Estimate skew angle via projection-profile variance sweep."""
    angles = np.arange(-limit, limit + delta, delta)
    h, w = gray.shape
    center = (w // 2, h // 2)
    scores = []
    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        scores.append(np.var(np.sum(rotated, axis=1)))
    return float(angles[int(np.argmax(scores))])


def deskew_image(image: np.ndarray, scale_factor: float = 0.15) -> tuple[np.ndarray, float]:
    """Straighten a skewed image.

    Args:
        image: BGR image (from cv2).
        scale_factor: Down-scale ratio for fast angle detection.

    Returns:
        (rotated_image, detected_angle_degrees)
    """
    try:
        small = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
        angle = _find_skew_angle(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        if abs(angle) < 0.1:
            return image, 0.0
        angle = max(-45.0, min(45.0, angle))
        return rotate_image(image, angle, (255, 255, 255)), angle
    except Exception as exc:
        logger.warning(f"deskew_image: failed — {exc}")
        return image, 0.0


def denoise_image(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """Remove noise with OpenCV fast Non-Local Means Denoising."""
    if image.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)


def enhance_contrast(image: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """Apply CLAHE on the L-channel (LAB) to boost contrast without over-exposure."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    return clahe.apply(image)


def preprocess_image_pil(
    src_path: str,
    dst_path: str,
    contrast: float = 1.0,
    sharpness: float = 1.0,
    use_clahe: bool = True,
    use_deskew: bool = True,
    use_denoise: bool = True,
    use_enhancement: bool = True,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
) -> tuple[bool, dict]:
    """Apply image enhancements to a single raster image and save to *dst_path*.

    Steps (in order):
      1. CLAHE adaptive contrast  — if *use_clahe*
      2. Deskew                   — if *use_deskew*
      3. Denoise                  — if *use_denoise*
      4. Enhance contrast (CLAHE) — if *use_enhancement*
      5. PIL contrast             — if *contrast* ≠ 1.0
      6. PIL sharpness            — if *sharpness* ≠ 1.0

    Returns:
        (success, info_dict)
    """
    info: dict = {"applied": [], "mode": "none", "src": src_path, "dst": dst_path}

    # No-op shortcut
    if (
        contrast == 1.0
        and sharpness == 1.0
        and not use_clahe
        and not use_deskew
        and not use_denoise
        and not use_enhancement
    ):
        info["mode"] = "passthrough"
        if os.path.abspath(src_path) != os.path.abspath(dst_path):
            try:
                from shutil import copyfile
                copyfile(src_path, dst_path)
            except Exception as exc:
                logger.error(f"preprocess_image_pil: copy failed — {exc}")
                return False, info
        return True, info

    try:
        from PIL import Image, ImageEnhance, ImageFilter
    except ImportError:
        logger.error("Pillow not installed.  Run: pip install pillow")
        return False, info

    try:
        img = Image.open(src_path)

        # Normalise mode to RGB/L for processing
        if img.mode == "P" or img.mode == "1":
            img = img.convert("RGB")
        elif img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg

        # ---- OpenCV pipeline (CLAHE / deskew / denoise / enhance_contrast) ----
        need_cv = use_clahe or use_deskew or use_denoise or use_enhancement
        if need_cv:
            try:
                arr = np.array(img)
                # PIL is RGB; convert to BGR for OpenCV
                if arr.ndim == 3 and arr.shape[2] == 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

                # Step 1 — CLAHE adaptive contrast
                if use_clahe:
                    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(
                        clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid)
                    )
                    arr = cv2.cvtColor(
                        cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR
                    )
                    info["applied"].append(f"CLAHE(clip={clahe_clip}, grid={clahe_grid})")

                # Step 2 — Deskew
                if use_deskew:
                    arr, angle = deskew_image(arr)
                    if abs(angle) > 0.1:
                        info["applied"].append(f"deskew({angle:.2f}°)")

                # Step 3 — Denoise
                if use_denoise:
                    arr = denoise_image(arr)
                    info["applied"].append("denoise")

                # Step 4 — Contrast enhancement (CLAHE on L)
                if use_enhancement:
                    arr = enhance_contrast(arr)
                    info["applied"].append("enhance_contrast")

                # Back to RGB for PIL
                if arr.ndim == 3 and arr.shape[2] == 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(arr)
                info["mode"] = "opencv"

            except Exception as exc:
                logger.error(f"preprocess_image_pil: OpenCV pipeline failed — {exc}")
                # Continue to PIL steps with original img

        # ---- PIL contrast ----
        if contrast != 1.0:
            img = ImageEnhance.Contrast(img).enhance(contrast)
            info["applied"].append(f"contrast(×{contrast:.2f})")
            if info["mode"] == "none":
                info["mode"] = "pil"

        # ---- PIL sharpness ----
        if sharpness != 1.0:
            if sharpness <= 2.0:
                img = ImageEnhance.Sharpness(img).enhance(sharpness)
                info["applied"].append(f"sharpness(×{sharpness:.2f})")
            else:
                # Unsharp mask for aggressive sharpening
                img = img.filter(
                    ImageFilter.UnsharpMask(
                        radius=1.5,
                        percent=int(min(300, (sharpness - 1.0) * 100)),
                        threshold=2,
                    )
                )
                info["applied"].append(f"unsharp_mask(pct={int((sharpness-1.0)*100)})")
            if info["mode"] == "none":
                info["mode"] = "pil"

        # ---- Save ----
        ext = Path(dst_path).suffix.lower()
        save_kwargs: dict = {}
        if ext in (".jpg", ".jpeg"):
            save_kwargs = {"quality": 95, "optimize": True}
        elif ext == ".png":
            save_kwargs = {"compress_level": 6}
        elif ext in (".tif", ".tiff"):
            save_kwargs = {"compression": "tiff_lzw"}

        img.save(dst_path, **save_kwargs)
        return True, info

    except Exception as exc:
        logger.error(f"preprocess_image_pil: failed — {exc}")
        return False, info


# ---------------------------------------------------------------------------
# High-level entry point (called from tasks.py)
# ---------------------------------------------------------------------------

def maybe_preprocess(
    file_path: str,
    contrast: float = 1.0,
    sharpness: float = 1.0,
    use_clahe: bool = True,
    use_deskew: bool = True,
    use_denoise: bool = True,
    use_enhancement: bool = True,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
) -> tuple[str, dict]:
    """Detect file type and apply preprocessing if applicable.

    - Raster images (PNG, JPEG, TIFF, BMP, WebP): enhanced via OpenCV/PIL pipeline.
    - PDF files: returned as-is — MinerU parses PDFs natively; pre-processing
      would convert text layers to raster (lossy).
    - Other types: returned as-is.

    Returns:
        (path_to_use, info_dict)
        ``path_to_use`` is either the original path or a new preprocessed file
        written alongside the source (``<stem>_preprocessed<ext>``).
    """
    info: dict = {"file_path": file_path, "preprocessed": False, "details": None}

    # No-op shortcut
    if (
        contrast == 1.0
        and sharpness == 1.0
        and not use_clahe
        and not use_deskew
        and not use_denoise
        and not use_enhancement
    ):
        info["details"] = {"mode": "skipped (no enhancement requested)"}
        return file_path, info

    src = Path(file_path)
    suffix = src.suffix.lower()

    # PDF — pass through
    if suffix == ".pdf":
        info["details"] = {"mode": "skipped (PDF — handled natively by MinerU)"}
        return file_path, info

    # Unsupported raster type
    if suffix not in PREPROCESSABLE_EXTS:
        info["details"] = {"mode": f"skipped (unsupported type: {suffix})"}
        return file_path, info

    out_path = src.parent / f"{src.stem}_preprocessed{src.suffix}"

    ok, det = preprocess_image_pil(
        str(src),
        str(out_path),
        contrast=contrast,
        sharpness=sharpness,
        use_clahe=use_clahe,
        use_deskew=use_deskew,
        use_denoise=use_denoise,
        use_enhancement=use_enhancement,
        clahe_clip=clahe_clip,
        clahe_grid=clahe_grid,
    )

    info["details"] = det
    if ok:
        info["preprocessed"] = True
        info["file_path"] = str(out_path)
        logger.info(f"[image_preprocess] applied: {det.get('applied', [])}")
        return str(out_path), info

    logger.warning(f"[image_preprocess] preprocessing failed, using original: {file_path}")
    return file_path, info
