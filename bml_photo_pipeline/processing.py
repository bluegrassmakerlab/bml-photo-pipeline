from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from html import escape
import io
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass


@dataclass(frozen=True)
class ExportSpec:
    width: int
    height: int


@dataclass(frozen=True)
class PhotoQuality:
    subject_found: bool
    center_offset_x: float
    center_offset_y: float
    fill_percent: float
    subject_luminance: float
    tilt_degrees: float | None
    passes: bool


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}

USAGE_GUIDE = {
    "etsy_main": {
        "destination": "Etsy listing photo #1",
        "usage": "Use as the listing hero image. Pick the cleanest, most centered product shot.",
    },
    "etsy_gallery": {
        "destination": "Etsy listing gallery",
        "usage": "Use for alternate angles, details, scale, packaging, or color variants.",
    },
    "social_4x5": {
        "destination": "Instagram/Facebook feed",
        "usage": "Use for normal feed posts where the image should fill vertical feed space.",
    },
    "social_9x16": {
        "destination": "Stories and vertical photo posts",
        "usage": "Use for Instagram/Facebook stories, TikTok photo mode, and Shorts-style image posts.",
    },
    "etsy_video": {
        "destination": "Etsy listing video",
        "usage": "Use as the simple square product-motion video on the listing.",
    },
    "social_reels": {
        "destination": "TikTok/Reels/Shorts",
        "usage": "Use for TikTok, Instagram Reels, Facebook Reels, and YouTube Shorts.",
    },
    "video_thumbnail": {
        "destination": "Video/reel cover",
        "usage": "Use as the cover image or thumbnail for short-form video posts.",
    },
}


def safe_filename_component(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip()).strip("-._")
    return cleaned.lower()


def source_output_stem(source: Path) -> str:
    parent = safe_filename_component(source.parent.name)
    if parent and parent not in {"incoming", "work", "processed"}:
        return f"{parent}_{source.stem}"
    return source.stem


def deterministic_index(seed: str, total: int) -> int:
    if total <= 1:
        return 0
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % total
USAGE_ORDER = {name: index for index, name in enumerate(USAGE_GUIDE)}


def media_type(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return None


def open_image(path: Path) -> Image.Image:
    if path.suffix.lower() in {".heic", ".heif"}:
        try:
            import pillow_heif

            return pillow_heif.open_heif(path).to_pillow().convert("RGB")
        except Exception:
            if shutil.which("ffmpeg"):
                proc = subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(path),
                        "-frames:v",
                        "1",
                        "-f",
                        "image2pipe",
                        "-vcodec",
                        "png",
                        "-",
                    ],
                    check=False,
                    capture_output=True,
                )
                if proc.returncode == 0 and proc.stdout:
                    return Image.open(io.BytesIO(proc.stdout)).convert("RGB")

    image = Image.open(path)
    return ImageOps.exif_transpose(image).convert("RGB")


def resize_for_processing(image: Image.Image, max_dimension: int) -> Image.Image:
    if max_dimension <= 0 or max(image.size) <= max_dimension:
        return image
    scale = max_dimension / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def content_bounds(image: Image.Image, threshold: int) -> tuple[int, int, int, int] | None:
    arr = np.asarray(image.convert("RGB")).astype(np.int16)
    corners = np.array(
        [
            arr[0, 0],
            arr[0, -1],
            arr[-1, 0],
            arr[-1, -1],
        ]
    )
    bg = np.median(corners, axis=0)
    diff = np.abs(arr - bg).mean(axis=2)
    mask = diff > threshold

    if not mask.any():
        return None

    y_indices, x_indices = np.where(mask)
    left, right = int(x_indices.min()), int(x_indices.max())
    top, bottom = int(y_indices.min()), int(y_indices.max())
    return left, top, right + 1, bottom + 1


def mask_bounds(mask: np.ndarray, inset_percent: float = 0.01) -> tuple[int, int, int, int] | None:
    if not mask.any():
        return None

    y_indices, x_indices = np.where(mask)
    if inset_percent > 0 and len(x_indices) >= 64:
        lower = inset_percent
        upper = 1 - inset_percent
        left = int(np.quantile(x_indices, lower))
        right = int(np.quantile(x_indices, upper)) + 1
        top = int(np.quantile(y_indices, lower))
        bottom = int(np.quantile(y_indices, upper)) + 1
    else:
        left, right = int(x_indices.min()), int(x_indices.max()) + 1
        top, bottom = int(y_indices.min()), int(y_indices.max()) + 1

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def subject_mask(
    image: Image.Image,
    threshold: int,
    saturation_threshold: int = 45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(image.convert("RGB")).astype(np.int16)
    corners = np.array(
        [
            arr[0, 0],
            arr[0, -1],
            arr[-1, 0],
            arr[-1, -1],
        ]
    )
    bg = np.median(corners, axis=0)
    diff = np.abs(arr - bg).mean(axis=2)
    saturation = arr.max(axis=2) - arr.min(axis=2)
    luminance = (0.2126 * arr[:, :, 0]) + (0.7152 * arr[:, :, 1]) + (0.0722 * arr[:, :, 2])
    bg_luminance = float((0.2126 * bg[0]) + (0.7152 * bg[1]) + (0.0722 * bg[2]))

    colorful = (saturation > saturation_threshold) & (diff > max(10, threshold))
    neutral_product = (diff > max(34, threshold * 1.6)) & (np.abs(luminance - bg_luminance) > 18)
    return colorful | neutral_product, colorful, diff


def subject_bounds(
    image: Image.Image,
    threshold: int,
    saturation_threshold: int = 45,
    min_area_percent: float = 0.005,
    min_saturated_area_percent: float = 0.03,
) -> tuple[int, int, int, int] | None:
    mask, colorful_mask, _diff = subject_mask(image, threshold, saturation_threshold)

    broad_bounds = content_bounds(image, threshold)
    if broad_bounds:
        left, top, right, bottom = broad_bounds
        broad_area = max(1, (right - left) * (bottom - top))
    else:
        broad_area = max(1, image.width * image.height)
    min_area = max(
        int(image.width * image.height * min_area_percent),
        int(broad_area * min_saturated_area_percent),
    )
    if colorful_mask.sum() >= max(1, min_area):
        colorful_bounds = mask_bounds(colorful_mask, inset_percent=0.01)
        if colorful_bounds and broad_bounds:
            color_left, color_top, color_right, color_bottom = colorful_bounds
            broad_left, broad_top, broad_right, broad_bottom = broad_bounds
            color_width = color_right - color_left
            color_height = color_bottom - color_top
            broad_width = max(1, broad_right - broad_left)
            broad_height = max(1, broad_bottom - broad_top)
            color_area = color_width * color_height
            color_density = float(colorful_mask.sum()) / max(1, color_area)
            if (
                color_area >= broad_area * 0.08
                and color_density >= 0.35
                and color_width >= broad_width * 0.18
                and color_height >= broad_height * 0.18
            ):
                return colorful_bounds
            if (
                color_width >= broad_width * 0.45
                and color_height >= broad_height * 0.45
                and color_area >= broad_area * 0.25
                and color_top <= broad_top + (broad_height * 0.18)
            ):
                return colorful_bounds
        elif colorful_bounds:
            return colorful_bounds
    if mask.sum() >= max(1, min_area):
        combined_bounds = mask_bounds(mask, inset_percent=0.015)
        if combined_bounds and broad_bounds:
            combined_left, combined_top, combined_right, combined_bottom = combined_bounds
            broad_left, broad_top, broad_right, broad_bottom = broad_bounds
            combined_width = combined_right - combined_left
            combined_height = combined_bottom - combined_top
            broad_width = max(1, broad_right - broad_left)
            broad_height = max(1, broad_bottom - broad_top)
            if (
                combined_width >= broad_width * 0.45
                and combined_height >= broad_height * 0.45
                and combined_top <= broad_top + (broad_height * 0.18)
            ):
                return combined_bounds
            return broad_bounds
        return combined_bounds

    return broad_bounds


def subject_luminance(image: Image.Image, bounds: tuple[int, int, int, int]) -> float:
    left, top, right, bottom = bounds
    crop = image.convert("RGB").crop((left, top, right, bottom))
    arr = np.asarray(crop).astype(np.float32)
    luminance = (0.2126 * arr[:, :, 0]) + (0.7152 * arr[:, :, 1]) + (0.0722 * arr[:, :, 2])
    return float(np.median(luminance))


def normalize_subject_luminance(
    image: Image.Image,
    bounds: tuple[int, int, int, int],
    target_luminance: float,
    max_adjustment: float,
) -> Image.Image:
    if target_luminance <= 0 or max_adjustment <= 0:
        return image
    current = subject_luminance(image, bounds)
    if current <= 1:
        delta = min(target_luminance - current, 255 * max_adjustment)
        if delta <= 1:
            return image
        arr = np.asarray(image.convert("RGB")).astype(np.float32)
        luminance = (0.2126 * arr[:, :, 0]) + (0.7152 * arr[:, :, 1]) + (0.0722 * arr[:, :, 2])
        highlight_protection = np.power(np.clip((255 - luminance) / 255, 0, 1), 1.6)
        lifted = np.clip(arr + (delta * highlight_protection[:, :, None]), 0, 255).astype(np.uint8)
        return Image.fromarray(lifted, "RGB")
    factor = target_luminance / current
    lower = max(0.1, 1 - max_adjustment)
    upper = 1 + max_adjustment
    factor = max(lower, min(upper, factor))
    if abs(factor - 1) < 0.03:
        return image
    if factor > 1:
        arr = np.asarray(image.convert("RGB")).astype(np.float32)
        luminance = (0.2126 * arr[:, :, 0]) + (0.7152 * arr[:, :, 1]) + (0.0722 * arr[:, :, 2])
        highlight_protection = np.power(np.clip((255 - luminance) / 255, 0, 1), 1.6)
        protected_factor = 1 + ((factor - 1) * highlight_protection)
        lifted = np.clip(arr * protected_factor[:, :, None], 0, 255).astype(np.uint8)
        return Image.fromarray(lifted, "RGB")
    return ImageEnhance.Brightness(image).enhance(factor)


def background_luminance(image: Image.Image) -> float:
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    height, width = arr.shape[:2]
    border = max(8, min(width, height) // 12)
    samples = np.concatenate(
        [
            arr[:border, :, :].reshape(-1, 3),
            arr[-border:, :, :].reshape(-1, 3),
            arr[:, :border, :].reshape(-1, 3),
            arr[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    luminance = (0.2126 * samples[:, 0]) + (0.7152 * samples[:, 1]) + (0.0722 * samples[:, 2])
    return float(np.median(luminance))


def lift_neutral_background(
    image: Image.Image,
    threshold: int,
    saturation_threshold: int,
    target_luminance: float,
    max_lift: float,
    subject_protection_px: int = 18,
) -> Image.Image:
    if target_luminance <= 0 or max_lift <= 0:
        return image

    current = background_luminance(image)
    lift = min(max_lift, target_luminance - current)
    if lift <= 1:
        return image

    rgb = image.convert("RGB")
    arr = np.asarray(rgb).astype(np.float32)
    subject, _colorful, _diff = subject_mask(rgb, threshold, saturation_threshold)
    subject_image = Image.fromarray((subject.astype(np.uint8) * 255), "L")
    if subject_protection_px > 0:
        kernel = max(3, subject_protection_px)
        if kernel % 2 == 0:
            kernel += 1
        subject_image = subject_image.filter(ImageFilter.MaxFilter(kernel))
    subject_protection = np.asarray(subject_image).astype(np.float32) / 255

    saturation = arr.max(axis=2) - arr.min(axis=2)
    luminance = (0.2126 * arr[:, :, 0]) + (0.7152 * arr[:, :, 1]) + (0.0722 * arr[:, :, 2])
    neutral_weight = np.clip((38 - saturation) / 38, 0, 1)
    dark_floor = np.clip((luminance - 45) / 90, 0, 1)
    highlight_protection = np.power(np.clip((255 - luminance) / 96, 0, 1), 0.8)
    background_weight = neutral_weight * dark_floor * highlight_protection * (1 - subject_protection)
    if float(background_weight.max()) <= 0:
        return image

    lifted = np.clip(arr + (lift * background_weight[:, :, None]), 0, 255).astype(np.uint8)
    return Image.fromarray(lifted, "RGB")


def estimate_tilt_degrees(image: Image.Image, threshold: int) -> float | None:
    analysis = image.convert("RGB")
    max_dimension = max(analysis.size)
    if max_dimension > 1000:
        scale = 1000 / max_dimension
        analysis = analysis.resize(
            (max(1, round(analysis.width * scale)), max(1, round(analysis.height * scale))),
            Image.Resampling.BILINEAR,
        )

    bounds = content_bounds(analysis, threshold)
    if not bounds:
        return None
    left, top, right, bottom = bounds
    crop = analysis.crop((left, top, right, bottom))
    mask, _colorful_mask, _diff = subject_mask(crop, threshold)
    if mask.sum() < max(64, int(crop.width * crop.height * 0.01)):
        return None

    y_indices, x_indices = np.where(mask)
    points = np.column_stack((x_indices, y_indices)).astype(np.float32)
    points -= points.mean(axis=0)
    covariance = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    if values[0] <= 0 or values[1] / values[0] < 1.7:
        return None
    major = vectors[:, int(np.argmax(values))]
    angle = float(np.degrees(np.arctan2(major[1], major[0])))
    while angle <= -90:
        angle += 180
    while angle > 90:
        angle -= 180
    if abs(angle) > 45:
        angle = angle - 90 if angle > 0 else angle + 90
    return angle


def straighten_subject(
    image: Image.Image,
    background_color: tuple[int, int, int],
    threshold: int,
    max_degrees: float,
) -> Image.Image:
    if max_degrees <= 0:
        return image
    tilt = estimate_tilt_degrees(image, threshold)
    if tilt is None or abs(tilt) < 1.0 or abs(tilt) > max_degrees:
        return image
    rotated = image.rotate(tilt, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=background_color)
    return crop_rotation_fill(rotated, image.size, tilt)


def crop_rotation_fill(image: Image.Image, original_size: tuple[int, int], angle_degrees: float) -> Image.Image:
    if abs(angle_degrees) < 0.01:
        return image

    mask = Image.new("L", original_size, 255)
    mask = mask.rotate(angle_degrees, resample=Image.Resampling.NEAREST, expand=True, fillcolor=0)
    aspect = original_size[0] / original_size[1]
    max_height = int(min(mask.height, mask.width / aspect))
    if max_height <= 0:
        return image

    center_x = mask.width // 2
    center_y = mask.height // 2

    def is_valid(height: int) -> bool:
        width = max(1, round(height * aspect))
        left = center_x - width // 2
        top = center_y - height // 2
        right = left + width
        bottom = top + height
        if left < 0 or top < 0 or right > mask.width or bottom > mask.height:
            return False
        return bool(np.asarray(mask.crop((left, top, right, bottom))).min() >= 250)

    low, high = 1, max_height
    while low <= high:
        mid = (low + high) // 2
        if is_valid(mid):
            low = mid + 1
        else:
            high = mid - 1

    crop_height = max(1, high - 2)
    crop_width = max(1, round(crop_height * aspect) - 2)
    left = max(0, center_x - crop_width // 2)
    top = max(0, center_y - crop_height // 2)
    right = min(image.width, left + crop_width)
    bottom = min(image.height, top + crop_height)
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def trim_background(image: Image.Image, threshold: int, padding_percent: float) -> Image.Image:
    bounds = content_bounds(image, threshold)
    if not bounds:
        return image
    left, top, right, bottom = bounds
    width, height = image.size
    pad = int(max(right - left, bottom - top) * padding_percent)
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(width, right + pad)
    bottom = min(height, bottom + pad)

    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def frame_subject(
    image: Image.Image,
    target_ratio: float,
    background_color: tuple[int, int, int],
    threshold: int,
    padding_percent: float,
    saturation_threshold: int = 45,
    target_subject_height_percent: float = 0,
    max_subject_width_percent: float = 0.86,
) -> Image.Image:
    bounds = subject_bounds(image, threshold, saturation_threshold=saturation_threshold)
    if not bounds:
        return image.convert("RGB")
    left, top, right, bottom = bounds
    subject_width = max(1, right - left)
    subject_height = max(1, bottom - top)
    if target_subject_height_percent > 0 and (
        subject_width / image.width > 0.92 or subject_height / image.height > 0.9
    ):
        return image.convert("RGB")
    pad = int(max(subject_width, subject_height) * padding_percent)
    if target_subject_height_percent > 0:
        target_subject_height_percent = max(0.2, min(0.92, target_subject_height_percent))
        max_subject_width_percent = max(0.35, min(0.95, max_subject_width_percent))
        min_frame_height_for_height = subject_height / target_subject_height_percent
        min_frame_height_for_width = subject_width / (target_ratio * max_subject_width_percent)
        min_frame_height_for_padding = (subject_height + (2 * pad))
        min_frame_width_for_padding = (subject_width + (2 * pad))
        frame_height = max(
            min_frame_height_for_height,
            min_frame_height_for_width,
            min_frame_height_for_padding,
            min_frame_width_for_padding / target_ratio,
        )
        frame_width = round(frame_height * target_ratio)
        frame_height = round(frame_height)
    else:
        left -= pad
        top -= pad
        right += pad
        bottom += pad

        box_width = max(1, right - left)
        box_height = max(1, bottom - top)
        if box_width / box_height > target_ratio:
            frame_width = box_width
            frame_height = round(box_width / target_ratio)
        else:
            frame_height = box_height
            frame_width = round(box_height * target_ratio)

    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    crop_left = round(center_x - frame_width / 2)
    crop_top = round(center_y - frame_height / 2)
    crop_right = crop_left + frame_width
    crop_bottom = crop_top + frame_height

    source = image.convert("RGB")
    if crop_left >= 0 and crop_top >= 0 and crop_right <= source.width and crop_bottom <= source.height:
        return source.crop((crop_left, crop_top, crop_right, crop_bottom))

    framed = Image.new("RGB", (frame_width, frame_height), background_color)
    framed.paste(source, (-crop_left, -crop_top))
    return framed


def remove_background_if_available(image: Image.Image, enabled: bool) -> Image.Image:
    if not enabled:
        return image
    try:
        from rembg import remove
    except Exception:
        return image

    transparent = remove(image.convert("RGBA"))
    return transparent.convert("RGBA")


def flatten(image: Image.Image, background_color: tuple[int, int, int]) -> Image.Image:
    if image.mode != "RGBA":
        return image.convert("RGB")
    background = Image.new("RGBA", image.size, (*background_color, 255))
    composited = Image.alpha_composite(background, image)
    return composited.convert("RGB")


def white_balance_background(image: Image.Image, strength: float) -> Image.Image:
    if strength <= 0:
        return image
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    height, width = arr.shape[:2]
    border = max(8, min(width, height) // 18)
    samples = np.concatenate(
        [
            arr[:border, :, :].reshape(-1, 3),
            arr[-border:, :, :].reshape(-1, 3),
            arr[:, :border, :].reshape(-1, 3),
            arr[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    brightness = samples.mean(axis=1)
    neutralish = samples[(brightness > 110) & (brightness < 245)]
    if len(neutralish) < 128:
        neutralish = samples

    bg = np.median(neutralish, axis=0)
    target = float(np.mean(bg))
    if target <= 0 or np.any(bg <= 1):
        return image

    scale = target / bg
    scale = 1 + ((scale - 1) * min(strength, 1.0))
    balanced = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return Image.fromarray(balanced, "RGB")


def autocontrast_luminance(image: Image.Image, cutoff: float) -> Image.Image:
    ycbcr = image.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = ImageOps.autocontrast(y, cutoff=cutoff)
    return Image.merge("YCbCr", (y, cb, cr)).convert("RGB")


def polish(image: Image.Image, config: dict) -> Image.Image:
    settings = config["processing"]
    bg_color = tuple(settings.get("background_color", [248, 248, 245]))
    if settings.get("preserve_photo_edits", False):
        return flatten(image, bg_color)

    if settings.get("white_balance", True):
        image = white_balance_background(image, float(settings.get("white_balance_strength", 0.85)))

    if settings.get("auto_rotate", True):
        image = straighten_subject(
            image,
            bg_color,
            int(settings.get("subject_threshold", settings.get("trim_threshold", 22))),
            float(settings.get("auto_rotate_max_degrees", 5)),
        )

    if settings.get("trim_background", True):
        image = trim_background(
            image,
            int(settings.get("trim_threshold", 22)),
            float(settings.get("trim_padding_percent", 0.08)),
        )

    image = remove_background_if_available(image, bool(settings.get("remove_background", False)))
    image = flatten(image, bg_color)

    if settings.get("normalize_subject_brightness", True):
        bounds = subject_bounds(
            image,
            int(settings.get("subject_threshold", settings.get("trim_threshold", 22))),
            int(settings.get("subject_saturation_threshold", 45)),
        )
        if bounds:
            image = normalize_subject_luminance(
                image,
                bounds,
                float(settings.get("target_subject_luminance", 172)),
                float(settings.get("max_subject_brightness_adjustment", 0.18)),
            )

    if settings.get("lift_neutral_background", True):
        image = lift_neutral_background(
            image,
            int(settings.get("subject_threshold", settings.get("trim_threshold", 22))),
            int(settings.get("subject_saturation_threshold", 45)),
            float(settings.get("target_background_luminance", 236)),
            float(settings.get("max_background_lift", 34)),
            int(settings.get("background_subject_protection_px", 18)),
        )

    cutoff = float(settings.get("autocontrast_cutoff", 1))
    if settings.get("autocontrast_luminance", True):
        image = autocontrast_luminance(image, cutoff)
    else:
        image = ImageOps.autocontrast(image, cutoff=cutoff)
    image = ImageEnhance.Brightness(image).enhance(float(settings.get("brightness", 1.05)))
    image = ImageEnhance.Contrast(image).enhance(float(settings.get("contrast", 1.08)))
    image = ImageEnhance.Color(image).enhance(float(settings.get("color", 1.02)))
    image = ImageEnhance.Sharpness(image).enhance(float(settings.get("sharpness", 1.18)))
    if settings.get("normalize_subject_brightness", True):
        bounds = subject_bounds(
            image,
            int(settings.get("subject_threshold", settings.get("trim_threshold", 22))),
            int(settings.get("subject_saturation_threshold", 45)),
        )
        if bounds:
            image = normalize_subject_luminance(
                image,
                bounds,
                float(settings.get("target_subject_luminance", 172)),
                float(settings.get("max_subject_brightness_adjustment", 0.18)),
            )
    return image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=3))


def assess_photo_quality(
    image: Image.Image,
    threshold: int,
    saturation_threshold: int = 45,
    max_center_offset_percent: float = 0.08,
    min_fill_percent: float = 0.12,
    max_fill_percent: float = 0.93,
    min_subject_luminance: float = 70,
    max_subject_luminance: float = 230,
) -> PhotoQuality:
    bounds = subject_bounds(image, threshold, saturation_threshold)
    tilt = estimate_tilt_degrees(image, threshold)
    if not bounds:
        return PhotoQuality(False, 1.0, 1.0, 0.0, 0.0, tilt, False)

    left, top, right, bottom = bounds
    center_offset_x = (((left + right) / 2) - (image.width / 2)) / image.width
    center_offset_y = (((top + bottom) / 2) - (image.height / 2)) / image.height
    fill_percent = ((right - left) * (bottom - top)) / (image.width * image.height)
    luminance = subject_luminance(image, bounds)
    passes = (
        abs(center_offset_x) <= max_center_offset_percent
        and abs(center_offset_y) <= max_center_offset_percent
        and min_fill_percent <= fill_percent <= max_fill_percent
        and min_subject_luminance <= luminance <= max_subject_luminance
    )
    return PhotoQuality(True, center_offset_x, center_offset_y, fill_percent, luminance, tilt, passes)


def correct_export_quality(
    image: Image.Image,
    spec: ExportSpec,
    background_color: tuple[int, int, int],
    *,
    subject_threshold: int,
    subject_padding_percent: float,
    subject_saturation_threshold: int,
    target_subject_luminance: float,
    max_subject_brightness_adjustment: float,
    target_subject_height_percent: float,
    max_subject_width_percent: float,
) -> Image.Image:
    quality = assess_photo_quality(image, subject_threshold, subject_saturation_threshold)
    corrected = image
    if quality.subject_found and not quality.passes:
        corrected = normalize_subject_luminance(
            corrected,
            subject_bounds(corrected, subject_threshold, subject_saturation_threshold) or (0, 0, corrected.width, corrected.height),
            target_subject_luminance,
            max_subject_brightness_adjustment,
        )
        corrected = frame_subject(
            corrected,
            spec.width / spec.height,
            background_color,
            subject_threshold,
            subject_padding_percent,
            subject_saturation_threshold,
            target_subject_height_percent,
            max_subject_width_percent,
        )
        corrected = fit_on_canvas(
            corrected,
            spec,
            background_color,
            center_subject=False,
            subject_threshold=subject_threshold,
            subject_padding_percent=subject_padding_percent,
            subject_saturation_threshold=subject_saturation_threshold,
        )
    return corrected


def fit_on_canvas(
    image: Image.Image,
    spec: ExportSpec,
    background_color: tuple[int, int, int],
    *,
    center_subject: bool = True,
    subject_threshold: int = 18,
    subject_padding_percent: float = 0.16,
    subject_saturation_threshold: int = 45,
    target_subject_height_percent: float = 0,
    max_subject_width_percent: float = 0.86,
) -> Image.Image:
    target_ratio = spec.width / spec.height
    if center_subject:
        image = frame_subject(
            image,
            target_ratio,
            background_color,
            subject_threshold,
            subject_padding_percent,
            subject_saturation_threshold,
            target_subject_height_percent,
            max_subject_width_percent,
        )
    src_ratio = image.width / image.height

    if src_ratio > target_ratio:
        new_width = spec.width
        new_height = round(spec.width / src_ratio)
    else:
        new_height = spec.height
        new_width = round(spec.height * src_ratio)

    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (spec.width, spec.height), background_color)
    x = (spec.width - new_width) // 2
    y = (spec.height - new_height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def process_image(source: Path, output_dir: Path, config: dict) -> dict[str, Path]:
    max_source_dimension = int(config["processing"].get("max_source_dimension", 3200))
    image = polish(resize_for_processing(open_image(source), max_source_dimension), config)
    bg_color = tuple(config["processing"].get("background_color", [248, 248, 245]))
    stem = source_output_stem(source)

    exports: dict[str, Path] = {}
    image_exports = config.get("image_exports", config.get("exports", {}))
    center_subject = bool(config["processing"].get("center_subject", True))
    subject_threshold = int(config["processing"].get("subject_threshold", config["processing"].get("trim_threshold", 22)))
    subject_padding_percent = float(config["processing"].get("subject_padding_percent", 0.16))
    subject_saturation_threshold = int(config["processing"].get("subject_saturation_threshold", 45))
    target_subject_height_percent = float(config["processing"].get("target_subject_height_percent", 0.56))
    max_subject_width_percent = float(config["processing"].get("max_subject_width_percent", 0.86))
    for name, spec_data in image_exports.items():
        spec = ExportSpec(width=int(spec_data["width"]), height=int(spec_data["height"]))
        rendered = fit_on_canvas(
            image,
            spec,
            bg_color,
            center_subject=center_subject,
            subject_threshold=subject_threshold,
            subject_padding_percent=subject_padding_percent,
            subject_saturation_threshold=subject_saturation_threshold,
            target_subject_height_percent=target_subject_height_percent,
            max_subject_width_percent=max_subject_width_percent,
        )
        if config["processing"].get("correct_export_quality", False):
            rendered = correct_export_quality(
                rendered,
                spec,
                bg_color,
                subject_threshold=subject_threshold,
                subject_padding_percent=subject_padding_percent,
                subject_saturation_threshold=subject_saturation_threshold,
                target_subject_luminance=float(config["processing"].get("target_subject_luminance", 172)),
                max_subject_brightness_adjustment=float(config["processing"].get("max_subject_brightness_adjustment", 0.12)),
                target_subject_height_percent=target_subject_height_percent,
                max_subject_width_percent=max_subject_width_percent,
            )
        target_dir = output_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{stem}_{name}.jpg"
        rendered.save(target, "JPEG", quality=92, optimize=True)
        exports[name] = target

    return exports


def ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg is required for video processing but was not found on PATH")
    return path


def video_filter(spec: ExportSpec, background_color: tuple[int, int, int]) -> str:
    color = "0x" + "".join(f"{channel:02x}" for channel in background_color)
    return (
        f"scale={spec.width}:{spec.height}:force_original_aspect_ratio=decrease,"
        f"pad={spec.width}:{spec.height}:(ow-iw)/2:(oh-ih)/2:color={color},"
        "fps=30,format=yuv420p"
    )


def run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1:] or result.stdout.strip().splitlines()[-1:]
        raise RuntimeError(f"ffmpeg failed: {detail[0] if detail else 'unknown error'}")


def process_video(source: Path, output_dir: Path, config: dict) -> dict[str, Path]:
    ffmpeg = ffmpeg_path()
    bg_color = tuple(config["processing"].get("background_color", [248, 248, 245]))
    settings = config.get("video_processing", {})
    duration = str(settings.get("max_duration_seconds", 12))
    crf = str(settings.get("crf", 23))
    preset = str(settings.get("preset", "veryfast"))
    stem = source_output_stem(source)

    exports: dict[str, Path] = {}
    for name, spec_data in config["video_exports"].items():
        spec = ExportSpec(width=int(spec_data["width"]), height=int(spec_data["height"]))
        target_dir = output_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{stem}_{name}.mp4"
        run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-i",
                str(source),
                "-t",
                duration,
                "-an",
                "-vf",
                video_filter(spec, bg_color),
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                crf,
                "-movflags",
                "+faststart",
                str(target),
            ]
        )
        exports[name] = target

    thumbnail_config = config.get("video_thumbnail")
    if thumbnail_config:
        spec = ExportSpec(width=int(thumbnail_config["width"]), height=int(thumbnail_config["height"]))
        name = "video_thumbnail"
        target_dir = output_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{stem}_{name}.jpg"
        run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-ss",
                str(thumbnail_config.get("timestamp_seconds", 1)),
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                video_filter(spec, bg_color),
                "-q:v",
                "3",
                str(target),
            ]
        )
        exports[name] = target

    return exports


def process_file(source: Path, output_dir: Path, config: dict) -> dict[str, Path]:
    kind = media_type(source)
    if kind == "image":
        return process_image(source, output_dir, config)
    if kind == "video":
        return process_video(source, output_dir, config)
    raise ValueError(f"unsupported file type: {source.suffix}")


def image_size(path: Path) -> str:
    try:
        with Image.open(path) as image:
            return f"{image.width}x{image.height}"
    except Exception:
        return ""


def media_dimensions(path: Path) -> str:
    if media_type(path) == "image":
        return image_size(path)
    return ""


def posting_pack_rows(source: Path, exports: dict[str, Path]) -> list[dict[str, str]]:
    rows = []
    for name, path in sorted(exports.items(), key=lambda item: USAGE_ORDER.get(item[0], 999)):
        guide = USAGE_GUIDE.get(name, {})
        rows.append(
            {
                "source_file": source.name,
                "export_type": name,
                "file_name": path.name,
                "dimensions": media_dimensions(path),
                "destination": guide.get("destination", ""),
                "usage": guide.get("usage", ""),
            }
        )
    return rows


def render_media_thumb(path: Path, size: tuple[int, int], background_color: tuple[int, int, int]) -> Image.Image:
    if media_type(path) == "image":
        try:
            return fit_on_canvas(open_image(path), ExportSpec(*size), background_color)
        except Exception:
            pass

    tile = Image.new("RGB", size, background_color)
    draw = ImageDraw.Draw(tile)
    label = path.suffix.upper().lstrip(".") or "FILE"
    draw.rectangle((40, 40, size[0] - 40, size[1] - 40), outline=(70, 70, 70), width=4)
    draw.text((size[0] // 2 - 35, size[1] // 2 - 10), label, fill=(40, 40, 40))
    return tile


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], max_chars: int, fill: tuple[int, int, int]) -> int:
    x, y = xy
    words = text.split()
    line = ""
    line_height = 22
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > max_chars and line:
            draw.text((x, y), line, fill=fill)
            y += line_height
            line = word
        else:
            line = candidate
    if line:
        draw.text((x, y), line, fill=fill)
        y += line_height
    return y


def create_contact_sheet(
    source: Path,
    exports: dict[str, Path],
    target: Path,
    config: dict,
) -> Path:
    settings = config.get("posting_pack", {})
    width = int(settings.get("contact_sheet_width", 2200))
    thumb_size = (
        int(settings.get("thumbnail_width", 420)),
        int(settings.get("thumbnail_height", 420)),
    )
    background_color = tuple(config.get("processing", {}).get("background_color", [248, 248, 245]))
    rows = posting_pack_rows(source, exports)

    margin = 48
    row_height = thumb_size[1] + 92
    header_height = 150
    height = header_height + max(1, len(rows)) * row_height + margin
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    pack_stem = source_output_stem(source)
    draw.text((margin, 36), f"Posting Pack: {pack_stem}", fill=(25, 25, 25))
    draw.text((margin, 72), "Use this sheet to pick the right file for Etsy and social posts.", fill=(80, 80, 80))
    draw.line((margin, 124, width - margin, 124), fill=(210, 210, 210), width=2)

    y = header_height
    for row in rows:
        path = exports[row["export_type"]]
        thumb = render_media_thumb(path, thumb_size, background_color)
        sheet.paste(thumb, (margin, y))
        text_x = margin + thumb_size[0] + 36
        draw.text((text_x, y + 8), row["export_type"], fill=(20, 20, 20))
        draw.text((text_x, y + 42), row["destination"], fill=(40, 80, 130))
        draw.text((text_x, y + 76), row["file_name"], fill=(70, 70, 70))
        if row["dimensions"]:
            draw.text((text_x, y + 108), row["dimensions"], fill=(100, 100, 100))
        draw_wrapped(draw, row["usage"], (text_x, y + 148), 90, (45, 45, 45))
        y += row_height

    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, "JPEG", quality=90, optimize=True)
    return target


def create_manifest_csv(source: Path, exports: dict[str, Path], target: Path) -> Path:
    rows = posting_pack_rows(source, exports)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_file", "export_type", "file_name", "dimensions", "destination", "usage"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return target


def create_manifest_html(source: Path, exports: dict[str, Path], target: Path) -> Path:
    rows = posting_pack_rows(source, exports)
    lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        f"<title>Posting Pack - {escape(source_output_stem(source))}</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:32px;color:#222;}",
        "table{border-collapse:collapse;width:100%;}",
        "th,td{border:1px solid #ddd;padding:10px;text-align:left;vertical-align:top;}",
        "th{background:#f3f3f3;}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Posting Pack: {escape(source_output_stem(source))}</h1>",
        "<p>Use this manifest to match each processed file to Etsy and social destinations.</p>",
        "<table>",
        "<tr><th>Export</th><th>File</th><th>Size</th><th>Destination</th><th>How to use it</th></tr>",
    ]
    for row in rows:
        lines.append(
            "<tr>"
            f"<td>{escape(row['export_type'])}</td>"
            f"<td>{escape(row['file_name'])}</td>"
            f"<td>{escape(row['dimensions'])}</td>"
            f"<td>{escape(row['destination'])}</td>"
            f"<td>{escape(row['usage'])}</td>"
            "</tr>"
        )
    lines.extend(["</table>", "</body>", "</html>"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def create_posting_pack(source: Path, exports: dict[str, Path], output_dir: Path, config: dict) -> dict[str, Path]:
    settings = config.get("posting_pack", {})
    if not settings.get("enabled", True) or not exports:
        return {}

    pack_stem = source_output_stem(source)
    pack_dir = output_dir / "posting_pack" / pack_stem
    return {
        "posting_pack_contact_sheet": create_contact_sheet(
            source,
            exports,
            pack_dir / f"{pack_stem}_posting_contact_sheet.jpg",
            config,
        ),
        "posting_pack_manifest_csv": create_manifest_csv(
            source,
            exports,
            pack_dir / f"{pack_stem}_posting_manifest.csv",
        ),
        "posting_pack_manifest_html": create_manifest_html(
            source,
            exports,
            pack_dir / f"{pack_stem}_posting_manifest.html",
        ),
    }


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "upload-ready"


def product_tokens(value: str) -> set[str]:
    stop_words = {"3d", "printed", "print", "prints", "the", "and", "with", "for", "hand"}
    return {
        token
        for token in re.split(r"[^a-z0-9]+", value.lower())
        if len(token) > 1 and token not in stop_words
    }


def resolve_tracker_db_path(settings: dict) -> Path | None:
    value = settings.get("tracker_db_path")
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def load_tracker_products(settings: dict) -> list[dict]:
    path = resolve_tracker_db_path(settings)
    if not path or not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(products)").fetchall()}
        category_select = "category" if "category" in columns else "'' AS category"
        order_by = "category, name" if "category" in columns else "name"
        rows = conn.execute(
            f"""
            SELECT id, name, sku, {category_select}, event_price, quantity_in_stock, discontinued
            FROM products
            WHERE COALESCE(discontinued, 0) = 0
            ORDER BY {order_by}
            """
        ).fetchall()
    return [dict(row) for row in rows]


def source_product_hint(media_items: list[dict], settings: dict) -> str:
    item_hints = {str(item.get("product_hint") or "").strip() for item in media_items if item.get("product_hint")}
    item_hints.discard("")
    if len(item_hints) == 1:
        return next(iter(item_hints))

    explicit = str(settings.get("product_hint") or "").strip()
    if explicit:
        return explicit

    configured_name = str(settings.get("default_product_name") or "").strip()
    if configured_name and slugify(configured_name) != "3d-printed-product":
        return configured_name

    parents = {
        Path(item["source"]).parent.name
        for item in media_items
        if item.get("source") and Path(item["source"]).parent.name not in {"", ".", "incoming"}
    }
    if len(parents) == 1:
        return next(iter(parents))

    return ""


def match_tracker_product(hint: str, settings: dict) -> dict | None:
    hint = hint.strip()
    if not hint:
        return None
    hint_slug = slugify(hint)
    hint_words = product_tokens(hint)
    best: tuple[float, dict] | None = None
    tied = False

    for product in load_tracker_products(settings):
        product_name = str(product.get("name") or "")
        sku = str(product.get("sku") or "")
        product_slug = slugify(product_name)
        if hint_slug == slugify(sku):
            return product
        if hint_slug == product_slug:
            return product

        score = 0.0
        if hint_words:
            name_words = product_tokens(product_name)
            if name_words:
                score = len(hint_words & name_words) / max(len(name_words), 1)
                if name_words <= hint_words:
                    score = max(score, 0.95)

        if score > 0 and (best is None or score > best[0]):
            best = (score, product)
            tied = False
        elif best and score == best[0]:
            tied = True

    minimum = float(settings.get("minimum_product_match_score", 0.75))
    if best and best[0] >= minimum and not tied:
        return best[1]
    return None


def json_from_command_output(output: str) -> dict:
    decoder = json.JSONDecoder()
    for index, char in enumerate(output):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def vision_source_image(media_items: list[dict]) -> Path | None:
    for item in media_items:
        exports = item.get("exports") or {}
        if exports.get("etsy_main"):
            return Path(exports["etsy_main"])
    for item in media_items:
        exports = item.get("exports") or {}
        if exports.get("video_thumbnail"):
            return Path(exports["video_thumbnail"])
    for item in media_items:
        source = Path(item["source"])
        if media_type(source) == "image":
            return source
    return None


def vision_product_prompt(products: list[dict]) -> str:
    candidates = "\n".join(
        f"- SKU: {product.get('sku') or ''} | Name: {product.get('name') or ''} | Category: {product.get('category') or ''}".strip()
        for product in products
    )
    return f"""Identify the exact Tracker product represented by this product photo.

Choose exactly one product from this Tracker candidate list only when the image clearly matches the complete product being sold. Do not choose a generic animal, accessory, soap bottle, or component if the photo shows a more specific product such as a soap holder, tray, stand, base, or set.

Important matching rules:
- Prefer the most specific complete product name and category.
- If the object includes a base, tray, holder area, or soap bottle, prefer a Soap Holder product over a standalone animal product or a soap product.
- If multiple products look similar and you cannot tell the exact Tracker product, return an empty product_name and confidence below 0.7.
- Never infer a product that is not in the candidate list.

Return only compact JSON with keys: product_name, sku, confidence.

Tracker candidates:
{candidates}
"""


def match_product_with_vision(media_items: list[dict], settings: dict) -> dict | None:
    if not settings.get("vision_match_enabled", False):
        return None

    image_path = vision_source_image(media_items)
    if not image_path or not image_path.exists():
        return None

    products = load_tracker_products(settings)
    if not products:
        return None

    model = str(settings.get("vision_model") or "openai/gpt-5.5")
    timeout = int(settings.get("vision_timeout_seconds", 120))
    command = [
        "openclaw",
        "infer",
        "model",
        "run",
        "--gateway",
        "--model",
        model,
        "--file",
        str(image_path),
        "--json",
        "--prompt",
        vision_product_prompt(products),
    ]
    proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        return None

    payload = json_from_command_output(proc.stdout)
    outputs = payload.get("outputs") or []
    text = ""
    if outputs and isinstance(outputs[0], dict):
        text = str(outputs[0].get("text") or "")
    guess = json_from_command_output(text)
    confidence = float(guess.get("confidence") or 0)
    if confidence < float(settings.get("vision_minimum_confidence", 0.78)):
        return None

    sku = str(guess.get("sku") or "").strip()
    product_name = str(guess.get("product_name") or guess.get("product_name_guess") or "").strip()
    if sku:
        match = match_tracker_product(sku, settings)
        if match:
            return match
    if product_name:
        return match_tracker_product(product_name, settings)
    return None


def upload_ready_settings(config: dict, media_items: list[dict] | None = None) -> dict:
    settings = config.get("upload_ready", {})
    resolved = {
        "enabled": settings.get("enabled", True),
        "product_name": settings.get("default_product_name", "3D Printed Product"),
        "price": str(settings.get("default_price", "")),
        "quantity": str(settings.get("default_quantity", "")),
        "sku": str(settings.get("default_sku", "")),
        "material": settings.get("default_material", "3D printed plastic / PLA"),
        "shop_name": settings.get("shop_name", "Bluegrass Maker Lab"),
        "max_auto_images": int(settings.get("max_auto_images", 4)),
        "max_auto_videos": int(settings.get("max_auto_videos", 1)),
        "require_product_match": bool(settings.get("require_product_match", False)),
        "sync_tracker_product_folders": bool(settings.get("sync_tracker_product_folders", False)),
        "tracker_db_path": settings.get("tracker_db_path", ""),
        "minimum_product_match_score": float(settings.get("minimum_product_match_score", 0.75)),
        "product_hint": settings.get("product_hint", ""),
        "vision_match_enabled": bool(settings.get("vision_match_enabled", False)),
        "vision_model": settings.get("vision_model", "openai/gpt-5.5"),
        "vision_minimum_confidence": float(settings.get("vision_minimum_confidence", 0.78)),
        "vision_timeout_seconds": int(settings.get("vision_timeout_seconds", 120)),
    }
    if media_items:
        hint = source_product_hint(media_items, settings)
        product = match_tracker_product(hint, settings)
        if not product:
            product = match_product_with_vision(media_items, settings)
        if product:
            price = product.get("event_price")
            quantity = product.get("quantity_in_stock")
            resolved.update(
                {
                    "product_name": product.get("name") or resolved["product_name"],
                    "sku": product.get("sku") or resolved["sku"],
                    "price": f"{float(price):.2f}" if price not in (None, "") and float(price) > 0 else resolved["price"],
                    "quantity": str(quantity) if quantity not in (None, "") else resolved["quantity"],
                    "category": product.get("category") or "",
                    "tracker_product_id": product.get("id"),
                    "product_match_hint": hint,
                }
            )
        elif resolved["require_product_match"]:
            resolved["product_match_error"] = f"no confident Tracker product match for '{hint or 'unnamed batch'}'"
    return resolved


def upload_ready_group_issue(media_items: list[dict], settings: dict) -> str | None:
    images = [item for item in media_items if media_type(Path(item["source"])) == "image"]
    videos = [item for item in media_items if media_type(Path(item["source"])) == "video"]
    max_auto_images = int(settings.get("max_auto_images", 4))
    max_auto_videos = int(settings.get("max_auto_videos", 1))
    has_product_match = bool(settings.get("tracker_product_id"))

    if len(images) > max_auto_images and not has_product_match:
        return f"skipped ambiguous upload-ready group: {len(images)} images is more than the {max_auto_images} image auto-pack limit"
    if len(videos) > max_auto_videos:
        return f"skipped ambiguous upload-ready group: {len(videos)} videos is more than the {max_auto_videos} video auto-pack limit"
    if settings.get("product_match_error"):
        return f"skipped upload-ready group: {settings['product_match_error']}"
    return None


def batch_slug(media_items: list[dict], settings: dict) -> str:
    product_slug = slugify(settings.get("product_name", ""))
    if product_slug and product_slug != "3d-printed-product":
        return product_slug
    stems = [Path(item["source"]).stem for item in media_items if item.get("source")]
    prefix = stems[0] if stems else "batch"
    suffix = stems[-1] if len(stems) > 1 else ""
    return slugify(f"{prefix}-{suffix}" if suffix and suffix != prefix else prefix)


def collect_exports(media_items: list[dict], export_name: str) -> list[Path]:
    paths = []
    for item in media_items:
        path = (item.get("exports") or {}).get(export_name)
        if path:
            paths.append(Path(path))
    return paths


def copy_upload_asset(source: Path, target: Path, max_size: tuple[int, int] | None = None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if max_size and source.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        image = Image.open(source).convert("RGB")
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        image.save(target, quality=92, optimize=True)
        return target
    shutil.copy2(source, target)
    return target


def copy_square_listing_asset(source: Path, target: Path, *, min_size: int = 600, max_size: int = 2000) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(source).convert("RGB")
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    canvas_size = max(min_size, image.width, image.height)
    if image.width < min_size or image.height < min_size:
        scale = max(min_size / image.width, min_size / image.height)
        next_size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(next_size, Image.Resampling.LANCZOS)
        canvas_size = max(min_size, image.width, image.height)
    if image.width != canvas_size or image.height != canvas_size:
        canvas = Image.new("RGB", (canvas_size, canvas_size), "white")
        canvas.paste(image, ((canvas_size - image.width) // 2, (canvas_size - image.height) // 2))
        image = canvas
    image.save(target, "JPEG", quality=92, optimize=True)
    return target


def photo_visual_metrics(path: Path) -> dict[str, float | str | bool]:
    image = Image.open(path).convert("RGB")
    bounds = subject_bounds(image, threshold=22, saturation_threshold=45)
    metrics: dict[str, float | str | bool] = {
        "file": path.name,
        "subject_found": bool(bounds),
        "subject_height_percent": 0.0,
        "subject_fill_percent": 0.0,
        "subject_luminance": 0.0,
        "background_luminance": background_luminance(image),
    }
    if bounds:
        left, top, right, bottom = bounds
        metrics["subject_height_percent"] = (bottom - top) / image.height
        metrics["subject_fill_percent"] = ((right - left) * (bottom - top)) / (image.width * image.height)
        metrics["subject_luminance"] = subject_luminance(image, bounds)
    return metrics


def create_photo_consistency_report(media_items: list[dict], target: Path) -> Path:
    candidates = collect_exports(media_items, "social_4x5") or collect_exports(media_items, "etsy_main")
    rows = [photo_visual_metrics(path) for path in candidates if path.exists()]
    warnings: list[str] = []
    heights = [float(row["subject_height_percent"]) for row in rows if row["subject_found"]]
    backgrounds = [float(row["background_luminance"]) for row in rows]

    for row in rows:
        if not row["subject_found"]:
            warnings.append(f"{row['file']}: subject was not confidently detected.")
            continue
        height = float(row["subject_height_percent"])
        background = float(row["background_luminance"])
        if height < 0.56:
            warnings.append(f"{row['file']}: product looks small in frame ({height:.0%} of image height).")
        if height > 0.84:
            warnings.append(f"{row['file']}: product looks very tight in frame ({height:.0%} of image height).")
        if background < 222:
            warnings.append(f"{row['file']}: background still reads gray/dim (median {background:.0f}/255).")

    if len(heights) > 1 and max(heights) - min(heights) > 0.16:
        warnings.append(f"Batch scale varies too much ({min(heights):.0%} to {max(heights):.0%} of image height).")
    if len(backgrounds) > 1 and max(backgrounds) - min(backgrounds) > 24:
        warnings.append(f"Batch background brightness varies too much ({min(backgrounds):.0f} to {max(backgrounds):.0f}/255).")

    lines = ["Photo consistency QA", ""]
    if rows:
        lines.append("Measurements:")
        for row in rows:
            lines.append(
                "- {file}: subject height {height:.0%}, fill {fill:.0%}, subject luminance {subject:.0f}/255, background {background:.0f}/255".format(
                    file=row["file"],
                    height=float(row["subject_height_percent"]),
                    fill=float(row["subject_fill_percent"]),
                    subject=float(row["subject_luminance"]),
                    background=float(row["background_luminance"]),
                )
            )
    else:
        lines.append("No image exports were available for visual QA.")
    lines.append("")
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("Warnings: none.")
    lines.append("")
    lines.append("Use this as a consistency check only. Trust the real product over an over-edited photo.")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def normalize_foaming_hand_soap_copy(value):
    replacements = {
        "bar-soap": "foaming-hand-soap",
        "bar soap": "foaming hand soap",
        "Bar soap": "Foaming hand soap",
        "soap dish": "soap bottle holder",
        "Soap dish": "Soap bottle holder",
        "Soap Dish": "Soap Bottle Holder",
        "#SoapDish": "#FoamingHandSoap",
        "plain dish": "plain bottle holder",
    }
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [normalize_foaming_hand_soap_copy(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_foaming_hand_soap_copy(item) for key, item in value.items()}
    return value


def varied_social_copy(profile: dict, settings: dict, combined: str) -> dict:
    product_name = settings["product_name"]
    shop_name = settings["shop_name"]
    seed = str(settings.get("copy_seed") or settings.get("sku") or product_name)

    if "soap holder" in combined or "soap dish" in combined:
        animal = product_name.replace("Soap Holder", "").replace("soap holder", "").strip() or "little helper"
        primary_options = [
            f"{product_name} is ready for sink duty: a useful Bath & Body Works foaming hand soap bottle holder with enough personality to make the counter feel less boring.",
            f"Small sink upgrade, big personality. This {animal.lower()} dresses up a foaming hand soap bottle and adds a little character to the bathroom or kitchen.",
            f"Printed, checked, and ready for the counter: {product_name}, made for Bath & Body Works foaming hand soap bottles, guest baths, and practical gift baskets.",
            f"Foaming hand soap finally gets a fun little display spot. This {animal.lower()} holder keeps the bottle dressed up without taking over the counter.",
            f"A little useful, a little ridiculous, and exactly the kind of sink-side helper I like making: {product_name}.",
        ]
        feed_options = [
            f"{product_name} from {shop_name}: a small-batch printed holder for Bath & Body Works foaming hand soap bottles, guest sinks, kitchen counters, and housewarming gifts.",
            f"Fresh from the print table: {product_name}. It dresses up a foaming hand soap bottle and gives the sink a little extra personality.",
            f"This {animal.lower()} foaming hand soap holder is one of those useful little pieces that makes a bathroom, kitchen, or gift basket feel more finished.",
            f"Small-batch printed by {shop_name}, this {product_name} is made for everyday foaming hand soap bottles with a little more character.",
            f"Sink setup looking too plain? {product_name} dresses up foaming hand soap and adds a playful printed accent without taking over the counter.",
        ]
        video_options = [
            f"{product_name} getting a quick spin before heading to sink duty.",
            f"A closer look at the details on this {animal.lower()} soap holder.",
            f"One small printed soap holder, checked from every angle.",
            f"{product_name} from the print table to the bathroom counter.",
            f"Turning this {animal.lower()} around so you can see the shape before it goes in the shop.",
        ]
        hook_options = [
            "POV: your foaming hand soap got a tiny helper.",
            "The sink did not ask for personality, but it got some anyway.",
            "A plain soap bottle would have been too easy.",
            "Useful bathroom decor, but make it small-batch printed.",
            "One more reason the guest bath might get compliments.",
        ]
    elif "fidget" in combined or "clicker" in combined:
        primary_options = [
            f"{product_name} is printed, checked, and ready for idle hands, desk breaks, and anyone who likes a small thing to fiddle with.",
            f"Fresh fidget batch: {product_name}. Small enough for a desk, fun enough to keep picking back up.",
            f"This one is made for the hands that need something to do between tasks: {product_name}, printed by {shop_name}.",
            f"Desk toy, pocket fiddle, small gift: {product_name} is ready for the next batch of restless hands.",
        ]
        feed_options = [
            f"{product_name} from {shop_name}: a small-batch printed fidget for desks, gift bags, and quick little brain breaks.",
            f"Fresh off the printer, this {product_name} is made for anyone who likes a small handheld toy within reach.",
            f"A simple little fidget with enough personality to earn a spot on the desk: {product_name}.",
            f"Printed in small batches, checked, and ready to ship: {product_name} for idle hands and office gifts.",
        ]
        video_options = [
            f"{product_name} in motion, which is honestly the whole point.",
            f"A quick look at how this {product_name} moves.",
            f"Fresh fidget test before this {product_name} heads to the shop.",
            f"{product_name} getting its close-up from the print table.",
        ]
        hook_options = [
            "POV: your desk needed something to fiddle with.",
            "For the hands that refuse to sit still.",
            "The printer made a tiny desk distraction.",
            "This is why fidgets need video.",
        ]
    else:
        primary_options = [
            f"Fresh off the printer: {product_name}. A small 3D printed piece from {shop_name}, ready to add character wherever it lands.",
            f"New small-batch print on the table: {product_name}. Printed, checked, and ready for gifting, display, or everyday use.",
            f"{product_name} just came through the print queue. It is the kind of small piece that makes a desk, shelf, or gift basket feel more personal.",
            f"Printed in Kentucky by {shop_name}: {product_name}, ready for someone who likes useful little things with personality.",
            f"Another small-batch print ready for the shop: {product_name}. Simple, fun, and made one layer at a time.",
        ]
        feed_options = [
            f"Fresh small-batch print from {shop_name}: {product_name}. Good for gifting, display, or adding personality to everyday spaces.",
            f"{product_name} is printed, checked, and ready for the shop. A small handmade-style piece for desks, shelves, gifts, or collections.",
            f"Made one layer at a time by {shop_name}, this {product_name} is a small printed piece with a little more character than the usual shelf filler.",
            f"Small gift, desk piece, display accent, or just something fun: {product_name} is ready for its next home.",
            f"Fresh out of the print queue: {product_name}. Small-batch made, packed with care, and ready for gifting or display.",
        ]
        video_options = [
            f"{product_name} from every angle, fresh from {shop_name}.",
            f"A quick spin of {product_name} before it heads into the shop.",
            f"{product_name} getting checked from all sides after printing.",
            f"One small print, one slow turn: {product_name}.",
            f"Fresh print table look at {product_name}.",
        ]
        hook_options = [
            "POV: the printer made the small version cute.",
            "Fresh off the print bed and ready for the shop.",
            "One layer at a time, then one quick close-up.",
            "Small-batch print check before it gets packed.",
            "The kind of tiny thing that makes a shelf less boring.",
        ]

    index = deterministic_index(seed, len(primary_options))
    prompt_options = [
        f"Which color should I print this {product_name} in next?",
        "Small-batch print, ready for its close-up.",
        "Made in Kentucky, one layer at a time.",
        "Would you keep this one or gift it?",
        "This batch is headed from the print table to the shop.",
        "Simple little print, but it has some personality.",
    ]
    prompt_start = deterministic_index(f"{seed}:prompts", len(prompt_options))
    prompts = [prompt_options[(prompt_start + offset) % len(prompt_options)] for offset in range(3)]

    profile.update(
        {
            "primary_caption": primary_options[index],
            "short_caption": [f"Fresh print: {product_name}.", f"New small-batch drop: {product_name}.", f"{product_name}, ready for the shop."][index % 3],
            "video_caption": video_options[index % len(video_options)],
            "feed_caption": feed_options[index % len(feed_options)],
            "reels_hook": hook_options[index % len(hook_options)],
            "caption_prompts": prompts,
        }
    )
    return profile


def product_copy_profile(settings: dict) -> dict:
    product_name = settings["product_name"]
    product_lower = product_name.lower()
    category = str(settings.get("category") or "").lower()
    combined = f"{product_lower} {category}"
    shop_name = settings["shop_name"]

    profile = {
        "title": f"{product_name} - 3D Printed Gift - Cute Desk Decor - Small Handmade Gift",
        "opener": f"Bring a little personality to a desk, shelf, gift basket, or display spot with this 3D printed {product_name}.",
        "good_for": [
            "Desk decor",
            "Small gifts",
            "Collectors",
            "Stocking stuffers",
            "Office or shelf display",
        ],
        "tags": [
            "3d printed gift",
            "desk decor",
            "cute gift",
            "stocking stuffer",
            "small gift",
            "maker gift",
            "printed decor",
            "novelty gift",
            "birthday gift",
            "office decor",
            "collectible",
            "handmade gift",
            "kentucky made",
        ],
        "primary_caption": f"Fresh off the printer: {product_name}. A small 3D printed piece from {shop_name}, ready to add a little character wherever it lands.",
        "short_caption": f"{product_name}, fresh from {shop_name}.",
        "video_caption": f"{product_name} from every angle. Printed by {shop_name}.",
        "caption_prompts": [
            f"Which color should I print this {product_name} in next?",
            "Small-batch print, ready for its close-up.",
            "Made in Kentucky, one layer at a time.",
        ],
        "reels_hook": "POV: the 3D printer made the practical version cute.",
        "feed_caption": f"Fresh small-batch print from {shop_name}: {product_name}. Good for gifting, display, or adding a little personality to the everyday stuff.",
        "hashtags": [
            "#BluegrassMakerLab",
            "#3DPrinted",
            "#3DPrinting",
            "#MakerBusiness",
            "#EtsySeller",
            "#HandmadeGift",
            "#DeskDecor",
            "#SmallBusiness",
            "#KentuckyMade",
            "#GiftIdeas",
        ],
    }

    if "soap holder" in combined or "soap dish" in combined:
        animal = product_name.replace("Soap Holder", "").replace("soap holder", "").strip() or "little helper"
        animal_key = animal.lower()
        soap_caption_sets = {
            "chicken": {
                "opener": f"Add a little farmhouse charm to the sink. This Chicken Soap Holder keeps bar soap handy while giving a kitchen, bathroom, guest bath, or gift basket a warm country touch.",
                "primary_caption": f"Farmhouse sink energy, minus the chores. This chicken soap holder keeps bar soap close and gives the bathroom or kitchen a little country charm.",
                "short_caption": f"Chicken Soap Holder: tiny farmhouse sink upgrade.",
                "video_caption": "A little chicken spin for the sink-side lineup.",
                "caption_prompts": [
                    "For the sink that needs a little cluck and character.",
                    "Kitchen sink, guest bath, or garden-shed wash station?",
                    "Small-batch printed for anyone who likes practical things with personality.",
                ],
                "reels_hook": "POV: your bar soap got a tiny farmhouse roommate.",
                "feed_caption": f"This Chicken Soap Holder is a small sink upgrade with farmhouse personality. Printed by {shop_name} for kitchens, bathrooms, and gift baskets that need something useful and cute.",
            },
            "duck": {
                "opener": f"Bring a little splash-zone humor to the counter. This Duck Soap Holder gives bar soap a useful place to sit while adding playful bathroom, kitchen, or guest-bath personality.",
                "primary_caption": f"Built for splash-zone duty. This duck soap holder brings a little bath-time humor to bar soap, guest sinks, and kitchen counters.",
                "short_caption": f"Duck Soap Holder, ready for splash duty.",
                "video_caption": "Duck Soap Holder on its way to the splash zone.",
                "caption_prompts": [
                    "The sink called. It wanted a duck.",
                    "This one belongs beside a bathroom sink, but the kitchen counter may argue.",
                    "A small useful gift for anyone who likes their decor a little playful.",
                ],
                "reels_hook": "POV: the soap dish understood the assignment.",
                "feed_caption": f"Fresh from {shop_name}: a Duck Soap Holder that keeps bar soap handy and makes the sink feel a little more fun.",
            },
            "flamingo": {
                "opener": f"Brighten up the sink without taking over the whole counter. This Flamingo Soap Holder keeps bar soap handy and adds a cheerful accent to a guest bath, bathroom, kitchen, or gift basket.",
                "primary_caption": f"Guest bath, but make it bright. This flamingo soap holder adds a little pink-leaning personality to bar soap without taking over the whole counter.",
                "short_caption": f"Flamingo Soap Holder for a brighter sink.",
                "video_caption": "Flamingo Soap Holder getting its close-up before guest-bath duty.",
                "caption_prompts": [
                    "A tiny pop of flamingo energy for the sink.",
                    "This one feels made for a guest bath or a cheerful kitchen counter.",
                    "Useful enough for everyday soap, fun enough to give as a housewarming add-on.",
                ],
                "reels_hook": "POV: the guest bath got the fun soap holder.",
                "feed_caption": f"The Flamingo Soap Holder is a small-batch printed sink accent from {shop_name}, made for bar soap, bright bathrooms, and cheerful gift baskets.",
            },
            "goose": {
                "opener": f"Give the sink a little helpful attitude. This Goose Soap Holder keeps bar soap parked while adding a playful accent to a kitchen, bathroom, guest bath, or housewarming gift.",
                "primary_caption": f"Sink-side goose behavior, but helpful. This goose soap holder keeps bar soap parked while adding a little harmless attitude to the counter.",
                "short_caption": f"Goose Soap Holder: useful, with attitude.",
                "video_caption": "Goose Soap Holder doing one last lap before sink duty.",
                "caption_prompts": [
                    "For anyone whose bathroom could use a tiny bit of goose attitude.",
                    "Would this go by your kitchen sink or guest bath?",
                    "Printed in a small batch, ready to guard the soap.",
                ],
                "reels_hook": "POV: the sink hired a goose to guard the soap.",
                "feed_caption": f"Goose Soap Holder from {shop_name}: a practical little bar-soap spot with just enough attitude for a kitchen, bathroom, or housewarming gift.",
            },
            "hedgehog": {
                "opener": f"Give bar soap a tidy little landing spot. This Hedgehog Soap Holder adds a small, useful accent to a bathroom sink, kitchen counter, guest bath, or gift basket.",
                "primary_caption": f"Small, useful, and just a little spiky-looking. This hedgehog soap holder gives bar soap a tidy landing spot without making the sink feel boring.",
                "short_caption": f"Hedgehog Soap Holder for a tidy little sink.",
                "video_caption": "Hedgehog Soap Holder showing off the sink-side details.",
                "caption_prompts": [
                    "A little hedgehog for the sink that keeps losing the soap.",
                    "Cute enough for a gift basket, practical enough to actually use.",
                    "Guest bath decor that still earns its counter space.",
                ],
                "reels_hook": "POV: your soap finally got a tidy little home.",
                "feed_caption": f"This Hedgehog Soap Holder is a useful little sink accent from {shop_name}, made for bar soap, guest baths, and small gifts that do more than sit there.",
            },
            "otter": {
                "opener": f"Let the otter handle sink duty. This Otter Soap Holder keeps bar soap within reach and adds a water-loving little accent to bathrooms, kitchens, guest sinks, or gift baskets.",
                "primary_caption": f"Let the otter hold the soap. This otter soap holder keeps the bar in reach and adds a playful little water-loving touch to the sink.",
                "short_caption": f"Otter Soap Holder, reporting for sink duty.",
                "video_caption": "Otter Soap Holder making the sink setup a little more fun.",
                "caption_prompts": [
                    "The most responsible otter in the bathroom.",
                    "This one feels right at home by water.",
                    "A small practical gift for anyone who loves useful-but-cute things.",
                ],
                "reels_hook": "POV: an otter volunteered to hold the soap.",
                "feed_caption": f"Fresh from {shop_name}: an Otter Soap Holder for bar soap, bathroom counters, kitchen sinks, and anyone who likes practical gifts with a little charm.",
            },
            "pig": {
                "opener": f"Add cheerful farmhouse personality to the counter. This Pig Soap Holder gives bar soap a real spot to sit while making a bathroom, kitchen, guest bath, or housewarming basket feel more fun.",
                "primary_caption": f"Farmhouse cute without the mud. This pig soap holder gives bar soap a real spot to sit and makes the sink feel a little more cheerful.",
                "short_caption": f"Pig Soap Holder: farmhouse sink charm.",
                "video_caption": "Pig Soap Holder taking a spin before heading to the sink.",
                "caption_prompts": [
                    "For the kitchen sink that needed a little farm-stand personality.",
                    "Would you put this pig in a bathroom, kitchen, or gift basket?",
                    "Small-batch printed and ready to make hand-washing slightly less boring.",
                ],
                "reels_hook": "POV: the farmhouse sink got a tiny soap helper.",
                "feed_caption": f"This Pig Soap Holder from {shop_name} keeps bar soap handy and adds a cheerful farmhouse touch to kitchens, bathrooms, and housewarming gifts.",
            },
        }
        soap_copy = soap_caption_sets.get(animal_key, {})
        profile.update(
            {
                "title": f"{product_name} - Foaming Hand Soap Bottle Holder - Cute Bathroom Decor",
                "opener": f"Make the sink a little less boring. This {product_name} dresses up Bath & Body Works foaming hand soap bottles while adding a playful 3D printed accent to a bathroom, kitchen, guest bath, or gift basket.",
                "good_for": [
                    "Bathroom sink decor",
                    "Kitchen sink hand soap",
                    "Guest bath gifts",
                    "Housewarming baskets",
                    "Animal lovers",
                    "Small handmade gifts",
                ],
                "tags": [
                    "foaming soap holder",
                    "hand soap holder",
                    "bath and body works",
                    "bathroom decor",
                    "kitchen sink",
                    "guest bath gift",
                    "animal soap holder",
                    "3d printed gift",
                    "housewarming gift",
                    "cute bathroom",
                    "foaming soap bottle",
                    "handmade gift",
                    "kentucky made",
                    "small gift",
                ],
                "primary_caption": f"This {animal.lower()} has one job: make foaming hand soap look cuter. 3D printed by {shop_name} and ready for bathroom, kitchen, or guest-bath duty.",
                "short_caption": f"A tiny sink upgrade: {product_name}.",
                "video_caption": f"{product_name} doing a slow spin before sink duty.",
                "caption_prompts": [
                    "The sink did not ask for personality, but it got some anyway.",
                    f"Would you put this {animal.lower()} by the bathroom sink or the kitchen sink?",
                    "Small-batch printed, practical enough to use, cute enough to gift.",
                ],
                "hashtags": [
                    "#BluegrassMakerLab",
                    "#SoapHolder",
                    "#FoamingHandSoap",
                    "#BathAndBodyWorksFinds",
                    "#BathroomDecor",
                    "#KitchenSink",
                    "#3DPrinted",
                    "#HandmadeGift",
                    "#HousewarmingGift",
                    "#SmallBusiness",
                    "#KentuckyMade",
                ],
            }
        )
        profile.update(soap_copy)
        profile = normalize_foaming_hand_soap_copy(profile)
    elif "fidget" in combined or "clicker" in combined:
        profile.update(
            {
                "title": f"{product_name} - 3D Printed Fidget Toy - Desk Toy - Small Gift",
                "opener": f"Keep your hands busy and your desk a little more fun. This {product_name} is a small-batch 3D printed fidget made for quick breaks, office desks, gift bags, and everyday fiddle time.",
                "good_for": [
                    "Desk fidgeting",
                    "Office gifts",
                    "Stocking stuffers",
                    "Small rewards",
                    "Fidget toy collectors",
                    "Birthday gifts",
                ],
                "tags": [
                    "fidget toy",
                    "desk toy",
                    "3d printed fidget",
                    "sensory toy",
                    "office gift",
                    "stocking stuffer",
                    "small gift",
                    "handheld toy",
                    "stress toy",
                    "maker gift",
                    "handmade gift",
                    "kentucky made",
                    "gift for kids",
                ],
                "primary_caption": f"Desk fidget, but make it small-batch. {product_name} is printed, packed, and ready for idle hands.",
                "short_caption": f"Fresh fidget drop: {product_name}.",
                "video_caption": f"{product_name} in motion - exactly how a fidget should be shown.",
                "caption_prompts": [
                    "This is the kind of thing your desk slowly adopts.",
                    "For anyone who needs something to click, spin, flex, or fiddle with.",
                    "Small enough for a desk, fun enough to keep picking up.",
                ],
                "hashtags": [
                    "#BluegrassMakerLab",
                    "#FidgetToy",
                    "#DeskToy",
                    "#3DPrinted",
                    "#SensoryToy",
                    "#OfficeGift",
                    "#StockingStuffer",
                    "#EtsySeller",
                    "#SmallBusiness",
                    "#KentuckyMade",
                ],
            }
        )
    elif "keychain" in combined:
        profile.update(
            {
                "title": f"{product_name} - 3D Printed Keychain - Backpack Charm - Small Gift",
                "opener": f"Add a little printed personality to keys, bags, backpacks, or gift baskets. This {product_name} is lightweight, small-batch made, and easy to gift.",
                "good_for": [
                    "Keys",
                    "Backpacks",
                    "Gift baskets",
                    "Party favors",
                    "Small souvenirs",
                    "Everyday carry",
                ],
                "tags": [
                    "3d printed keychain",
                    "keychain gift",
                    "backpack charm",
                    "bag charm",
                    "small gift",
                    "party favor",
                    "stocking stuffer",
                    "maker gift",
                    "handmade gift",
                    "kentucky made",
                    "cute keychain",
                    "gift ideas",
                    "printed accessory",
                ],
                "primary_caption": f"Keys, bags, backpacks - {product_name} is ready to tag along.",
                "short_caption": f"New keychain drop: {product_name}.",
                "video_caption": f"{product_name}, ready for keys or a backpack.",
            }
        )
    elif "hitch cover" in combined:
        profile.update(
            {
                "title": f"{product_name} - 3D Printed Hitch Cover - Vehicle Accessory - Gift",
                "opener": f"Give the trailer hitch a cleaner, more personal look with this 3D printed {product_name}. It is a small-batch vehicle accessory made by {shop_name}.",
                "good_for": [
                    "Trailer hitch decor",
                    "Vehicle gifts",
                    "Truck accessories",
                    "Jeep accessories",
                    "Outdoor lovers",
                    "Custom-style gifts",
                ],
                "tags": [
                    "hitch cover",
                    "trailer hitch",
                    "truck accessory",
                    "jeep accessory",
                    "vehicle gift",
                    "3d printed gift",
                    "car accessory",
                    "outdoor gift",
                    "handmade gift",
                    "kentucky made",
                    "maker gift",
                    "custom style",
                    "gift ideas",
                ],
                "primary_caption": f"The hitch gets a little personality with this {product_name}.",
                "short_caption": f"New hitch cover: {product_name}.",
                "video_caption": f"{product_name} close-up before it heads for the hitch.",
            }
        )
    elif any(word in combined for word in ["cross", "scene", "sign"]):
        profile.update(
            {
                "title": f"{product_name} - 3D Printed Decor - Handmade Shelf or Wall Accent",
                "opener": f"Add a small handmade accent to a shelf, desk, entry table, or gift basket. This {product_name} is 3D printed by {shop_name} in Kentucky.",
                "good_for": [
                    "Shelf decor",
                    "Desk decor",
                    "Entry table accents",
                    "Small gifts",
                    "Faith-inspired gifts",
                    "Home decor baskets",
                ],
                "tags": [
                    "3d printed decor",
                    "shelf decor",
                    "desk decor",
                    "home accent",
                    "small gift",
                    "handmade gift",
                    "kentucky made",
                    "maker gift",
                    "gift basket",
                    "office decor",
                    "printed decor",
                    "home gift",
                    "gift ideas",
                ],
                "primary_caption": f"A small printed accent with a little more character than the usual shelf filler: {product_name}.",
                "short_caption": f"New decor print: {product_name}.",
                "video_caption": f"{product_name} from the print table to the display shelf.",
            }
        )

    return varied_social_copy(profile, settings, combined)


def create_etsy_listing_text(product_slug: str, settings: dict, etsy_files: list[str]) -> str:
    product_name = settings["product_name"]
    price = settings["price"] or "[fill from Tracker]"
    quantity = settings["quantity"] or "[fill from Tracker]"
    sku = settings["sku"] or "[fill from Tracker]"
    material = settings["material"]
    shop_name = settings["shop_name"]
    profile = product_copy_profile(settings)
    tags = profile["tags"]
    upload_order = "\n".join(f"{index + 1}. {name}" for index, name in enumerate(etsy_files))
    return f"""Etsy Listing Packet

Product name: {product_name}
Upload-ready folder: {product_slug}
SKU: {sku}
Recommended price: {price}
Quantity: {quantity}

FILES TO UPLOAD
{upload_order}

ETSY STEP-BY-STEP
1. Open Etsy Shop Manager.
2. Go to Listings.
3. Click Add a listing.
4. Upload the JPG files above in numbered order.
5. Upload the MP4 file as the listing video if one is included.
6. Set the thumbnail using 01_MAIN_{product_slug}.jpg.
7. Paste the title below.
8. Choose the closest Etsy category for the product.
9. Listing type: Physical item.
10. Who made it: I did / My shop.
11. What is it: A finished product.
12. Renewal: Automatic.
13. Paste the description below.
14. Personalization: Off unless this product is intentionally customizable.
15. Price: {price}.
16. Quantity: {quantity}.
17. SKU: {sku}.
18. Variations: None unless this product has ready-to-fulfill color or size choices.
19. Add the tags below.
20. Materials/attributes: {material}.
21. Shipping: use your existing small 3D printed item shipping profile unless the package size is unusual.
22. Preview the listing.
23. Confirm first photo, title, price, SKU, quantity, shipping profile, and tags.
24. Publish.
25. After publishing, copy the Etsy listing URL back into Tracker if it does not sync automatically.

TITLE
{profile["title"]}

DESCRIPTION
{profile["opener"]}

This listing is for the exact style shown in the photos and video. Each piece is printed in small batches, checked, and packed by {shop_name}.

Good for:
{chr(10).join(f"- {item}" for item in profile["good_for"])}

Details:
- Product: {product_name}
- SKU: {sku}
- Material: {material}
- Made by {shop_name} in Kentucky

Because this is a 3D printed item, small layer lines or minor surface variations may be visible. That is normal for the process and part of how these pieces are made.

Care:
- Wipe clean with a damp cloth.
- Keep away from high heat.
- Do not put in a dishwasher.

TAGS
{chr(10).join(tags)}

PHOTO ALT TEXT
Photo 1: 3D printed {product_name.lower()} shown as the main listing photo.
Photo 2: Alternate view of the 3D printed {product_name.lower()}.
Photo 3: Detail or side view of the 3D printed {product_name.lower()}.

FINAL CHECK BEFORE PUBLISHING
- Photos are uploaded in numbered order.
- Video is uploaded if present.
- Thumbnail is centered.
- Title is pasted.
- Price is correct.
- Quantity is correct.
- SKU is correct.
- Tags are filled.
- Shipping profile is selected.
- Listing URL is added back to Tracker after publishing/sync.
"""


def create_tiktok_shop_listing_text(product_slug: str, settings: dict, image_files: list[str], video_file: str = "") -> str:
    product_name = settings["product_name"]
    price = settings["price"] or "[fill from Tracker]"
    quantity = settings["quantity"] or "[fill from Tracker]"
    sku = settings["sku"] or "[fill from Tracker]"
    material = settings["material"]
    shop_name = settings["shop_name"]
    profile = product_copy_profile(settings)
    upload_order = "\n".join(f"{index + 1}. {name}" for index, name in enumerate(image_files))
    return f"""TikTok Shop Listing Packet

Product name: {product_name}
Upload-ready folder: {product_slug}
SKU: {sku}
Recommended price: {price}
Quantity: {quantity}

FILES TO UPLOAD
{upload_order}
{f"Video: {video_file}" if video_file else "Video: [optional - no <=5MB TikTok Shop video included]"}

TIKTOK SHOP STEP-BY-STEP
1. Open TikTok Shop Seller Center.
2. Go to Products.
3. Click Add new product.
4. Upload the JPG files above in numbered order.
5. Use 01_MAIN_{product_slug}_tiktok-shop.jpg as the first product image.
6. Upload the video only if one is included.
7. Paste the title below.
8. Choose the closest TikTok Shop product category.
9. Fill product attributes honestly and consistently with the photos.
10. Paste the description below.
11. Set price to {price}.
12. Set available quantity to {quantity}.
13. Set seller SKU to {sku}.
14. Fill package/shipping details from the actual packed product.
15. Review compliance: main product is clear, images have no text/graphics/watermarks, and the listing describes exactly what ships.
16. Submit/publish after TikTok Shop validation passes.

TITLE
{profile["title"]}

DESCRIPTION
{profile["opener"]}

This listing is for the exact style shown in the photos. Each piece is printed in small batches, checked, and packed by {shop_name}.

Good for:
{chr(10).join(f"- {item}" for item in profile["good_for"])}

Details:
- Product: {product_name}
- SKU: {sku}
- Material: {material}
- Made by {shop_name} in Kentucky

Because this is a 3D printed item, small layer lines or minor surface variations may be visible. That is normal for the process and part of how these pieces are made.

Care:
- Wipe clean with a damp cloth.
- Keep away from high heat.
- Do not put in a dishwasher.

IMAGE NOTES
- TikTok Shop allows up to 9 square product images.
- Images must be at least 600 x 600 px.
- Main image should show the product clearly.
- Avoid added text, logos, borders, watermarks, and unrelated props.
"""


def create_social_text(settings: dict, has_video: bool) -> str:
    product_name = settings["product_name"]
    profile = product_copy_profile(settings)
    prompts = profile["caption_prompts"]
    hashtags = " ".join(profile["hashtags"])
    return f"""Ready-to-post social captions

Primary caption:
{profile["primary_caption"]}

{prompts[0]}

Short caption:
{profile["short_caption"]}

Video caption:
{profile["video_caption"]}

Alternate captions:
1. {prompts[0]}
2. {prompts[1]}
3. {prompts[2]}

TikTok/Reels hook:
{profile["reels_hook"]}

Facebook/Instagram feed:
{profile["feed_caption"]}

Hashtags:
{hashtags}

Posting order:
1. {"Post reel-short-video.mp4 first with reel-cover.jpg as the cover." if has_video else "Post instagram-facebook-feed.jpg first."}
2. Use instagram-facebook-feed.jpg for a still feed post later.
3. Use story-tiktok-photo.jpg for stories or TikTok photo mode.
"""


def create_buffer_instructions(has_video: bool) -> str:
    return f"""Buffer upload guide

Use 01_FEED_POST_IMAGE_buffer-safe-4x5.jpg for a single-image scheduled post.
Use 01_FEED_POST_IMAGE_01_buffer-safe-4x5.jpg, 01_FEED_POST_IMAGE_02_buffer-safe-4x5.jpg, etc. together for a multi-photo feed post.
Do not use 03_STORY_ONLY_IMAGE_9x16.jpg as a normal feed post. It is only for stories or vertical photo modes.
{"Use 02_REEL_TIKTOK_SHORT_video.mp4 for TikTok, Instagram Reels, Facebook Reels, and YouTube Shorts when present." if has_video else "This packet is image-only. Schedule it as a feed/photo post, not a reel/video post."}
{"Use reel-cover.jpg as the cover image when Buffer or the platform asks for one." if has_video else ""}

Buffer API phase:
- buffer-post-draft.json is a local draft payload for a future Buffer API push script.
- buffer-queue.csv is the human-editable scheduling row.
- Fill etsy_listing_url and scheduled_at before pushing or scheduling.
"""


def create_buffer_post_draft(settings: dict, caption: str, has_video: bool, feed_image_count: int = 1) -> dict:
    feed_images = [
        f"01_FEED_POST_IMAGE_{index:02d}_buffer-safe-4x5.jpg"
        for index in range(1, max(feed_image_count, 0) + 1)
    ]
    return {
        "status": "draft",
        "product_name": settings["product_name"],
        "sku": settings["sku"],
        "etsy_listing_url": "",
        "scheduled_at": "",
        "platforms": ["instagram", "facebook", "tiktok"],
        "post_type": "video" if has_video else "image",
        "caption": caption,
        "assets": {
            "feed_image": "01_FEED_POST_IMAGE_buffer-safe-4x5.jpg",
            "feed_images": feed_images,
            "story_image": "03_STORY_ONLY_IMAGE_9x16.jpg",
            **({"video": "02_REEL_TIKTOK_SHORT_video.mp4", "video_cover": "reel-cover.jpg"} if has_video else {}),
        },
        "notes": "Review caption, add Etsy listing URL, and confirm scheduled_at before pushing to Buffer.",
    }


def create_upload_ready_pack(media_items: list[dict], output_dir: Path, config: dict) -> tuple[Path | None, list[Path]]:
    settings = upload_ready_settings(config, media_items)
    if not settings["enabled"] or not media_items:
        return None, []
    issue = upload_ready_group_issue(media_items, settings)
    if issue:
        raise ValueError(issue)

    slug = batch_slug(media_items, settings)
    pack_dir = output_dir / "upload_ready" / slug
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    etsy_dir = pack_dir / "Etsy_Upload"
    tiktok_shop_dir = pack_dir / "TikTok_Shop_Upload"
    social_dir = pack_dir / "Social_Upload"
    buffer_dir = pack_dir / "Buffer_Upload"
    notes_dir = pack_dir / "Notes"
    for path in [etsy_dir, tiktok_shop_dir, social_dir, buffer_dir, notes_dir]:
        path.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    etsy_file_names: list[str] = []
    tiktok_shop_file_names: list[str] = []
    etsy_main = collect_exports(media_items, "etsy_main")
    etsy_gallery = collect_exports(media_items, "etsy_gallery")
    etsy_video = collect_exports(media_items, "etsy_video")
    social_4x5 = collect_exports(media_items, "social_4x5")
    social_9x16 = collect_exports(media_items, "social_9x16")
    social_reels = collect_exports(media_items, "social_reels")
    video_thumbnails = collect_exports(media_items, "video_thumbnail")

    if etsy_main:
        target = copy_upload_asset(etsy_main[0], etsy_dir / f"01_MAIN_{slug}.jpg")
        files.append(target)
        etsy_file_names.append(target.name)

    gallery_sources = []
    for candidate in [*etsy_gallery, *etsy_main[1:]]:
        if candidate not in gallery_sources:
            gallery_sources.append(candidate)
    for index, source in enumerate(gallery_sources[:8], start=2):
        target = copy_upload_asset(source, etsy_dir / f"{index:02d}_GALLERY_{source.stem}.jpg")
        files.append(target)
        etsy_file_names.append(target.name)

    if etsy_video:
        target = copy_upload_asset(etsy_video[0], etsy_dir / f"{len(etsy_file_names) + 1:02d}_VIDEO_{slug}.mp4")
        files.append(target)
        etsy_file_names.append(target.name)

    tiktok_shop_sources = []
    for candidate in [*etsy_main, *etsy_gallery]:
        if candidate not in tiktok_shop_sources:
            tiktok_shop_sources.append(candidate)
    for index, source in enumerate(tiktok_shop_sources[:9], start=1):
        prefix = "MAIN" if index == 1 else "GALLERY"
        name = f"{index:02d}_{prefix}_{slug if index == 1 else source.stem}_tiktok-shop.jpg"
        target = copy_square_listing_asset(source, tiktok_shop_dir / name)
        files.append(target)
        tiktok_shop_file_names.append(target.name)
    tiktok_shop_video_name = ""
    if etsy_video and etsy_video[0].stat().st_size <= 5 * 1024 * 1024:
        target = copy_upload_asset(etsy_video[0], tiktok_shop_dir / f"{len(tiktok_shop_file_names) + 1:02d}_VIDEO_{slug}_tiktok-shop.mp4")
        files.append(target)
        tiktok_shop_video_name = target.name

    if social_4x5:
        files.append(copy_upload_asset(social_4x5[0], social_dir / "instagram-facebook-feed.jpg"))
        files.append(
            copy_upload_asset(
                social_4x5[0],
                buffer_dir / "01_FEED_POST_IMAGE_buffer-safe-4x5.jpg",
                max_size=(1080, 1350),
            )
        )
        for index, source in enumerate(social_4x5[:10], start=1):
            files.append(
                copy_upload_asset(
                    source,
                    buffer_dir / f"01_FEED_POST_IMAGE_{index:02d}_buffer-safe-4x5.jpg",
                    max_size=(1080, 1350),
                )
            )
    if social_9x16:
        files.append(copy_upload_asset(social_9x16[0], social_dir / "story-tiktok-photo.jpg", max_size=(1080, 1920)))
        files.append(copy_upload_asset(social_9x16[0], buffer_dir / "03_STORY_ONLY_IMAGE_9x16.jpg", max_size=(1080, 1920)))
    if social_reels:
        files.append(copy_upload_asset(social_reels[0], social_dir / "reel-short-video.mp4"))
        files.append(copy_upload_asset(social_reels[0], buffer_dir / "02_REEL_TIKTOK_SHORT_video.mp4"))
    if video_thumbnails:
        files.append(copy_upload_asset(video_thumbnails[0], social_dir / "reel-cover.jpg"))
        files.append(copy_upload_asset(video_thumbnails[0], buffer_dir / "reel-cover.jpg"))

    social_settings = {**settings, "copy_seed": slug}
    listing_text = create_etsy_listing_text(slug, settings, etsy_file_names)
    for target in [etsy_dir / "listing-copy.txt", etsy_dir / "etsy-step-by-step.md"]:
        target.write_text(listing_text, encoding="utf-8")
        files.append(target)

    tiktok_shop_text = create_tiktok_shop_listing_text(slug, settings, tiktok_shop_file_names, tiktok_shop_video_name)
    for target in [tiktok_shop_dir / "listing-copy.txt", tiktok_shop_dir / "tiktok-shop-step-by-step.md"]:
        target.write_text(tiktok_shop_text, encoding="utf-8")
        files.append(target)

    tiktok_shop_csv = tiktok_shop_dir / "tiktok-shop-listing.csv"
    with tiktok_shop_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["status", "product_name", "sku", "price", "quantity", "title", "description", "image_files", "video_file"])
        profile = product_copy_profile(settings)
        writer.writerow(
            [
                "draft",
                settings["product_name"],
                settings["sku"],
                settings["price"],
                settings["quantity"],
                profile["title"],
                profile["opener"],
                ";".join(tiktok_shop_file_names),
                tiktok_shop_video_name,
            ]
        )
    files.append(tiktok_shop_csv)

    social_text = create_social_text(social_settings, bool(social_reels))
    captions = social_dir / "captions.txt"
    captions.write_text(social_text, encoding="utf-8")
    files.append(captions)

    buffer_notes = buffer_dir / "buffer-instructions.txt"
    buffer_notes.write_text(create_buffer_instructions(bool(social_reels)), encoding="utf-8")
    files.append(buffer_notes)

    buffer_draft = buffer_dir / "buffer-post-draft.json"
    buffer_draft.write_text(
        json.dumps(
            create_buffer_post_draft(
                social_settings,
                product_copy_profile(social_settings)["feed_caption"],
                bool(social_reels),
                len(social_4x5[:10]),
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    files.append(buffer_draft)

    buffer_queue = buffer_dir / "buffer-queue.csv"
    with buffer_queue.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "status",
                "product_name",
                "sku",
                "etsy_listing_url",
                "scheduled_at",
                "platforms",
                "post_type",
                "caption",
                "feed_image",
                "feed_images",
                "story_image",
                "video",
            ]
        )
        writer.writerow(
            [
                "draft",
                settings["product_name"],
                settings["sku"],
                "",
                "",
                "instagram;facebook;tiktok",
                "video" if social_reels else "image",
                product_copy_profile(social_settings)["feed_caption"],
                "01_FEED_POST_IMAGE_buffer-safe-4x5.jpg" if social_4x5 else "",
                ";".join(f"01_FEED_POST_IMAGE_{index:02d}_buffer-safe-4x5.jpg" for index in range(1, len(social_4x5[:10]) + 1)),
                "03_STORY_ONLY_IMAGE_9x16.jpg" if social_9x16 else "",
                "02_REEL_TIKTOK_SHORT_video.mp4" if social_reels else "",
            ]
        )
    files.append(buffer_queue)

    qa_report = create_photo_consistency_report(media_items, notes_dir / "photo-consistency-report.txt")
    files.append(qa_report)

    upload_first = pack_dir / "UPLOAD_ME_FIRST.txt"
    upload_first.write_text(
        """This folder is ready to upload.

Etsy:
1. Open Etsy_Upload/etsy-step-by-step.md first.
2. Upload the numbered files in Etsy_Upload in order.
3. Copy/paste the title, description, tags, SKU, price, quantity, alt text, and checklist from etsy-step-by-step.md.

TikTok Shop:
Open TikTok_Shop_Upload/tiktok-shop-step-by-step.md and upload the numbered square JPG files in order.
Use TikTok_Shop_Upload/tiktok-shop-listing.csv as the quick copy/paste row.

Social:
Use Social_Upload/reel-short-video.mp4 first if present. Captions are in Social_Upload/captions.txt.

Buffer:
Use Buffer_Upload/01_FEED_POST_IMAGE_buffer-safe-4x5.jpg for single-image scheduled posts.
Use the numbered Buffer_Upload/01_FEED_POST_IMAGE_##_buffer-safe-4x5.jpg files together for a multi-photo feed post.
Use Buffer_Upload/02_REEL_TIKTOK_SHORT_video.mp4 for reels/shorts/TikTok when present.
Do not use the 9x16 story image as a normal Buffer feed post.
Fill Buffer_Upload/buffer-queue.csv before scheduling.

Quality check:
Review Notes/photo-consistency-report.txt before using the main image.

No photo sorting needed.
""",
        encoding="utf-8",
    )
    files.append(upload_first)

    manifest = notes_dir / "upload-ready-manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "file", "purpose"])
        for path in files:
            section = path.parent.name if path.parent != pack_dir else "root"
            writer.writerow([section, path.name, "ready upload asset"])
    files.append(manifest)

    return pack_dir, files
