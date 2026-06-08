from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass


@dataclass(frozen=True)
class ExportSpec:
    width: int
    height: int


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def media_type(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return None


def open_image(path: Path) -> Image.Image:
    image = Image.open(path)
    return ImageOps.exif_transpose(image).convert("RGB")


def trim_background(image: Image.Image, threshold: int, padding_percent: float) -> Image.Image:
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
        return image

    y_indices, x_indices = np.where(mask)
    left, right = int(x_indices.min()), int(x_indices.max())
    top, bottom = int(y_indices.min()), int(y_indices.max())

    width, height = image.size
    pad = int(max(right - left, bottom - top) * padding_percent)
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(width - 1, right + pad)
    bottom = min(height - 1, bottom + pad)

    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right + 1, bottom + 1))


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


def polish(image: Image.Image, config: dict) -> Image.Image:
    settings = config["processing"]
    if settings.get("trim_background", True):
        image = trim_background(
            image,
            int(settings.get("trim_threshold", 22)),
            float(settings.get("trim_padding_percent", 0.08)),
        )

    image = remove_background_if_available(image, bool(settings.get("remove_background", False)))
    bg_color = tuple(settings.get("background_color", [248, 248, 245]))
    image = flatten(image, bg_color)

    image = ImageOps.autocontrast(image, cutoff=float(settings.get("autocontrast_cutoff", 1)))
    image = ImageEnhance.Brightness(image).enhance(float(settings.get("brightness", 1.05)))
    image = ImageEnhance.Contrast(image).enhance(float(settings.get("contrast", 1.08)))
    image = ImageEnhance.Color(image).enhance(float(settings.get("color", 1.02)))
    image = ImageEnhance.Sharpness(image).enhance(float(settings.get("sharpness", 1.18)))
    return image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=3))


def fit_on_canvas(image: Image.Image, spec: ExportSpec, background_color: tuple[int, int, int]) -> Image.Image:
    target_ratio = spec.width / spec.height
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
    image = polish(open_image(source), config)
    bg_color = tuple(config["processing"].get("background_color", [248, 248, 245]))
    stem = source.stem

    exports: dict[str, Path] = {}
    image_exports = config.get("image_exports", config.get("exports", {}))
    for name, spec_data in image_exports.items():
        spec = ExportSpec(width=int(spec_data["width"]), height=int(spec_data["height"]))
        rendered = fit_on_canvas(image, spec, bg_color)
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
    stem = source.stem

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
