from __future__ import annotations

import csv
from dataclasses import dataclass
from html import escape
from pathlib import Path
import shutil
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
USAGE_ORDER = {name: index for index, name in enumerate(USAGE_GUIDE)}


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

    draw.text((margin, 36), f"Posting Pack: {source.stem}", fill=(25, 25, 25))
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
        f"<title>Posting Pack - {escape(source.stem)}</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:32px;color:#222;}",
        "table{border-collapse:collapse;width:100%;}",
        "th,td{border:1px solid #ddd;padding:10px;text-align:left;vertical-align:top;}",
        "th{background:#f3f3f3;}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Posting Pack: {escape(source.stem)}</h1>",
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

    pack_dir = output_dir / "posting_pack" / source.stem
    return {
        "posting_pack_contact_sheet": create_contact_sheet(
            source,
            exports,
            pack_dir / f"{source.stem}_posting_contact_sheet.jpg",
            config,
        ),
        "posting_pack_manifest_csv": create_manifest_csv(
            source,
            exports,
            pack_dir / f"{source.stem}_posting_manifest.csv",
        ),
        "posting_pack_manifest_html": create_manifest_html(
            source,
            exports,
            pack_dir / f"{source.stem}_posting_manifest.html",
        ),
    }
