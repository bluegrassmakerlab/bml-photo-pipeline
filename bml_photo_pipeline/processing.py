from __future__ import annotations

import csv
from dataclasses import dataclass
from html import escape
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
    if settings.get("white_balance", True):
        image = white_balance_background(image, float(settings.get("white_balance_strength", 0.85)))

    if settings.get("trim_background", True):
        image = trim_background(
            image,
            int(settings.get("trim_threshold", 22)),
            float(settings.get("trim_padding_percent", 0.08)),
        )

    image = remove_background_if_available(image, bool(settings.get("remove_background", False)))
    bg_color = tuple(settings.get("background_color", [248, 248, 245]))
    image = flatten(image, bg_color)

    cutoff = float(settings.get("autocontrast_cutoff", 1))
    if settings.get("autocontrast_luminance", True):
        image = autocontrast_luminance(image, cutoff)
    else:
        image = ImageOps.autocontrast(image, cutoff=cutoff)
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
        rows = conn.execute(
            """
            SELECT id, name, sku, event_price, quantity_in_stock, discontinued
            FROM products
            WHERE COALESCE(discontinued, 0) = 0
            ORDER BY name
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

        score = 0.0
        if hint_slug == product_slug:
            score = 1.0
        elif hint_words:
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
        source = Path(item["source"])
        if media_type(source) == "image":
            return source
    return None


def vision_product_prompt(products: list[dict]) -> str:
    candidates = "\n".join(
        f"- {product.get('sku') or ''} | {product.get('name') or ''}".strip()
        for product in products
    )
    return f"""Identify the 3D printed product in this product photo.

Choose exactly one product from this Tracker candidate list only when the image clearly matches it. If uncertain, return product_name as an empty string and confidence below 0.7.

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


def copy_upload_asset(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def create_etsy_listing_text(product_slug: str, settings: dict, etsy_files: list[str]) -> str:
    product_name = settings["product_name"]
    price = settings["price"] or "[fill from Tracker]"
    quantity = settings["quantity"] or "[fill from Tracker]"
    sku = settings["sku"] or "[fill from Tracker]"
    material = settings["material"]
    shop_name = settings["shop_name"]
    tags = [
        "3d printed gift",
        "desk toy",
        "fidget toy",
        "cute gift",
        "stocking stuffer",
        "small gift",
        "maker gift",
        "printed decor",
        "novelty gift",
        "birthday gift",
        "office decor",
        "collectible toy",
        "handmade gift",
    ]
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
{product_name} - 3D Printed Gift - Cute Desk Toy - Fidget Friendly Decor - Small Handmade Gift

DESCRIPTION
Add a fun 3D printed piece from {shop_name} to a desk, shelf, gift basket, or display area. This listing is for the product shown in the photos and video.

Good for:
- Desk decor
- Small gifts
- Fidget-friendly gifts
- Collectors
- Stocking stuffers
- Office or shelf display

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


def create_social_text(settings: dict, has_video: bool) -> str:
    product_name = settings["product_name"]
    return f"""Ready-to-post social captions

Primary caption:
Fresh off the printer: {product_name}. A fun little 3D printed piece from Bluegrass Maker Lab, ready for a desk, shelf, gift basket, or display spot.

Short caption:
New 3D printed drop from Bluegrass Maker Lab.

Video caption:
{product_name} in motion. Printed by Bluegrass Maker Lab.

Hashtags:
#BluegrassMakerLab #3DPrinted #3DPrinting #MakerBusiness #EtsySeller #HandmadeGift #DeskDecor #FidgetToy #SmallBusiness #KentuckyMade

Posting order:
1. {"Post reel-short-video.mp4 first with reel-cover.jpg as the cover." if has_video else "Post instagram-facebook-feed.jpg first."}
2. Use instagram-facebook-feed.jpg for a still feed post later.
3. Use story-tiktok-photo.jpg for stories or TikTok photo mode.
"""


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
    social_dir = pack_dir / "Social_Upload"
    metricool_dir = pack_dir / "Metricool_Upload"
    notes_dir = pack_dir / "Notes"
    for path in [etsy_dir, social_dir, metricool_dir, notes_dir]:
        path.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    etsy_file_names: list[str] = []
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

    if social_4x5:
        files.append(copy_upload_asset(social_4x5[0], social_dir / "instagram-facebook-feed.jpg"))
        files.append(copy_upload_asset(social_4x5[0], metricool_dir / "01_FEED_POST_IMAGE_metricool-safe-4x5.jpg"))
    if social_9x16:
        files.append(copy_upload_asset(social_9x16[0], social_dir / "story-tiktok-photo.jpg"))
        files.append(copy_upload_asset(social_9x16[0], metricool_dir / "03_STORY_ONLY_IMAGE_9x16.jpg"))
    if social_reels:
        files.append(copy_upload_asset(social_reels[0], social_dir / "reel-short-video.mp4"))
        files.append(copy_upload_asset(social_reels[0], metricool_dir / "02_REEL_TIKTOK_SHORT_video.mp4"))
    if video_thumbnails:
        files.append(copy_upload_asset(video_thumbnails[0], social_dir / "reel-cover.jpg"))
        files.append(copy_upload_asset(video_thumbnails[0], metricool_dir / "reel-cover.jpg"))

    listing_text = create_etsy_listing_text(slug, settings, etsy_file_names)
    for target in [etsy_dir / "listing-copy.txt", etsy_dir / "etsy-step-by-step.md"]:
        target.write_text(listing_text, encoding="utf-8")
        files.append(target)

    social_text = create_social_text(settings, bool(social_reels))
    captions = social_dir / "captions.txt"
    captions.write_text(social_text, encoding="utf-8")
    files.append(captions)

    metricool_notes = metricool_dir / "metricool-instructions.txt"
    metricool_notes.write_text(
        """Metricool upload guide

Use 01_FEED_POST_IMAGE_metricool-safe-4x5.jpg for normal auto-published image posts.
Do not use 03_STORY_ONLY_IMAGE_9x16.jpg as a normal feed post. It is only for stories or vertical photo modes.
Use 02_REEL_TIKTOK_SHORT_video.mp4 for TikTok, Instagram Reels, Facebook Reels, and YouTube Shorts when present.
Use reel-cover.jpg as the cover image when the platform asks for one.

If Metricool says an image ratio must be between 3:4 and 1.91:1, pick the 4x5 feed image, not the 9x16 story image.
""",
        encoding="utf-8",
    )
    files.append(metricool_notes)

    upload_first = pack_dir / "UPLOAD_ME_FIRST.txt"
    upload_first.write_text(
        """This folder is ready to upload.

Etsy:
1. Open Etsy_Upload/etsy-step-by-step.md first.
2. Upload the numbered files in Etsy_Upload in order.
3. Copy/paste the title, description, tags, SKU, price, quantity, alt text, and checklist from etsy-step-by-step.md.

Social:
Use Social_Upload/reel-short-video.mp4 first if present. Captions are in Social_Upload/captions.txt.

Metricool:
Use Metricool_Upload/01_FEED_POST_IMAGE_metricool-safe-4x5.jpg for normal image posts.
Use Metricool_Upload/02_REEL_TIKTOK_SHORT_video.mp4 for reels/shorts/TikTok when present.
Do not use the 9x16 story image as a normal Metricool feed post.

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
