from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - exercised only in broken installs.
    register_heif_opener = None


HEIC_EXTENSIONS = {".heic", ".heif"}


@dataclass(frozen=True)
class ConversionResult:
    source: Path
    target: Path
    status: str
    message: str = ""


def heic_sources(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in HEIC_EXTENSIONS else []
    return sorted(
        (path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in HEIC_EXTENSIONS),
        key=lambda path: path.as_posix().lower(),
    )


def jpeg_target(source: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        relative = Path(source.stem + ".jpg")
    else:
        relative = source.relative_to(input_root).with_suffix(".jpg")
    return output_root / relative


def convert_heic_file(source: Path, target: Path, *, quality: int, overwrite: bool = False) -> ConversionResult:
    if target.exists() and not overwrite:
        return ConversionResult(source, target, "skipped", "target exists")

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(source) as image:
            rendered = ImageOps.exif_transpose(image).convert("RGB")
            rendered.save(target, "JPEG", quality=quality, optimize=True)
    except Exception as exc:
        return ConversionResult(source, target, "failed", str(exc))
    return ConversionResult(source, target, "converted")


def convert_heic_tree(input_path: Path, output_root: Path, *, quality: int = 95, overwrite: bool = False) -> list[ConversionResult]:
    if register_heif_opener is None:
        raise RuntimeError("pillow-heif is required to convert HEIC/HEIF files")
    register_heif_opener()

    input_path = input_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    sources = heic_sources(input_path)
    results = []
    for source in sources:
        target = jpeg_target(source, input_path, output_root)
        results.append(convert_heic_file(source, target, quality=quality, overwrite=overwrite))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bulk convert HEIC/HEIF photos to JPEG copies.")
    parser.add_argument("input", help="HEIC/HEIF file or folder to convert")
    parser.add_argument("output", help="Folder where JPEG copies should be written")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality, 1-100. Default: 95")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JPEG targets")
    args = parser.parse_args(argv)

    quality = max(1, min(100, args.quality))
    results = convert_heic_tree(Path(args.input), Path(args.output), quality=quality, overwrite=args.overwrite)
    converted = sum(1 for result in results if result.status == "converted")
    skipped = sum(1 for result in results if result.status == "skipped")
    failed = sum(1 for result in results if result.status == "failed")

    for result in results:
        if result.status == "failed":
            print(f"failed: {result.source} -> {result.target}: {result.message}", flush=True)
        elif result.status == "skipped":
            print(f"skipped: {result.target} ({result.message})", flush=True)
        else:
            print(f"converted: {result.source} -> {result.target}", flush=True)

    print(f"summary: {converted} converted, {skipped} skipped, {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
