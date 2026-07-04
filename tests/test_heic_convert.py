from pathlib import Path

from PIL import Image
from pillow_heif import register_heif_opener

from bml_photo_pipeline.heic_convert import convert_heic_tree, heic_sources, jpeg_target


def write_heic(path: Path, color: tuple[int, int, int]) -> None:
    register_heif_opener()
    image = Image.new("RGB", (32, 24), color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def test_heic_sources_finds_supported_files_recursively(tmp_path: Path) -> None:
    write_heic(tmp_path / "Product" / "IMG_0001.HEIC", (10, 20, 30))
    (tmp_path / "Product" / "notes.txt").write_text("skip", encoding="utf-8")

    sources = heic_sources(tmp_path)

    assert sources == [tmp_path / "Product" / "IMG_0001.HEIC"]


def test_jpeg_target_preserves_relative_folders(tmp_path: Path) -> None:
    source = tmp_path / "input" / "Bigfoot Soap Holder" / "IMG_0001.HEIC"

    target = jpeg_target(source, tmp_path / "input", tmp_path / "jpeg-out")

    assert target == tmp_path / "jpeg-out" / "Bigfoot Soap Holder" / "IMG_0001.jpg"


def test_convert_heic_tree_writes_jpeg_copies_and_skips_existing(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    source = input_dir / "Bigfoot Soap Holder" / "IMG_0001.heic"
    output_dir = tmp_path / "jpeg-out"
    write_heic(source, (80, 110, 140))

    results = convert_heic_tree(input_dir, output_dir)
    second_results = convert_heic_tree(input_dir, output_dir)

    target = output_dir / "Bigfoot Soap Holder" / "IMG_0001.jpg"
    assert results[0].status == "converted"
    assert target.exists()
    assert Image.open(target).format == "JPEG"
    assert second_results[0].status == "skipped"
