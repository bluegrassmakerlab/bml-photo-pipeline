# Bluegrass Maker Lab Photo Pipeline

Automated product photo cleanup for Bluegrass Maker Lab.

Drop raw product photos into the OneDrive incoming folder. The pipeline polls OneDrive with `rclone`, downloads new files, creates Etsy/social-ready exports, uploads the results, and archives the originals.

## OneDrive Folder Structure

Remote root:

```text
onedrive:Bluegrass Maker Lab/Product Photo Pipeline/
```

Subfolders:

```text
00_Incoming/          raw phone photos land here
10_Ready/Etsy_Main/   square 2000 x 2000 listing images
10_Ready/Etsy_Gallery/4:3 gallery images
10_Ready/Social_4x5/  vertical feed images
10_Ready/Social_9x16/ story/reel images
20_Needs_Review/      files that failed or need manual review
90_Archive/Originals/ original images after successful processing
```

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m bml_photo_pipeline --once
```

Continuous polling:

```bash
python -m bml_photo_pipeline --interval 300
```

## What It Does

- polls OneDrive via the existing `rclone` remote
- skips files it has already processed
- auto-orients phone photos using EXIF
- trims excess light-box border when possible
- improves white balance, contrast, brightness, and sharpness
- optionally removes the background when `rembg` is installed
- exports Etsy and social crops
- uploads processed images back to OneDrive
- moves successful originals to archive
- moves failed/problem files to needs-review

## Tuning

Edit [config/default.json](config/default.json). The defaults are conservative for light-box phone photos.

The first real batch should be 5-10 photos so the crop/brightness settings can be tuned before you run a whole product set.

