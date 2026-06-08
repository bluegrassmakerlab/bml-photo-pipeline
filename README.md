# Bluegrass Maker Lab Photo Pipeline

Automated product photo and video cleanup for Bluegrass Maker Lab.

Drop raw product photos and short product videos into the same OneDrive incoming folder. The pipeline polls OneDrive with `rclone`, downloads new files, creates Etsy/social-ready exports based on file type, uploads the results, and archives the originals.

## OneDrive Folder Structure

Remote root:

```text
onedrive:Bluegrass Maker Lab/Product Photo Pipeline/
```

Subfolders:

```text
00_Incoming/          raw phone photos and videos land here
10_Ready/Etsy_Main/   square 2000 x 2000 listing images
10_Ready/Etsy_Gallery/4:3 gallery images
10_Ready/Social_4x5/  vertical feed images
10_Ready/Social_9x16/ story/reel images
10_Ready/Etsy_Video/  square muted MP4 listing videos
10_Ready/Social_Reels/vertical muted MP4 reels/shorts
10_Ready/Video_Thumbnails/thumbnail stills from videos
20_Needs_Review/      files that failed or need manual review
90_Archive/Originals/ original images/videos after successful processing
```

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m bml_photo_pipeline --once
```

Video exports require `ffmpeg` on the host PATH. Photo processing still works without it; video files will move to Needs Review if `ffmpeg` is missing.

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
- detects videos in the same incoming folder as photos
- exports muted Etsy and social MP4 videos
- creates a thumbnail still from each video
- uploads processed images back to OneDrive
- moves successful originals to archive
- moves failed/problem files to needs-review

## Tuning

Edit [config/default.json](config/default.json). The defaults are conservative for light-box phone photos and simple tripod product videos.

The first real batch should be 5-10 photos/videos so the crop, brightness, and video framing settings can be tuned before you run a whole product set.
