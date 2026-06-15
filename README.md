# Bluegrass Maker Lab Photo Pipeline

Automated product photo and video cleanup for Bluegrass Maker Lab.

Drop raw product photos and short product videos into the OneDrive incoming folder. For hands-off Etsy packets, put each product in a subfolder named after the Tracker product, such as `00_Incoming/Duck Soap Holder/`. The pipeline polls OneDrive with `rclone`, downloads new files, creates Etsy/social-ready exports based on file type, uploads the results, and archives the originals.

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
10_Ready/Posting_Packs/contact sheets and manifests that explain where each output goes
30_Upload_Ready/      hands-off Etsy/social upload packets
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
- creates a posting pack with a contact sheet, CSV manifest, and HTML manifest for each processed file
- matches product folders to Tracker products for SKU, price, quantity, and exact product name
- falls back to Gateway vision matching when a flat/unnamed photo batch needs product identification
- creates an upload-ready packet after each confident product batch with ordered Etsy assets, listing copy, social assets, captions, and an `UPLOAD_ME_FIRST.txt`
- adds a `Metricool_Upload` folder with a feed-safe 4:5 image, reel/short video, cover image, story-only image, and Metricool instructions
- skips ambiguous upload-ready packets instead of mixing multiple products into one folder
- uploads processed images back to OneDrive
- moves successful originals to archive
- moves failed/problem files to needs-review

## How To Use The Outputs

- `Etsy_Main`: use as the first Etsy listing image.
- `Etsy_Gallery`: use for alternate angles, detail shots, color variants, scale, and packaging.
- `Social_4x5`: use for Instagram and Facebook feed posts.
- `Social_9x16`: use for stories, TikTok photo mode, and vertical image posts.
- `Etsy_Video`: use as the Etsy listing video.
- `Social_Reels`: use for TikTok, Instagram Reels, Facebook Reels, and YouTube Shorts.
- `Video_Thumbnails`: use as the cover image for short-form videos.
- `Posting_Packs`: open the contact sheet or manifest when you want a quick guide for what file goes where.
- `30_Upload_Ready`: use this first when you want the fastest no-sorting path to Etsy/social posting. These packets are only created when the batch can be matched to a Tracker product.
- `Metricool_Upload`: use `01_FEED_POST_IMAGE_metricool-safe-4x5.jpg` for normal Metricool image posts. Use the 9:16 image only for stories/vertical photo modes.

## Hands-Off Product Batches

Use one incoming subfolder per product:

```text
00_Incoming/
  Duck Soap Holder/
    IMG_0001.jpeg
    IMG_0002.jpeg
    IMG_0003.MOV
  Chicken Soap Holder/
    IMG_0004.jpeg
    IMG_0005.jpeg
    IMG_0006.MOV
```

The folder name is matched against Tracker product names/SKUs. When the match is confident, the upload-ready pack uses Tracker values for product name, SKU, price, and current quantity.

If the folder name is missing or ambiguous, the pipeline can ask the OpenClaw Gateway model to identify the first product image, then match that answer back to Tracker. If the vision result is not confident enough, normal `10_Ready` exports and `Posting_Packs` are still created, but `30_Upload_Ready` is skipped so the pipeline does not make a wrong Etsy packet.

## Tuning

Edit [config/default.json](config/default.json). The defaults are conservative for light-box phone photos and simple tripod product videos.

The first real batch should be 5-10 photos/videos so the crop, brightness, and video framing settings can be tuned before you run a whole product set.
