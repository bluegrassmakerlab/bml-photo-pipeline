#!/usr/bin/env bash
set -euo pipefail

ROOT="onedrive:Bluegrass Maker Lab/Product Photo Pipeline"

rclone mkdir "$ROOT"
rclone mkdir "$ROOT/00_Incoming"
rclone mkdir "$ROOT/10_Ready/Etsy_Main"
rclone mkdir "$ROOT/10_Ready/Etsy_Gallery"
rclone mkdir "$ROOT/10_Ready/Social_4x5"
rclone mkdir "$ROOT/10_Ready/Social_9x16"
rclone mkdir "$ROOT/20_Needs_Review"
rclone mkdir "$ROOT/90_Archive/Originals"

echo "Created OneDrive product photo folders under: $ROOT"

