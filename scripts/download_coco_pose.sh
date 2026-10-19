#!/usr/bin/env bash
# Download the files required by configs/coco_pose_qwen.yaml.
#
# Data is intentionally placed under raw/ (which is gitignored).  Do not add
# the downloaded images or annotations to a commit.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_root="${RAW_ROOT:-${repo_root}/raw}"
dataset_root="${raw_root}/coco2017"
work_dir="${dataset_root}/.downloads"

val_url="http://images.cocodataset.org/zips/val2017.zip"
annotations_url="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

download() {
  local url="$1"
  local destination="$2"

  if [[ -f "$destination" ]]; then
    echo "Using existing archive: $destination"
    return
  fi

  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --retry-delay 2 --output "$destination" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --output-document="$destination" "$url"
  else
    echo "Error: install curl or wget to download COCO." >&2
    exit 1
  fi
}

if ! command -v unzip >/dev/null 2>&1; then
  echo "Error: install unzip to extract COCO archives." >&2
  exit 1
fi

mkdir -p "$dataset_root" "$work_dir"

if [[ ! -d "${dataset_root}/val2017" ]]; then
  val_archive="${work_dir}/val2017.zip"
  download "$val_url" "$val_archive"
  unzip -q "$val_archive" -d "$dataset_root"
else
  echo "Using existing images: ${dataset_root}/val2017"
fi

annotation_file="${dataset_root}/annotations/person_keypoints_val2017.json"
if [[ ! -f "$annotation_file" ]]; then
  annotations_archive="${work_dir}/annotations_trainval2017.zip"
  download "$annotations_url" "$annotations_archive"
  unzip -q "$annotations_archive" \
    "annotations/person_keypoints_val2017.json" -d "$dataset_root"
else
  echo "Using existing annotations: $annotation_file"
fi

image_count="$(find "${dataset_root}/val2017" -maxdepth 1 -type f -name '*.jpg' | wc -l | tr -d ' ')"
if [[ "$image_count" != "5000" ]]; then
  echo "Error: expected 5000 val2017 images, found $image_count." >&2
  exit 1
fi

echo "COCO pose data ready in: $dataset_root"
echo "This task does not require assets/; leave raw/ and assets/ uncommitted."
