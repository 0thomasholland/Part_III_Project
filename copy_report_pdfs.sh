#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$ROOT_DIR/outputs/report/figures"

SOURCE_DIRS=(
  "$ROOT_DIR/work"
  "$ROOT_DIR/notebooks/figures"
)

mkdir -p "$DEST_DIR"

copied_count=0

for source_dir in "${SOURCE_DIRS[@]}"; do
  if [[ ! -d "$source_dir" ]]; then
    echo "Skipping missing directory: $source_dir"
    continue
  fi

  while IFS= read -r -d '' pdf_file; do
    base_name="$(basename "$pdf_file")"
    target_path="$DEST_DIR/$base_name"

    cp -f "$pdf_file" "$target_path"
    ((copied_count++))
  done < <(find "$source_dir" -type f -iname '*.pdf' -print0)
done

echo "Copied ${copied_count} PDF(s) to: $DEST_DIR"
