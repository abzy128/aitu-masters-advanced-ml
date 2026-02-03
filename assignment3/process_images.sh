#!/bin/bash

for f in ./raw_photos/*.heic ./raw_photos/*.HEIC; do
    [ -f "$f" ] || continue
    filename=$(basename "$f")
    magick "$f" -resize 640x640 "./resized_photos/${filename%.*}.jpg"
    echo "Resized: $f"
done
