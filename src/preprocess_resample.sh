#!/bin/bash

INPUT_DIR=$1
OUTPUT_DIR=$2

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*; do
  filename=$(basename "$file")
  name="${filename%.*}"
  
  ffmpeg -y -i "$file" \
    -ac 1 \
    -ar 16000 \
    -sample_fmt s16 \
    "$OUTPUT_DIR/$name.wav"
done

echo "Processing completed."
