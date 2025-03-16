#!/bin/bash

# Check if required tools are installed
if ! command -v tabix &> /dev/null; then
    echo "Error: tabix is not installed or not in PATH"
    echo "It is typically part of the htslib package"
    exit 1
fi

if ! command -v bgzip &> /dev/null; then
    echo "Error: bgzip is not installed or not in PATH"
    echo "It is typically part of the htslib package"
    exit 1
fi

# Directory to process (default is current directory)
TARGET_DIR="${1:-.}"
# Output directory for the bgzf files and indices
OUTPUT_DIR="${2:-${TARGET_DIR}/tabix_indexed}"

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Processing all PAT.GZ files from $TARGET_DIR (not including subdirectories)..."
echo "Converted files and indices will be placed in $OUTPUT_DIR"

# Get list of files to process (non-recursively)
pat_files=("$TARGET_DIR"/*.pat.gz)

# Check if any files matched the pattern
if [ ! -f "${pat_files[0]}" ]; then
    echo "No PAT.GZ files found in $TARGET_DIR"
    exit 0
fi

# Count total files
total_files=${#pat_files[@]}
echo "Found $total_files PAT.GZ files to process"

# Process all PAT.GZ files
count=0
for patfile in "${pat_files[@]}"; do
    # Extract just the filename without the path
    filename=$(basename "$patfile")
    output_file="$OUTPUT_DIR/$filename"
    
    count=$((count + 1))
    echo "[$count/$total_files] Processing: $filename"
    
    # Skip if output file and index already exist
    if [ -f "${output_file}" ] && [ -f "${output_file}.csi" ]; then
        echo "  Output file and CSI index already exist, skipping"
        continue
    fi
    
    echo "  Converting to BGZF format..."
    
    # Directly convert to bgzip format (files are already sorted)
    gzip -dc "$patfile" | bgzip > "$output_file"
    
    # Check for compression errors
    if [ $? -ne 0 ]; then
        echo "  Error: Failed to convert to BGZF format"
        rm -f "$output_file"
        continue
    fi
    
    # Create tabix index using explicit column specification
    echo "  Creating tabix index with explicit column specification..."
    # -s 1: sequence name in column 1 (chromosome)
    # -b 2: begin position in column 2
    # -e 2: end position in column 2 (same as begin for point data)
    # -C: create CSI index instead of TBI (supports larger sequences)
    tabix -s 1 -b 2 -e 2 -C "$output_file"
    
    # Check if indexing was successful
    if [ $? -eq 0 ]; then
        echo "  Successfully created CSI index for $filename"
    else
        echo "  Error: Failed to create index for $filename"
    fi
done

echo "Processing complete. Converted and indexed $count PAT.GZ files in $OUTPUT_DIR."
echo "Original files in $TARGET_DIR remain unchanged."