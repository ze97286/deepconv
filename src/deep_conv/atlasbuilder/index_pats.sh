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
    if [ -f "${output_file}" ] && [ -f "${output_file}.tbi" ]; then
        echo "  Output file and index already exist, skipping"
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
    
    # Create tabix index (using bed format)
    echo "  Creating tabix index..."
    tabix -p bed "$output_file"
    
    # Also create CSI index (for more complete sequence support)
    echo "  Creating CSI index..."
    tabix -p bed -C "$output_file"
    
    # Check if indexing was successful
    if [ $? -eq 0 ]; then
        echo "  Successfully created indices for $filename"
    else
        echo "  Error: Failed to create index for $filename"
        # Try with explicit sequence and begin columns
        echo "  Trying with explicit column specification..."
        tabix -s 1 -b 2 -e 2 "$output_file"
        if [ $? -eq 0 ]; then
            echo "  Successfully created index with explicit columns"
        else
            echo "  All indexing attempts failed"
        fi
    fi
done

echo "Processing complete. Converted and indexed $count PAT.GZ files in $OUTPUT_DIR."
echo "Original files in $TARGET_DIR remain unchanged."