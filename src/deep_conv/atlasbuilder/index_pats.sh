#!/bin/bash

# Check if bcftools is installed
if ! command -v bcftools &> /dev/null; then
    echo "Error: bcftools is not installed or not in PATH"
    echo "Please install bcftools and try again"
    exit 1
fi

# Directory to process (default is current directory)
TARGET_DIR="${1:-.}"

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist"
    exit 1
fi

echo "Indexing all PAT.GZ files in $TARGET_DIR..."

# Count total files to process
total_files=$(find "$TARGET_DIR" -name "*.pat.gz" | wc -l)

if [ "$total_files" -eq 0 ]; then
    echo "No PAT.GZ files found in $TARGET_DIR"
    exit 0
fi

echo "Found $total_files PAT.GZ files to index"

# Process all PAT.GZ files
count=0
for patfile in "$TARGET_DIR"/*.pat.gz; do
    # Check if file exists (in case no files match the pattern)
    if [ -f "$patfile" ]; then
        count=$((count + 1))
        echo "[$count/$total_files] Indexing: $patfile"
        
        # Check if index already exists
        if [ -f "${patfile}.csi" ]; then
            echo "  Index already exists, skipping"
            continue
        fi
        
        # Create CSI index
        bcftools index --csi "$patfile"
        
        # Check if indexing was successful
        if [ $? -eq 0 ]; then
            echo "  Successfully created index: ${patfile}.csi"
        else
            echo "  Error: Failed to create index for $patfile"
        fi
    fi
done

echo "Indexing complete. Created indexes for $count PAT.GZ files."