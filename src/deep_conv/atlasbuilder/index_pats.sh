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

echo "Processing all PAT.GZ files from $TARGET_DIR..."
echo "Converted files and indices will be placed in $OUTPUT_DIR"

# Count total files to process
total_files=$(find "$TARGET_DIR" -name "*.pat.gz" | wc -l)

if [ "$total_files" -eq 0 ]; then
    echo "No PAT.GZ files found in $TARGET_DIR"
    exit 0
fi

echo "Found $total_files PAT.GZ files to process"

# Process all PAT.GZ files
count=0
for patfile in "$TARGET_DIR"/*.pat.gz; do
    # Extract just the filename without the path
    filename=$(basename "$patfile")
    output_file="$OUTPUT_DIR/$filename"
    
    # Check if file exists (in case no files match the pattern)
    if [ -f "$patfile" ]; then
        count=$((count + 1))
        echo "[$count/$total_files] Processing: $filename"
        
        # Skip if output file and index already exist
        if [ -f "${output_file}" ] && [ -f "${output_file}.tbi" ]; then
            echo "  Output file and index already exist, skipping"
            continue
        fi
        
        # Create a temporary file name
        temp_file="$OUTPUT_DIR/${filename%.gz}.temp"
        
        echo "  Preparing for tabix indexing..."
        
        # Uncompress to temporary file
        gzip -dc "$patfile" > "$temp_file"
        
        # Check for decompression errors
        if [ $? -ne 0 ]; then
            echo "  Error: Failed to decompress $filename"
            rm -f "$temp_file"
            continue
        fi
        
        # Sort the file by chromosome and position (required for tabix)
        echo "  Sorting file by chromosome and position..."
        sorted_temp="${temp_file}.sorted"
        sort -k1,1 -k2,2n "$temp_file" > "$sorted_temp"
        
        # Check for sort errors
        if [ $? -ne 0 ]; then
            echo "  Error: Failed to sort file"
            rm -f "$temp_file" "$sorted_temp"
            continue
        fi
        
        # Remove the original temp file
        rm -f "$temp_file"
        
        # Compress with bgzip
        echo "  Compressing with bgzip..."
        bgzip -c "$sorted_temp" > "$output_file"
        
        # Check for compression errors
        if [ $? -ne 0 ]; then
            echo "  Error: Failed to compress with bgzip"
            rm -f "$sorted_temp" "$output_file"
            continue
        fi
        
        # Remove sorted temp file
        rm -f "$sorted_temp"
        
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
        fi
    fi
done

echo "Processing complete. Converted and indexed $count PAT.GZ files in $OUTPUT_DIR."
echo "Original files in $TARGET_DIR remain unchanged."