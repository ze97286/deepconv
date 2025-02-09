#!/bin/bash

# /users/zetzioni/sharedscratch/deepconv/src/deep_conv/atlas/preprocess_pats.sh 4 /users/zetzioni/sharedscratch/atlas/classes/taps_atlas_class.csv /users/zetzioni/sharedscratch/atlas/pats/tmp4 32
# /users/zetzioni/sharedscratch/deepconv/src/deep_conv/atlas/preprocess_pats.sh 3 /users/zetzioni/sharedscratch/atlas/classes/taps_atlas_class.csv /users/zetzioni/sharedscratch/atlas/pats/tmp3 32

# Arguments
min_cpgs=$1
class_file=$2
output_dir=$3
threads=${4:-32}  # Default to 32 threads

export min_cpgs

mkdir -p "$output_dir"

# Create task list for GNU parallel
while IFS=, read -r pat_file group; do
    [[ $pat_file == "name" ]] && continue
    # Adjust the path to the .pat file
    pat_file=$(echo "$pat_file" | sed 's/\.beta/.pat.gz/' | sed 's/beta/pat\/Tissue/' | sed 's/\.calls\.all//')
    pat_basename=$(basename "$pat_file" .pat.gz)  # Extract the base name of the .pat file
    for chr in {1..22}; do
        echo "$pat_file,$group,$pat_basename,chr$chr"
    done
done < "$class_file" > "$output_dir/tasks.txt"

# Process task
process_file() {
    IFS=, read -r pat_file group pat_basename chr <<< "$1"
    echo "Processing: pat_file=$pat_file, group=$group, chr=$chr" >&2

    # Filter and process the .pat.gz file for the specified chromosome
    awk -v chr="$chr" -v min_cpgs="$min_cpgs" '
        BEGIN { FS="\t"; OFS="\t"; print "pos\tcount" }
        $1 == chr {
            pos = $2
            n = split($3, chars, "")
            for (i = 1; i <= n; i++) {
                ct_count = 0
                for (j = i; j <= n; j++) {
                    if (chars[j] ~ /[CT]/) {
                        ct_count++
                    }
                }
                if (ct_count >= min_cpgs)
                    print pos + i - 1, $4
            }
        }
    ' <(zcat "$pat_file") > "$output_dir/${group}_${pat_basename}_${chr}_index.txt"

    # Sanity check: Ensure no NaN values in the output
    if grep -q 'NaN' "$output_dir/${group}_${pat_basename}_${chr}_index.txt"; then
        echo "Error: File contains NaN values - $output_dir/${group}_${pat_basename}_${chr}_index.txt" >&2
        exit 1
    fi

    if [[ -f "$output_dir/${group}_${pat_basename}_${chr}_index.txt" ]]; then
        echo "File written successfully: $output_dir/${group}_${pat_basename}_${chr}_index.txt" >&2
    else
        echo "Failed to write file for: $pat_file $group $chr" >&2
    fi
}
export output_dir="${output_dir%/}"
export -f process_file

# Run tasks in parallel
cat "$output_dir/tasks.txt" | parallel -j "$threads" --progress process_file {}

# Cleanup temporary task list
rm "$output_dir/tasks.txt"

echo "Index generation complete! All files are saved with unique names in $output_dir"