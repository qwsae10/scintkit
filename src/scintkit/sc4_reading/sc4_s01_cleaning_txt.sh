#!/bin/bash

# Set your folder (or use "." for current directory)
FOLDER="./"
FOLDER="/home/jxg200016/scratch/measurements"
# Loop through all .txt files in the folder
for file in "$FOLDER"/sept1[5-8]*.txt; do
    # Skip files that already end with _F.txt
    if [[ "$file" == *_F.txt ]]; then
        continue
    fi

    # Build output filename
    outfile="${file%.txt}_F.txt"
    if [[ -f "$outfile" ]]; then
	echo "Skipping already cleaned file: $outfile"
	continue
    fi
    # Apply grep command grep -vE '\bError\b|,,'
    grep -vE '\bError\b|,,' "$file" > "$outfile"

    echo "Processed: $file -> $outfile"
done

echo "All files processed."

