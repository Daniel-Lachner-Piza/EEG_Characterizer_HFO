#!/bin/bash

# Script to rename all .EDF files to .edf in the specified directory
# Usage: ./rename_edf_extensions.sh

# Set the target directory
TARGET_DIR="/work/jacobs_lab/EEG_Data/Clustering_Patho_HFO"

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR does not exist!"
    exit 1
fi

echo "Renaming .EDF files to .edf in: $TARGET_DIR"
echo "----------------------------------------"

# Count total files to process
total_files=$(find "$TARGET_DIR" -name "*.EDF" | wc -l)

if [ "$total_files" -eq 0 ]; then
    echo "No .EDF files found in $TARGET_DIR"
    exit 0
fi

echo "Found $total_files .EDF files to rename"
echo ""

# Counter for renamed files
count=0

# Find all .EDF files and rename them
find "$TARGET_DIR" -name "*.EDF" | while read -r file; do
    # Get the directory and base name
    dir=$(dirname "$file")
    base=$(basename "$file" .EDF)
    
    # New filename with .edf extension
    new_file="$dir/$base.edf"
    
    # Check if target file already exists
    if [ -e "$new_file" ]; then
        echo "Warning: $new_file already exists. Skipping $file"
    else
        # Rename the file
        mv "$file" "$new_file"
        if [ $? -eq 0 ]; then
            echo "Renamed: $(basename "$file") -> $(basename "$new_file")"
            count=$((count + 1))
        else
            echo "Error: Failed to rename $file"
        fi
    fi
done

echo ""
echo "Renaming complete!"
echo "Total files processed: $total_files"
