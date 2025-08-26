#!/bin/bash

# Script to run inference for corrdiff input files at 3-hour intervals
# Hours: 0, 3, 6, 9, 12, 15, 18, 21, 24

# Set the base directory
BASE_DIR="/glade/work/li1995/earth2studio-project"
INPUT_DIR="$BASE_DIR/input"
OUTPUT_DIR="$BASE_DIR/output"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Array of forecast hours (every 3 hours from 0 to 24)
hours=(0 3 6 9 12 15 18 21 24)

echo "Starting inference for corrdiff input files..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Loop through each forecast hour
for hour in "${hours[@]}"; do
    input_file="$INPUT_DIR/corrdiff_inputs_f${hour}.npy"
    output_file="$OUTPUT_DIR/output_f${hour}.tar"
    
    # Check if input file exists
    if [ -f "$input_file" ]; then
        echo "Processing forecast hour $hour..."
        echo "Input file: $input_file"
        echo "Output file: $output_file"
        
        # Run the curl command
        curl -X POST \
             -F "input_array=@$input_file" \
             -F "samples=1" \
             -F "steps=16" \
             -o "$output_file" \
             http://localhost:8000/v1/infer
        
        # Check if the curl command was successful
        if [ $? -eq 0 ]; then
            echo "✓ Successfully processed forecast hour $hour"
        else
            echo "✗ Error processing forecast hour $hour"
        fi
        echo ""
        
        # Extract the tar file and rename extracted files with f{hour} suffix
        if [ -f "$output_file" ]; then
            echo "Extracting and renaming files from $output_file..."
            
            # Create a temporary directory for extraction
            temp_dir="$OUTPUT_DIR/temp_f${hour}"
            mkdir -p "$temp_dir"
            
            # Extract tar file to temporary directory
            tar -xf "$output_file" -C "$temp_dir"
            
            # Rename all extracted files with f{hour} suffix and move to output directory
            for extracted_file in "$temp_dir"/*; do
                if [ -f "$extracted_file" ]; then
                    filename=$(basename "$extracted_file")
                    extension="${filename##*.}"
                    basename_no_ext="${filename%.*}"
                    
                    # Add f{hour} suffix before the extension
                    if [ "$filename" = "$basename_no_ext" ]; then
                        # No extension
                        new_filename="${basename_no_ext}_f${hour}"
                    else
                        # Has extension
                        new_filename="${basename_no_ext}_f${hour}.${extension}"
                    fi
                    
                    mv "$extracted_file" "$OUTPUT_DIR/$new_filename"
                    echo "  Renamed: $filename -> $new_filename"
                fi
            done
            
            # Clean up temporary directory and original tar file
            rm -rf "$temp_dir"
            rm "$output_file"
            
            echo "✓ Extraction and renaming completed for forecast hour $hour"
        fi
    else
        echo "⚠ Warning: Input file not found for forecast hour $hour: $input_file"
        echo ""
    fi
done

echo "Inference completed for all available input files."
