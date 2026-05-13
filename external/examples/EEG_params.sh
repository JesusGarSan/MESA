#!/bin/bash
OUTPUT_FILE="external/examples/EEG_params.csv"

# Create/Overwrite file with header
echo "window_length,hop_fraction,PercSS" > "$OUTPUT_FILE"

WINDOWS=(60.0)
HOPS=(1.0 0.50 0.25 0.20)

for w in "${WINDOWS[@]}"; do
    for h in "${HOPS[@]}"; do
        # 1. Write the first two columns (no newline yet)
        printf "%s,%s," "$w" "$h" >> "$OUTPUT_FILE"
        
        # 2. Python (Only if Python adds more columns; otherwise skip)
        python3 external/examples/EEG_params.py --window "$w" --hop_fraction "$h"

        # 3. MATLAB (Appends the final value + newline)
        matlab -batch "EEG_params" -sd "external/examples"
    done
done