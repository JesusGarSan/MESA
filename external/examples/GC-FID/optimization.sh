#!/usr/bin/env bash

# Define total signal duration in samples
SIGNAL_LENGTH=45001

# Define parameter arrays as fractions
# Window length: Fraction of the TOTAL SIGNAL duration
WINDOW_LENGTH_FRACTIONS=(1 0.5 0.333 0.25 0.20 0.10 0.05 0.025 0.01)

# Hop: Fraction of the calculated WINDOW_LENGTH
HOP_FRACTIONS=(1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)

WINDOWS=("boxcar" "hann" "blackman")

# CSV File initialization
CSV_FILE="external/examples/GC-FID/optimization.csv"

# Clear old optimization results if they exist to start fresh
if [ -f "$CSV_FILE" ]; then
    rm "$CSV_FILE"
    echo "Cleared old $CSV_FILE"
fi

echo "Starting optimization loop..."

# Nested loops to grid search parameters
for wl_frac in "${WINDOW_LENGTH_FRACTIONS[@]}"; do
    for hp_frac in "${HOP_FRACTIONS[@]}"; do
        for win in "${WINDOWS[@]}"; do
            
            # --- Dynamic Parameter Resolution ---
            # 1. Calculate raw window length (fraction of total signal)
            wl=$(echo "scale=0; ($SIGNAL_LENGTH * ($wl_frac)) / 1" | bc -l)
            
            # Ensure window length is at least 1 sample to avoid crashes
            if [ "$wl" -lt 1 ]; then wl=1; fi

            # 3. Calculate Hop (fraction of window length)
            hp=$(echo "scale=0; ($wl * ($hp_frac)) / 1" | bc -l)
            if [ "$hp" -lt 1 ]; then hp=1; fi
            
            echo "=================================================="
            echo "Fractions -> WL_Frac:$wl_frac, HOP_Frac:$hp_frac"
            echo "Resolved Values -> WL=$wl, HOP=$hp, WIN=$win"
            echo "=================================================="

            # 1. Run Python script with calculated raw values
            echo "" | python3 external/examples/GC-FID/GC_FID_optimization.py \
                --window_length "$wl" \
                --hop "$hp" \
                --window "$win"

            # 2. Export raw values as environment variables for MATLAB to read
            export OPT_WINDOW_LENGTH="$wl"
            export OPT_HOP="$hp"
            export OPT_WINDOW="$win"

            # 3. Run MATLAB script in CLI mode
            matlab -batch "addpath(genpath('external/examples/GC-FID/MEDA')); cd('external/examples/GC-FID'); GC_FID;"
            
        done
    done
done

echo "Optimization complete! Results stored in $CSV_FILE"