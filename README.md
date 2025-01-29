# DFT_MEDA

This repository contains code used for refining the DFT into MEDA pipeline for spatio temporal data. The case scenario we are trying to cover is using the frequency espectra of the signals measured by different spacially located sensors.

./functions contains the files with the functiosn used for simulating data, plotting it, extracting features from the signals and testing.

./data contains the data (real or simulated) to analyze with MEDA

./simulation_cata contains the data that was used for generating the sinthetic sygnals.

main.py is a demo file showcasing the different functionalities implemented.

Processor.py contains the Processor class, used for working with the signal data.

generate_data.py is the script that generates sinthetic data given certain parameters.

main.m is a MATLAB script that runs the MEDA script on the data.

run.mat in a script that runst generate_data.py and main.m one after the other.

