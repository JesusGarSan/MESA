# MESA (Multivariate Exploratory Signal Analysis)



This project is structured to analyze data using Python and MATLAB. The project structure is divided into four main directories: `data/`, `simulation/`, `features/`, and `meda/`.

## Project Structure

- **data/**: Contains the data to be analyzed.
- **simulation/**: A module that contains the code used to generate simulated data.
- **features/**: A module that contains the code used to extract features from the signals.
- **meda/**: A module that contains the code used to apply multivariate analysis to the feature data. The code in this folder is in MATLAB.

## Requirements

- Python 3.x
- MATLAB R2021a or later
- Necessary Python libraries (specified in `requirements.txt`)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/your_project.git
    cd your_project
    ```

2. Install the necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Generate Simulated Data

To generate simulated data, run the following command:
```bash
python simulation/generate_data.py