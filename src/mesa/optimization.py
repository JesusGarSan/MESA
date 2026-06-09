import itertools
from typing import Callable, Dict, List
import csv



def pipeline_iterator(
    parameters: Dict[str, List],
    script: Callable,
    outfile: str = "optimization.csv",
):

    keys = list(parameters.keys())
    values = list(parameters.values())

    combinations = list(itertools.product(*values))
    print(f"Total iterations to run: {len(combinations)}\n")

    with open(outfile, mode="w", newline="", encoding="utf-8") as f:
        writer = None
        header_written = False

        for iteration, combination in enumerate(combinations):
            current_params = dict(zip(keys, combination))

            print(
                f"({iteration+1}/{len(combinations)}) Running for: {current_params}"
            )

            result = script(**current_params)

            # Initialize the CSV writer and header on the very first iteration
            if not header_written:
                # Combine parameter keys and whatever metric keys came back
                fieldnames = keys + list(result.keys())

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                header_written = True

            row_data = {**current_params, **result}
            writer.writerow(row_data)
            f.flush() 
    return


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


def plot_heatmaps(
    data_file: str,
    x_axis: str,
    y_axis: str,
    file_axis: List[str],
    metric: str,
):
    df = pd.read_csv(data_file)

    # Group by the columns that will define each separate plot/file
    for group_name, group_df in df.groupby(file_axis):

        # Robustness depending on if file_axis is a list or a single string
        if isinstance(group_name, tuple):
            group_string = "_".join(f"{k}-{v}" for k, v in zip(file_axis, group_name))
        else:
            group_string = f"{file_axis[0]}-{group_name}"

        print(f"Plotting heatmap for {group_string}")

        heatmap_data = group_df.pivot(index=y_axis, columns=x_axis, values=metric)

        # 1. Initialize a fresh matplotlib figure for this specific group
        plt.figure(figsize=(8, 6))

        # annot=True shows the numeric values in each cell
        # fmt=".2f" formats the numbers to 2 decimal places
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")

        plt.title(f"Heatmap of {metric}\n({group_string})")
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)

        base_name = os.path.splitext(data_file)[0]
        output_image_path = f"{base_name}_{group_string}_{metric}.png"

        # bbox_inches='tight' prevents axis labels from getting cut off
        plt.savefig(output_image_path, dpi=300, bbox_inches="tight")

        # 5. Close the figure to free up memory before the next loop iteration
        plt.close()

    return


if __name__ == "__main__":
    plot_heatmaps('external/examples/GC-FID/optimization.csv',
                  'hop_fraction',
                  'window_length_fraction',
                  'window',
                  'SS_time')