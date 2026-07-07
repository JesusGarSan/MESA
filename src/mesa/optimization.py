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

import os
from typing import List, Callable, Optional
import matlab.engine
class MatlabPipelineWrapper:

    def __init__(
        self, 
        matlab_script_folder: str, 
        matlab_script_name: str,
        output_metrics: List[str],
        python_pipeline_hook: Optional[Callable] = None,
        custom_added_path: Optional[str] = None
    ):
        self.script_folder = matlab_script_folder
        self.script_name = matlab_script_name
        self.output_metrics = output_metrics
        self.python_pipeline_hook = python_pipeline_hook
        self.custom_added_path = custom_added_path
        self.eng = None

    def start_session(self):
        print("Starting persistent MATLAB engine session...")
        self.eng = matlab.engine.start_matlab()
        
        if self.custom_added_path:
            self.eng.eval(f"addpath(genpath('{self.custom_added_path}'));", nargout=0)
            
        self.eng.addpath(self.script_folder, nargout=0)
        print("Initializing MATLAB parallel pool (parpool)...")
        self.eng.eval("gcp();", nargout=0)
        print("Parallel pool is ready!")

    def stop_session(self):
        if self.eng:
            print("Shutting down MATLAB engine...")
            self.eng.quit()

    def __call__(self, *args, **kwargs):
        # Run Python script hook if provided, passing down the exact parameters
        if self.python_pipeline_hook:
            self.python_pipeline_hook(*args, **kwargs)

        # Dynamically fetch the MATLAB function reference
        matlab_function = getattr(self.eng, self.script_name)

        # Handle arbitrary number of output arguments dynamically
        n_outputs = len(self.output_metrics)
        matlab_outputs = matlab_function(nargout=n_outputs)

        # MATLAB engine returns a single scalar if nargout=1, or a tuple if nargout > 1
        if n_outputs == 1:
            matlab_outputs = (matlab_outputs,)

        # Construct the metrics results dictionary dynamically mapping to your names
        metrics_results = {}
        for metric_name, raw_value in zip(self.output_metrics, matlab_outputs):
            metrics_results[metric_name] = float(raw_value)

        return metrics_results

import os
from typing import List, Union
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


import os
from typing import List, Union, Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_heatmaps(
    data: Union[str, pd.DataFrame],
    x_axis: str,
    y_axis: str,
    file_axis: List[str],
    metrics: List[str],
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    shared_colorbar: bool = True,
    output_dir: str = "heatmaps",
):
    
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(data, str):
        df = pd.read_csv(data)
        base_name = os.path.splitext(os.path.basename(data))[0]
    elif isinstance(data, pd.DataFrame):
        df = data
        base_name = "heatmap"
    else:
        raise TypeError("The 'data' parameter must be a file path string or a pandas DataFrame.")

    for metric in metrics:
        print(f"\n--- Processing plots for metric: {metric} ---")

        # Determine colorscale boundaries for the current metric
        current_vmin = vmin
        current_vmax = vmax

        if shared_colorbar:
            if current_vmin is None:
                current_vmin = float(df[metric].min())
            if current_vmax is None:
                current_vmax = float(df[metric].max())

        for group_name, group_df in df.groupby(file_axis):

            if isinstance(group_name, tuple):
                group_string = "_".join(
                    f"{k}-{v}" for k, v in zip(file_axis, group_name)
                )
            else:
                group_string = f"{file_axis[0]}-{group_name}"

            print(f"Plotting heatmap for {group_string} ({metric})")

            heatmap_data = group_df.pivot(
                index=y_axis, columns=x_axis, values=metric
            )

            fig, ax = plt.subplots(1,1, figsize=(12, 7))
            
            # Pass cmap, vmin, and vmax dynamically to seaborn
            sns.heatmap(
                heatmap_data, 
                annot=True, 
                fmt=".2f", 
                cmap=cmap, 
                vmin=current_vmin, 
                vmax=current_vmax
            )

            plt.title(f"Heatmap of {metric}\n({group_string})")
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)

            # Temp. Comment out or delete
            # ax.set_title("Cluster centroid distance: Factor Treatment\nwindow function: hann", fontsize  = 16)
            # ax.set_xlabel("Hop fraction", fontsize = 16)    
            # ax.set_ylabel("Window length fraction", fontsize = 16)    

            file_name = f"{base_name}_{group_string}_{metric}.png"
            output_image_path = os.path.join(output_dir, file_name)

            plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
            plt.close()

    return fig

if __name__ == "__main__":

    parameters = {
        "window_length_fraction": [1, 0.75, 0.50, 0.25],
        "hop_fraction": [1, 0.5, 0.25],
        "window_function": ["boxcar", "hann", "blackman"],
    }

    # Define the exact outputs your specific MATLAB function yields
    gc_fid_metrics = [
        "SS_time", "SS_treatment", "SS_sex", "SS_order", "SS_residuals",
        "p_time", "p_treatment", "p_sex", "p_order"
    ]

    # Instantiate any arbitrary function dynamically
    my_pipeline = MatlabPipelineWrapper(
        matlab_script_folder="external/examples/GC-FID",
        matlab_script_name="GC_FID_optimization",
        output_metrics=gc_fid_metrics,
        custom_added_path="external/examples/GC-FID/MEDA"
    )

    try:
        my_pipeline.start_session()
        pipeline_iterator(parameters, my_pipeline)
    finally:
        my_pipeline.stop_session()
