import os
import matlab.engine
import GC_FID
from mesa.optimization import pipeline_iterator, plot_heatmaps
from mesa import optimization

if __name__ == "__main__":
    compute = False
    plot = True

    if compute:
        parameters = {
            "window_length_fraction": [1, 0.5, 0.3, 0.25, 0.20, 0.10, 0.05, 0.025, 0.01, 0.005, 0.0025],
            "hop_fraction": [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],

            "window_function": ["boxcar", "hann", "blackman"],
        }

        gc_fid_metrics = [
            "Mean_SS", "Mean_perc_SS",
            "SS_time", "SS_treatment", "SS_sex", "SS_order", "SS_residuals",
            "F_time", "F_treatment", "F_sex", "F_order", "F_residuals",
            "p_time", "p_treatment", "p_sex", "p_order"
        ]

        pipeline = optimization.MatlabPipelineWrapper(
            matlab_script_folder="external/examples/GC-FID",
            matlab_script_name="GC_FID_optimization",
            output_metrics = gc_fid_metrics,
            python_pipeline_hook = GC_FID.main,
            custom_added_path="external/examples/GC-FID/MEDA"
        )

        try:
            pipeline.start_session()
            pipeline_iterator(parameters, pipeline, "external/examples/GC-FID/optimization.csv")
        finally:
            pipeline.stop_session()
    if plot:
        # plot_heatmaps("external/examples/GC-FID/optimization.csv",
        #             'hop_fraction',
        #             'window_length_fraction',
        #             'window_function',
        #             ['Mean_SS', 'Mean_perc_SS'],
        #             cmap = "ocean_r")
        # plot_heatmaps("external/examples/GC-FID/optimization.csv",
        #             'hop_fraction',
        #             'window_length_fraction',
        #             'window_function',
        #             ['SS_time', 'SS_treatment', 'SS_sex', 'SS_order'],
        #             vmin = 0, vmax = 30,
        #             cmap = "ocean_r")
        fig = plot_heatmaps("external/examples/GC-FID/optimization.csv",
                    'hop_fraction',
                    'window_length_fraction',
                    'window_function',
                    # ['F_time', 'F_treatment', 'F_sex', 'F_order'],
                    ['F_time'],
                    vmin = 0, vmax = 30,
                    cmap = "ocean_r")
        # plot_heatmaps("external/examples/GC-FID/optimization.csv",
        #             'hop_fraction',
        #             'window_length_fraction',
        #             'window_function',
        #             ['SS_residuals'],
        #             vmin = 0, vmax = 100,
        #             cmap="ocean_r")
        # plot_heatmaps("external/examples/GC-FID/optimization.csv",
        #             'hop_fraction',
        #             'window_length_fraction',
        #             'window_function',
        #             ['p_time', 'p_treatment', 'p_sex', 'p_order'], vmin = 0, vmax = 0.10,
        #             cmap="YlGnBu_r")