from mesa.optimization import pipeline_iterator, plot_heatmaps
import os

output_file = "tests/test_pipeline_iterator_dummy_script_test_file.csv"
print(os.getcwd())

def test_pipeline_iterator_dummy_script():
    parameters = {
        "param_a": [1, 0.75, 0.50, 0.25],
        "param_b": [1, 0.5, 0.25],
        "param_c": ["boxcar", "hann", "blackman"],
    }
    def function(param_a, param_b, param_c):
        return {"SS": 0.99, "p": 0.05, "unexpected_metric": 12.3}

    pipeline_iterator(parameters, function, output_file)
    return

def test_optimization_heatmap(): # Provide a dummy .csv file for testing
    plot_heatmaps(output_file,
                'param_a',
                'param_b',
                'param_c',
                ["SS", "p"], vmin=0, vmax=1, cmap='YlGnBu')
    return

# if os.path.exists(output_file):
#     os.remove(output_file)