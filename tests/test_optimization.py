from mesa.optimization import pipeline_iterator
import os

def test_pipeline_iterator_dummy_script():
    parameters = {
        "param_a": [1, 0.75, 0.50, 0.25],
        "param_b": [1, 0.5, 0.25],
        "param_c": ["boxcar", "hann", "blackman"],
    }
    def function(param_a, param_b, param_c):
        return {"SS": 0.99, "p": 0.05, "unexpected_metric": 12.3}
    
    output_file = "external/tests/test_pipeline_iterator_dummy_script_test_file.csv"
    pipeline_iterator(parameters, function, output_file)
    os.remove(output_file)

    return