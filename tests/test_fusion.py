from mesa.fusion import unfold_2D

import numpy as np


def test_unfold_2D():
    shape = np.array((3, 4, 5, 7, 10))
    row_ids = [1,2]
    col_ids = [0,4]
    X = np.random.rand(*shape)
    result, _, _ = unfold_2D(X, rows=row_ids, cols=col_ids)
    
    assert result.shape == (np.prod(shape[row_ids]),np.prod(shape[col_ids]))
    
def test_unfold_2D_labels():
    X = np.arange(12)
    X = X.reshape((2,3,2))

    labels = [["YES", "NO"],
              ["burbuja", "cactus", "pétalo"],
              [1,2]]
    label_names = ["Y/N", "supernenas", "numbers"]

    X_unfold, labels_unfold, label_names_unfold = unfold_2D(X, rows=[], cols=[0,1,2], labels=labels, label_names=label_names)

    assert label_names_unfold[1] == ["Y/N", "supernenas", "numbers"]

    assert (labels_unfold[1][0] == ['YES', 'NO', 'YES', 'NO', 'YES', 'NO', 'YES', 'NO', 'YES', 'NO', 'YES', 'NO']).all()
    assert (labels_unfold[1][1] == ['burbuja', 'burbuja', 'cactus', 'cactus', 'pétalo', 'pétalo',
                                    'burbuja', 'burbuja', 'cactus', 'cactus', 'pétalo', 'pétalo']).all()
    assert (labels_unfold[1][2] == [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]).all()



def test_unfold_2D_with_labels():
    channels = ["Channel 1", "Channel 2", "Channel 3"]
    sensors = ["Sensor 1", "Sensor 2", "Sensor 3", "Sensor 4"]
    verbose = ["A", "B", "C", "D", "E"]    
    times = np.arange(7)
    freqs = np.arange(10)
    
    shape = np.array((3, 4, 5, 7, 10))
    row_ids = [1,3]
    col_ids = [0,4]
    X = np.random.rand(*shape)
    result, labels, _ = unfold_2D(X, rows=row_ids, cols=col_ids,
                               labels=[channels, sensors, verbose, times, freqs],
                               label_names=["channels", "sensors", "verbose", "times", "freqs"])
    
    assert result.shape == (np.prod(shape[row_ids]),np.prod(shape[col_ids]))