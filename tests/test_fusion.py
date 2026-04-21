from mesa.fusion import unfold_2D

import numpy as np


def test_unfold_2D():
    shape = np.array((3, 4, 5, 7, 10))
    row_ids = [1,2]
    col_ids = [0,4]
    X = np.random.rand(*shape)
    result, _, _ = unfold_2D(X, rows=row_ids, cols=col_ids)
    
    assert result.shape == (np.prod(shape[row_ids]),np.prod(shape[col_ids]))
    
def test_unfold_2D_with_labels():
    X = np.arange(12)
    X = X.reshape((2,3,2))

    labels = [["YES", "NO"],
              ["burbuja", "cactus", "pétalo"],
              [1,2]]
    label_names = ["Y/N", "supernenas", "numbers"]

    X_unfold, labels_unfold, label_names_unfold = unfold_2D(X, rows=[], cols=[2,1,0], labels=labels)

    assert (labels_unfold[1][0] == [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]).all()
    assert (labels_unfold[1][1] == ['burbuja', 'burbuja', 'cactus', 'cactus', 'pétalo', 'pétalo',
                                    'burbuja', 'burbuja', 'cactus', 'cactus', 'pétalo', 'pétalo']).all()
    assert (labels_unfold[1][2] == ['YES', 'NO', 'YES', 'NO', 'YES', 'NO', 'YES', 'NO', 'YES', 'NO', 'YES', 'NO']).all()



def test_unfold_2D_with_label_names():
    X = np.arange(12)
    X = X.reshape((2,3,2))

    labels = [["YES", "NO"],
              ["burbuja", "cactus", "pétalo"],
              [1,2]]
    label_names = ["Y/N", "supernenas", "numbers"]

    X_unfold, labels_unfold, label_names_unfold = unfold_2D(X, rows=[], cols=[2,1,0], labels=labels, label_names=label_names)

    assert label_names_unfold[1] == ["numbers", "supernenas", "Y/N"]

    assert (labels_unfold[1][0] == [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]).all()
    assert (labels_unfold[1][1] == ['burbuja', 'burbuja', 'cactus', 'cactus', 'pétalo', 'pétalo',
                                    'burbuja', 'burbuja', 'cactus', 'cactus', 'pétalo', 'pétalo']).all()
    assert (labels_unfold[1][2] == ['YES', 'NO', 'YES', 'NO', 'YES', 'NO', 'YES', 'NO', 'YES', 'NO', 'YES', 'NO']).all()