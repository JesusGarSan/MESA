from mesa.fusion import unfold_2D

import numpy as np

def test_unfold_2D():
    shape = np.array((3, 4, 5, 7, 10))
    row_ids = [1,2]
    col_ids = [0,4]
    X = np.random.rand(*shape)
    result = unfold_2D(X, rows=row_ids, cols=col_ids)
    
    assert result.shape == (np.prod(shape[row_ids]),np.prod(shape[col_ids]))