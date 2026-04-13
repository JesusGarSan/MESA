""""
Fusion module of the MESA package.
Contains functions to unfold feature tensors and combine
extracted features from different signals.
"""
import numpy as np

def unfold_2D(X: np.ndarray, rows: list, cols: list):
    """
    Unfolds a multi-dimensional array X into a 2D matrix.
    
    Args:
        X: The input numpy array.
        rows: List of dimension indices to be mapped to the resulting matrix rows.
        cols: List of dimension indices to be mapped to the resulting matrix columns.
    """
    row_set = set(rows)
    col_set = set(cols)
    all_dims = set(range(X.ndim))
    provided_dims = row_set.union(col_set)
    
    ignored_dims = all_dims - provided_dims
    
    selection = [0 if i in ignored_dims else slice(None) for i in range(X.ndim)]
    X_sliced = X[tuple(selection)]
    
    remaining_dims = [i for i in range(X.ndim) if i not in ignored_dims]
    id_map = {old_id: new_id for new_id, old_id in enumerate(remaining_dims)}
    
    new_rows = [id_map[i] for i in rows]
    new_cols = [id_map[i] for i in cols]
    
    X_permuted = np.transpose(X_sliced, axes=new_rows + new_cols)
    
    row_size = np.prod([X_sliced.shape[i] for i in new_rows], dtype=int)
    col_size = np.prod([X_sliced.shape[i] for i in new_cols], dtype=int)
    
    return X_permuted.reshape(row_size, col_size)