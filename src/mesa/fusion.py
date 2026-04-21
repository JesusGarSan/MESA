""""
Fusion module of the MESA package.
Contains functions to unfold feature tensors and combine
extracted features from different signals.
"""
import numpy as np
from typing import Literal, List

def unfold_2D(X: np.ndarray, rows: list, cols: list, labels:list = None, label_names:List[str] = None):
    """
    Unfolds a multi-dimensional array X into a 2D matrix. Optionally provides according label unfolding.
    
    Args:
        X: The input numpy array to be transformed.
        rows: List of dimension indices to be mapped to the resulting matrix rows. 
              The order determines the hierarchy: earlier indices contain later indices (Kronecker product order).
        cols: List of dimension indices to be mapped to the resulting matrix columns. 
              The order determines the hierarchy: earlier indices contain later indices (Kronecker product order).
        labels: List of lists containing labels for each dimension of X. 
                Each inner list must match the size of its corresponding dimension in X.
        label_names: List of strings providing a name for each dimension in X.

    Returns:
        X_reshaped: The 2D unfolded version of the input array.
        labels_unfold: A list [row_labels, col_labels] containing the expanded label arrays, or None.
        label_names_unfold: A list [row_label_names, col_label_names] containing the names of the 
                            dimensions assigned to rows and columns, or None.
    """

    # Label dimensions verifications
    if label_names is not None:
        if labels is not None:
            assert len(label_names) == len(labels), f"label_names length ({len(label_names)})  must match labels length ({len(labels)}))."

        for i, label in enumerate(labels):
            assert len(label) == X.shape[i], f'label "{label_names[i]}" length ({len(label)}) must match X.shape[{i}] ({X.shape[i]}).'

    if labels is not None:
        for i, label in enumerate(labels):
            assert len(label) == X.shape[i], f"label[{i}] length ({len(label)}) must match X.shape[{i}] ({X.shape[i]})."


    row_set = set(rows)
    col_set = set(cols)
    all_dims = set(range(X.ndim))
    provided_dims = row_set.union(col_set)
    ignored_dims = all_dims - provided_dims

    # Get subselection of X Tensor
    selection = [0 if i in ignored_dims else slice(None) for i in range(X.ndim)]
    X_sliced = X[tuple(selection)]

    remaining_dims = [i for i in range(X.ndim) if i not in ignored_dims]
    id_map = {old_id: new_id for new_id, old_id in enumerate(remaining_dims)}
    
    new_rows = [id_map[i] for i in rows]
    new_cols = [id_map[i] for i in cols]
    
    X_permuted = np.transpose(X_sliced, axes=new_rows + new_cols)
    
    row_size = np.prod([X_sliced.shape[i] for i in new_rows], dtype=int)
    col_size = np.prod([X_sliced.shape[i] for i in new_cols], dtype=int)

    
    labels_unfold = None
    label_names_unfold = None
    if labels:
            def unfold_axis_labels(axis_indices):
                unfolded_list = []
                # Use the actual order of axis_indices, not a set
                for i, axis_idx in enumerate(axis_indices):
                    # How many times to repeat each element (dimensions to the right)
                    repeat_count = np.prod([X.shape[dim] for dim in axis_indices[i+1:]], dtype=int)
                    # How many times to tile the whole block (dimensions to the left)
                    tile_count = np.prod([X.shape[dim] for dim in axis_indices[:i]], dtype=int)
                    
                    # Apply Kronecker logic: Tile(Repeat(label))
                    aux = np.repeat(labels[axis_idx], repeat_count)
                    aux = np.tile(aux, tile_count)
                    unfolded_list.append(aux)
                return unfolded_list

            row_labels = unfold_axis_labels(rows)
            col_labels = unfold_axis_labels(cols)
            
            labels_unfold = [row_labels, col_labels]
            
            if label_names is not None:
                # Map names using the same original index
                row_label_names = [label_names[i] for i in rows]
                col_label_names = [label_names[i] for i in cols]
                label_names_unfold = [row_label_names, col_label_names]

    return X_permuted.reshape(row_size, col_size), labels_unfold, label_names_unfold