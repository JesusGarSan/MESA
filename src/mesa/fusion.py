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
        X: The input numpy array.
        rows: List of dimension indices to be mapped to the resulting matrix rows. Lower id axis contain higher id axis.
        cols: List of dimension indices to be mapped to the resulting matrix columns. Lower id axis contain higher id axis.
    """

    # Label dimensions verifications
    if label_names is not None:
        if labels is not None:
            assert len(label_names) == len(labels), f"label_names length ({len(label_names)}) must match labels length ({len(labels)}))."

        for i, label in enumerate(labels):
            assert len(label) == X.shape[i], f'label "{label_names[i]}" length ({len(label)}) should must match X.shape[{i}] ({X.shape[i]}).'

    if labels is not None:
        for i, label in enumerate(labels):
            assert len(label) == X.shape[i], f"label[{i}] length ({len(label)}) should must match X.shape[{i}] ({X.shape[i]})."


    row_set = set(rows)
    col_set = set(cols)
    all_dims = set(range(X.ndim))
    provided_dims = row_set.union(col_set)
    ignored_dims = all_dims - provided_dims
    
    # Get subselection of labels
    # if labels is not None:
    #     labels = [labels[i] for i in provided_dims]
    # if label_names is not None:
    #     label_names = [label_names[i] for i in provided_dims]

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
        row_labels = []
        row_label_names = []

        row_list = list(row_set)
        for row_id, row in enumerate(row_list):
            smaller_dims_id  =  row_list[:row_id] # Left hand-side
            smaller_dims = [X.shape[i] for i in smaller_dims_id]
            larger_dims_id =  row_list[row_id+1:] # Right hand-side
            larger_dims = [X.shape[i] for i in larger_dims_id]
            aux = np.repeat(labels[row], np.prod(smaller_dims))
            aux = np.tile (aux, np.prod(larger_dims))
            row_labels.append(aux)
            if label_names is not None:
                row_label_names.append(label_names[row_id])
            
        col_labels = []
        col_label_names = []

        col_list = list(col_set)
        for col_id, col in enumerate(col_list):
            smaller_dims_id  =  col_list[:col_id] # Left hand-side
            smaller_dims = [X.shape[i] for i in smaller_dims_id]
            larger_dims_id =  col_list[col_id+1:] # Right hand-side
            larger_dims = [X.shape[i] for i in larger_dims_id]
            aux = np.repeat(labels[col], np.prod(smaller_dims))
            aux = np.tile (aux, np.prod(larger_dims))
            col_labels.append(aux)
            if label_names is not None:
                col_label_names.append(label_names[col_id])

        labels_unfold = [row_labels, col_labels]
        if label_names is not None:
            label_names_unfold = [row_label_names, col_label_names]

    return X_permuted.reshape(row_size, col_size), labels_unfold, label_names_unfold