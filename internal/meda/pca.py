from sklearn.decomposition import PCA
import scipy

def load(filepath=".data/matrix.mat"):
    """
    Loads the data saved by the 'save' function from a .mat file.

    Args:
        filepath (str, optional): The path to the .mat file.
            Defaults to ".data/matrix.mat".

    Returns:
        tuple: A tuple containing the loaded matrix and, if it exists,
               the list of column names. Returns (None, None) if any
               error occurs during loading.
    """
    try:
        loaded_data = scipy.io.loadmat(filepath)
        matrix = loaded_data.get('matrix')
        column_names = loaded_data.get('column_names')

        # If a 1D array was saved, loadmat loads it as a row matrix,
        # here we return it to its original 1D shape if necessary.
        if matrix is not None and matrix.shape[0] == 1 and 'column_names' not in loaded_data:
            matrix = matrix.flatten()

        return matrix, column_names if column_names is not None else None
    except FileNotFoundError:
        print(f"Error: File not found at path: {filepath}")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None, None




if __name__ =="__main__":
    pass
    matrix, variables = load("tests/data/matrix.mat")

    print(matrix)
    print(variables)