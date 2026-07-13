from mesa.fusion import unfold_2D, resample

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

import unittest
class test_resample(unittest.TestCase):

    def setUp(self):
        # Create a simple 1D array: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.x_1d = np.arange(10)
        # Create a 2D array: shape (3, 6)
        self.x_2d = np.arange(18).reshape(3, 6)

    def test_basic_1d_mean(self):
        """Test a clean non-overlapping 1D split using 'mean'."""
        X_out, ids_out = resample(self.x_1d, axis=0, slice_len=2, slide=2, method="mean")
        
        # Expected outputs: [0,1]->0.5, [2,3]->2.5, [4,5]->4.5, [6,7]->6.5, [8,9]->8.5
        expected_X = np.array([0.5, 2.5, 4.5, 6.5, 8.5])
        np.testing.assert_array_equal(X_out, expected_X)
        
        # Test original index lists
        self.assertEqual(len(ids_out), 5)
        np.testing.assert_array_equal(ids_out[0], np.array([0, 1]))
        np.testing.assert_array_equal(ids_out[-1], np.array([8, 9]))

    def test_slide_default_behavior(self):
        """Test that slide defaults to slice_len when omitted."""
        X_out_explicit, _ = resample(self.x_1d, axis=0, slice_len=5, slide=5)
        X_out_implicit, _ = resample(self.x_1d, axis=0, slice_len=5)
        np.testing.assert_array_equal(X_out_explicit, X_out_implicit)

    def test_overlapping_max_min(self):
        """Test overlapping windows (slide < slice_len) using max and min."""
        # arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # slice=3, slide=2 -> windows: [0,1,2], [2,3,4], [4,5,6], [6,7,8] (9 is left behind)
        X_max, ids_out = resample(self.x_1d, axis=0, slice_len=3, slide=2, method="max")
        expected_max = np.array([2, 4, 6, 8])
        np.testing.assert_array_equal(X_max, expected_max)
        
        X_min, _ = resample(self.x_1d, axis=0, slice_len=3, slide=2, method="min")
        expected_min = np.array([0, 2, 4, 6])
        np.testing.assert_array_equal(X_min, expected_min)

    def test_2d_axis_reduction(self):
        """Test reshaping on a specific axis of a multi-dimensional array."""
        # self.x_2d shape is (3, 6):
        # [[ 0,  1,  2,  3,  4,  5],
        #  [ 6,  7,  8,  9, 10, 11],
        #  [12, 13, 14, 15, 16, 17]]
        
        # Reduce along axis 1 (columns), slice_len=3, slide=3
        X_out, ids_out = resample(self.x_2d, axis=1, slice_len=3, slide=3, method="mean")
        
        # Column windows: [0,1,2] and [3,4,5]
        # Expected row 1 means: mean([0,1,2])=1.0, mean([3,4,5])=4.0
        expected_X = np.array([
            [1.0, 4.0],
            [7.0, 10.0],
            [13.0, 16.0]
        ])
        np.testing.assert_array_equal(X_out, expected_X)
        self.assertEqual(X_out.shape, (3, 2))

    def test_negative_axis(self):
        """Ensure negative axis parsing handles safely."""
        X_out, _ = resample(self.x_2d, axis=-1, slice_len=3, slide=3, method="max")
        expected_X = np.array([
            [2, 5],
            [8, 11],
            [14, 17]
        ])
        np.testing.assert_array_equal(X_out, expected_X)

    def test_validation_errors(self):
        """Verify that bad parameters trigger the correct exceptions."""
        # Non-ndarray input
        with self.assertRaises(TypeError):
            resample([1, 2, 3], axis=0, slice_len=2)
            
        # Empty array
        with self.assertRaises(ValueError):
            resample(np.array([]), axis=0, slice_len=2)
            
        # Out-of-bounds axis
        with self.assertRaises(ValueError):
            resample(self.x_1d, axis=5, slice_len=2)
            
        # Invalid slice length (too big)
        with self.assertRaises(ValueError):
            resample(self.x_1d, axis=0, slice_len=20)
            
        # Invalid slice length (negative/zero)
        with self.assertRaises(ValueError):
            resample(self.x_1d, axis=0, slice_len=0)
            
        # Invalid slide value
        with self.assertRaises(ValueError):
            resample(self.x_1d, axis=0, slice_len=2, slide=-1)
            
        # Typo in execution method
        with self.assertRaises(ValueError):
            resample(self.x_1d, axis=0, slice_len=2, method="average")