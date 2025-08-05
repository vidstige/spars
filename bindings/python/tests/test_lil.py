from spars import lil_array

def test_lil():
    # Create a LIL array with some data
    data = lil_array((3, 3))
    
    # Check the shape of the array
    assert data.shape == (3, 3)
    
    # Build row-by-row
    data[0, 0] = 1.0
    data[0, 2] = 2.0
    data[1, 1] = 3.0
    data[2, 0] = 4.0
    data[2, 2] = 5.0

    # Check the values
    assert data[0, 0] == 1.0
    assert data[0, 1] == 0.0  # Default value for unset elements
    assert data[0, 2] == 2.0
    assert data[1, 0] == 0.0  # Default value for unset elements
    assert data[1, 1] == 3.0
    assert data[1, 2] == 0.0  # Default value for unset elements
    assert data[2, 0] == 4.0
    assert data[2, 1] == 0.0  # Default value for unset elements
    assert data[2, 2] == 5.0
