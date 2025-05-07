import numpy as np
import pandas as pd
import pytest


def add_column(df, column_name, values):
    """Add a new column to a DataFrame with the given values."""
    if len(values) != len(df):
        raise ValueError("Length of values must match DataFrame length")
    df[column_name] = values
    return df


class TestDataFrameOperations:
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]})

    def test_add_column_with_list(self, sample_dataframe):
        """Test adding a column with a list of values."""
        values = [10, 20, 30, 40, 50]
        result = add_column(sample_dataframe, "C", values)

        assert "C" in result.columns
        assert list(result["C"]) == values

    def test_add_column_with_numpy_array(self, sample_dataframe):
        """Test adding a column with a NumPy array."""
        values = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        result = add_column(sample_dataframe, "D", values)

        assert "D" in result.columns
        np.testing.assert_array_almost_equal(result["D"].values, values)

    def test_add_column_with_wrong_length(self, sample_dataframe):
        """Test that adding a column with wrong length raises ValueError."""
        values = [10, 20, 30]

        with pytest.raises(ValueError) as excinfo:
            add_column(sample_dataframe, "E", values)

        assert "Length of values must match DataFrame length" in str(excinfo.value)
