"""
importing pandas library
"""

import pandas as pd

# Reading the values to variable dataframe df
df = pd.read_csv(
    "https://media.githubusercontent.com/media/Aquibgitt/Crime_Data_Analysis/main/Crime_Data_from_2020_to_Present.csv"
)


class Data:
    """
    This class contains all the functions needed for data reading,interpretation and cleaning.
    """

    def __init__(self):
        pass
        """
        Initializes the Data class with a DataFrame object.

        Parameters:
        data_frame (DataFrame): The DataFrame object to be processed and analyzed.
        """

    def details_H():
        """
        Retrieves and displays the first 5 rows of the dataset.

        Returns:
        DataFrame: First 5 rows of the dataset.
        """
        return df.head()

    def details_T():
        """
        Retrieves and displays the last 5 rows of the dataset.

        Returns:
        DataFrame: Last 5 rows of the dataset.
        """
        return df.tail()

    def attributes():
        """
        Retrieves and displays the attributes/columns of the dataset.

        Returns:
        Index: Attributes/columns of the dataset.
        """
        return df.columns

    def info():
        """
        Displays information about the dataset including count of non-null values and data types.

        Returns:
        None
        """
        info = df.info()
        return info

    def nullvalues():
        """
        Calculates and displays the count of null values for each attribute.

        Returns:
        Series: Count of null values for each attribute.
        """
        nullvalues = pd.isnull(df).sum()
        return nullvalues

    def Missing_values_percentage():
        """
        Calculates and displays the percentage of missing values for each attribute.

        Returns:
        Series: Percentage of missing values for each attribute.
        """
        return (pd.isnull(df).sum() / len(df)) * 100

    def shape():
        """
        Displays the dimensions (rows, columns) of the dataset.

        Returns:
        Tuple: Dimensions of the dataset (rows, columns).
        """
        shape = df.shape
        return shape

    def reshape():
        """
        Combining the two columna which are LAT and LON into a seperate column named coordinates.

        Returns:
        Tuple: Dimensions of the dataset (rows, columns).
        """

        # Merging 'LAT' and 'LON' columns into 'Coordinates'
        df["Coordinates"] = df["LAT"].astype(str) + ", " + df["LON"].astype(str)

        # Displaying the first few rows of the updated DataFrame with the new 'Coordinates' column
        print(df[["LAT", "LON", "Coordinates"]])

    def describe():
        """
        Displays the statistical summary of the dataset.

        Returns:
        DataFrame: Statistical summary of the dataset.
        """
        describe = df.describe()
        return describe

    def data_type():
        """
        Displays the data types of each attribute/column in the dataset.

        Returns:
        Series: Data types of each attribute/column.
        """
        data_types = df.dtypes
        return data_types

