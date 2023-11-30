# The_Inference_Class
import pandas as pd

# Importing the Visualization libraries
import matplotlib.pyplot as plt  # visualizing data
import seaborn as sns

pd.set_option("display.max_columns", None)
import warnings

warnings.filterwarnings("ignore")
from Crime_Analysis_Package import the_eda

# Reading the values and storing it in a dataframe using pandas
df = pd.read_csv(
    "https://media.githubusercontent.com/media/Aquibgitt/Crime_Data_Analysis/main/Crime_Data_from_2020_to_Present.csv"
)


class inference_Analysis:
    """
    This class contains the functions to analyse and get the conclusions of the research question.
    """

    def __init__(self):
        pass

    def crime_rate_by_month():
        """
        Analyzes the fluctuation of crime rates throughout the year in Los Angeles.

        Parameters:
        df (DataFrame): Crime dataset containing 'DATE OCC' column.

        Returns:
        None (Displays a side-by-side comparison of crime rates by month using Matplotlib and Seaborn)
        """
        df["DATE OCC"] = pd.to_datetime(df["DATE OCC"])
        df["Month"] = df["DATE OCC"].dt.month

        # Calculate crime counts per month
        crime_counts = df["Month"].value_counts().sort_index()

        # Create subplots using Matplotlib and Seaborn
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Matplotlib plot
        axes[0].plot(crime_counts.index, crime_counts.values, marker="o")
        axes[0].set_xlabel("Month")
        axes[0].set_ylabel("Crime Count")
        axes[0].set_title("Crime Rate by Month (Matplotlib)")

        crime_counts_df = pd.DataFrame(
            {"Month": crime_counts.index, "Crime_Count": crime_counts.values}
        )

        # Seaborn plot
        sns.lineplot(
            data=crime_counts_df,
            x=crime_counts.index,
            y=crime_counts.values,
            marker="o",
            ax=axes[1],
        )
        axes[1].set_xlabel("Month")
        axes[1].set_ylabel("Crime Count")
        axes[1].set_title("Crime Rate by Month (Seaborn)")

        plt.tight_layout()
        plt.show()

    def crime_by_area_type_count():
        """
        Examines the areas in Los Angeles with the highest incidence of specific crimes.

        Parameters:
        df (DataFrame): Crime dataset containing 'AREA' and 'Crm Cd Desc' columns.

        Returns:
        None (Displays crime counts by area and crime type using Matplotlib)
        """
        # Group data by area and crime description
        crime_area_counts = (
            df.groupby(["AREA", "Crm Cd Desc"]).size().unstack().fillna(0)
        )

        # Set up the figure and axis
        plt.figure(figsize=(20, 8))

        # Plotting crime counts by area and crime type
        crime_area_counts.plot(kind="barh", stacked=True)

        plt.xlabel("Count")
        plt.ylabel("Area")
        plt.title("Crime Counts by Area and Crime Type")

        # Adjust legend outside the plot area for clarity
        plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))

        # plt.tight_layout()
        plt.show()

    def crime_by_age_group():
        """
        Explores the correlation between age groups and the types of crime committed in Los Angeles.

        Parameters:
        df (DataFrame): Crime dataset containing 'Age_Group' and 'Crm Cd Desc' columns.

        Returns:
        None (Displays a side-by-side comparison of crime distribution by age group using Matplotlib and Seaborn)
        """
        data_frame = the_eda.eda_Exploration.binning()

        # data_frame = The_EDA.eda_Exploration.binning()
        # Correlation between age group and crime description.
        # Create a pivot table to count occurrences of each age group and crime description with the Heat Map.
        correlation_data = (
            data_frame.groupby(["Age_Group", "Crm Cd Desc"])
            .size()
            .unstack(fill_value=0)
        )

        ## Setting the figure size and DPI for better resolution (We can change the size of figure anytime needed)
        plt.figure(figsize=(70, 20), dpi=100)

        # Create a heatmap with a different color palette
        sns.heatmap(
            correlation_data,
            cmap="YlGnBu",
            annot=True,
            fmt="d",
            cbar_kws={"label": "Count"},
        )
        plt.title("Correlation Between Age Groups and Crime Descriptions", size=40)
        plt.xlabel("Crime Description", size=40)
        plt.ylabel("Age_Group", size=40)

        plt.show()

    def crime_by_area_type():
        """
        Generates a heatmap illustrating the correlation between top 5 crime types and areas using Seaborn.

        Returns:
        None

        Description:
        This function identifies the top 5 most frequent crime types and computes their correlation with geographical areas. It filters the dataset to include only these top 5 crime types and subsequently visualizes their correlations with areas using a Seaborn heatmap. By focusing on the most prevalent crimes, this heatmap provides a clearer view of the specific areas most affected by these crimes, facilitating targeted interventions and law enforcement efforts in high-crime regions.
        """
        # Calculate the top 5 crime types
        top_crimes = df["Crm Cd Desc"].value_counts().head(5).index

        # Filter the DataFrame to include only the rows with the top 5 crime types
        filtered_df = df[df["Crm Cd Desc"].isin(top_crimes)]

        # Calculate the correlation between the filtered crime types and areas
        crime_area_correlation = (
            filtered_df.groupby(["Crm Cd Desc", "AREA"]).size().unstack(fill_value=0)
        )

        # Setting the figure size and DPI for better resolution
        plt.figure(figsize=(30, 20))

        # Create a heatmap with a different color palette
        sns.heatmap(
            crime_area_correlation,
            cmap="YlGnBu",
            annot=True,
            fmt="d",
            cbar_kws={"label": "Count"},
        )
        plt.title("Correlation Between Top 5 Crime Types and Areas", size=15)
        plt.xlabel("Area", size=12)
        plt.ylabel("Crime Description", size=12)
        plt.show()

    def crime_area_age_group():
        """
        Explores the correlation between crime types, age groups, and geographical areas in Los Angeles.

        Returns:
        None (Displays a heatmap illustrating the multivariate correlation)
        """
        # Binning the age group
        data_frame = the_eda.eda_Exploration.binning()

        # Correlation between age group, crime description, and area
        correlation_data = (
            data_frame.groupby(["Age_Group", "Crm Cd Desc", "AREA"])
            .size()
            .unstack(fill_value=0)
        )

        # Setting the figure size and DPI for better resolution
        plt.figure(figsize=(30, 20))

        # Create a heatmap with a different color palette
        sns.heatmap(
            correlation_data,
            cmap="YlGnBu",
            annot=True,
            fmt="d",
            cbar_kws={"label": "Count"},
        )
        plt.title("Correlation Between Crime Types, Age Groups, and Areas", size=20)
        plt.xlabel("Area", size=15)
        plt.ylabel("Crime Description - Age Group", size=15)

        plt.show()

    def crime_area_age_group_some():
        """
        Explores the correlation between the top 10 crime types, age groups, and geographical areas in Los Angeles.

        Returns:
        None (Displays a heatmap illustrating the multivariate correlation)
        """

        # Binning the age group
        df = the_eda.eda_Exploration.binning()
        crime_type_counts = df["Crm Cd Desc"].value_counts().head(10)

        # Extracting the names of the crime types
        crime_type_names = crime_type_counts.index.tolist()

        # Filter data_frame for only the top 10 crime types
        top_crime_types = df[df["Crm Cd Desc"].isin(crime_type_names)]

        # Correlation between age group, crime description, and area for top crime types
        correlation_data = (
            top_crime_types.groupby(["Age_Group", "Crm Cd Desc", "AREA"])
            .size()
            .unstack(fill_value=0)
        )

        # Setting the figure size and DPI for better resolution
        plt.figure(figsize=(30, 20))

        # Create a heatmap with a different color palette
        sns.heatmap(
            correlation_data,
            cmap="YlGnBu",
            annot=True,
            fmt="d",
            cbar_kws={"label": "Count"},
        )
        plt.title(
            "Correlation Between Top 10 Crime Types, Age Groups, and Areas", size=20
        )
        plt.xlabel("Area", size=15)
        plt.ylabel("Crime Description - Age Group", size=15)

        plt.show()
