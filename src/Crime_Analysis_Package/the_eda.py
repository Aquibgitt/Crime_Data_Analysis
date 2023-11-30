##Project-1 EDA FILE
## First and foremost thing to do is Importing the PYTHON libraries.


import pandas as pd

# Importing the Visualization libraries
import matplotlib.pyplot as plt  # visualizing data
import seaborn as sns


pd.set_option("display.max_columns", None)
import warnings

warnings.filterwarnings("ignore")
# Reading the values and storing it in a dataframe using pandas
df = pd.read_csv(
    "https://media.githubusercontent.com/media/Aquibgitt/Crime_Data_Analysis/main/Crime_Data_from_2020_to_Present.csv"
)


class eda_Exploration:
    """
    This class encompasses various functions dedicated to conducting exploratory data analysis,
    facilitating comparisons among different attributes, and presenting the findings through diverse visualizations created with the Matplotlib and Seaborn libraries.
    The ultimate goal is to draw insightful conclusions derived from the visual representations.
    """

    def __init__(self):
        pass

    def Cleaning():
        """
        Cleans the dataset by dropping unrelated/blank columns and removing rows with missing values.

        Returns:
        DataFrame: Cleaned dataset.
        """
        # drop unrelated/blank columns, and using inplace to commit and avoid the extra memory occupancy
        df.drop(
            [
                "Mocodes",
                "Crm Cd 2",
                "Crm Cd 3",
                "Crm Cd 4",
                "Cross Street",
            ],
            axis=1,
            inplace=True,
        )
        # To remove missing values from the rows
        df.dropna(axis=0)
        return df

    def outlier_check():
        """
        Checks for outliers in 'Vict Age' and 'TIME OCC' columns and visualizes them using box plots.

        Returns:
        None
        """
        # Box Plot for determining victim's age
        plt.subplot(1, 2, 1)
        sns.boxplot(df["Vict Age"])
        plt.xlabel("Vict Age")

        # Box plot for determining Time
        plt.subplot(1, 2, 2)
        sns.boxplot(df["TIME OCC"])
        plt.xlabel("Time OCC")
        # This is to show the visualization.
        plt.show()
        ##df is the Dataframe in which we read all the data.

    def binning():
        """
        Divides 'TIME OCC' into time zones and 'Vict Age' into age groups based on specified bin edges and labels.

        Returns:
        DataFrame: Updated dataset with 'Time Zone' and 'Age_Group' columns.
        """
        bin_edges = [0, 400, 500, 1200, 1800, 2400]
        bin_labels = ["Early Morning", "Morning", "Afternoon", "Evening", "Night"]

        # Create a new column by binning the 'TIME OCC' column
        df["Time Zone"] = pd.cut(df["TIME OCC"], bins=bin_edges, labels=bin_labels)

        # Now Binning the Age column to find out the Age group of Victims.
        bin_edges = [0, 12, 25, 40, 60, 100]
        bin_labels = ["Children", "Teen", "Adult", "Middle-Adult", "Old"]

        # Create a new column by binning the 'TIME OCC' column
        df["Age_Group"] = pd.cut(df["Vict Age"], bins=bin_edges, labels=bin_labels)
        return df

    def crime_by_time():
        """
        Visualizes the count of reported crimes by time zone using Matplotlib and Seaborn.

        Returns:
        None
        """
        # Define the order of time zones for plotting (I named 4 categories in a Day as Time Zone.)
        time_zone_order = ["Early Morning", "Morning", "Afternoon", "Evening", "Night"]

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Subplot 1: Matplotlib Bar Plot
        time_zone_counts = df["Time Zone"].value_counts().loc[time_zone_order]
        axes[0].bar(time_zone_counts.index, time_zone_counts.values, color="skyblue")
        axes[0].set_xlabel("Time Zone")
        axes[0].set_ylabel("Count of Reported Crimes")
        axes[0].set_title("Matplotlib - Reported Crimes by Time Zone")
        axes[0].tick_params(axis="x", rotation=0)

        # Subplot 2: Seaborn Count Plot
        sns.countplot(
            data=df, x="Time Zone", order=time_zone_order, palette="pastel", ax=axes[1]
        )
        axes[1].set_xlabel("Time Zone")
        axes[1].set_ylabel("Count of Reported Crimes")
        axes[1].set_title("Seaborn - Reported Crimes by Time Zone")

        plt.tight_layout()  # Adjusts spacing between subplots
        plt.show()

    def age_cat_gender():
        """
        Visualizes the count of reported crimes by time zone using Matplotlib and Seaborn.we are using the value counts init so that we can get the percentage of the counts.

        Returns:
        None
        """
        df["Age_cat_gender"] = df["Age_Group"].astype(str) + "-" + df["Vict Sex"]

        print(df["Age_cat_gender"].value_counts(normalize=True))

    def crime_by_gender_1():
        """
        Plots Male and female populations by age group using Matplotlib,here we are using the Binned column age group and victim's gender and match them accordingly.

        Args:
        df (DataFrame): DataFrame containing data on crime incidents.

        Returns:
        None
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        grouped_data = df.groupby(["Age_Group", "Vict Sex"]).size().unstack()

        # Plotting male populations in the first subplot using Matplotlib
        axes[0].bar(grouped_data.index, grouped_data["M"], color="blue", label="Male")
        axes[0].set_xlabel("Age_Group")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Male Populations by Age Group (Matplotlib)")
        axes[0].legend()

        # Plotting female populations in the second subplot using Matplotlib
        axes[1].bar(grouped_data.index, grouped_data["F"], color="pink", label="Female")
        axes[1].set_xlabel("Age_Group")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Female Populations by Age Group (Matplotlib)")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def crime_by_gender_2():
        """
        Plots male and female populations by age group using Seaborn.here we are using the groupby statement to proceed furthur.

        Args:
        df (DataFrame): DataFrame containing data on crime incidents.

        Returns:
        None
        """
        # Group data by 'Age_Group' and 'Vict Sex', then count occurrences
        grouped_data = (
            df.groupby(["Age_Group", "Vict Sex"]).size().reset_index(name="Count")
        )

        plt.figure(figsize=(15, 6))
        # Plotting using Seaborn barplot
        sns.barplot(
            data=grouped_data,
            x="Age_Group",
            y="Count",
            hue="Vict Sex",
            palette=["blue", "pink", "green", "red", "orange"],
        )
        plt.xlabel("Age Group")
        plt.ylabel("Count")
        plt.title("Population by Age Group and Gender")
        plt.legend(title="Gender")
        plt.show()

    def crime_by_area():
        """
        Analyzes reported crimes by area, identifies the most notorious area, and visualizes the findings using Matplotlib and Seaborn.

        Returns:
        None
        """
        # Create a DataFrame for the counts of reported crimes by area
        area_crime_counts = df["AREA"].value_counts().reset_index()
        area_crime_counts.columns = ["AREA", "Count"]

        # Find the area with the highest count
        most_notorious_area = area_crime_counts.loc[area_crime_counts["Count"].idxmax()]

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot using Matplotlib
        axes[0].bar(
            area_crime_counts["AREA"], area_crime_counts["Count"], color="skyblue"
        )
        axes[0].set_xlabel("Area")
        axes[0].set_ylabel("Count of Reported Crimes")
        axes[0].set_title("Reported Crimes by Area (Matplotlib)")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].text(
            most_notorious_area["AREA"],
            most_notorious_area["Count"],
            f'Most Notorious Area: {most_notorious_area["AREA"]}',
            ha="center",
            va="bottom",
            color="red",
        )

        # Plot using Seaborn
        sns.barplot(data=area_crime_counts, x="AREA", y="Count", ax=axes[1])
        axes[1].set_xlabel("Area")
        axes[1].set_ylabel("Count of Reported Crimes")
        axes[1].set_title("Reported Crimes by Area (Seaborn)")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].text(
            most_notorious_area["AREA"],
            most_notorious_area["Count"],
            f'Most Notorious Area: {most_notorious_area["AREA"]}',
            ha="center",
            va="bottom",
            color="red",
        )

        # Show the combined plot
        plt.tight_layout()
        plt.show()

        # Display the most notorious area
        print(
            f"The most notorious area for crimes is {most_notorious_area} (Area {most_notorious_area['AREA']}) with {most_notorious_area['Count']} reported crimes."
        )

    def crime_by_years():
        """
        Displays the count of reported times by year and month using a Seaborn histogram.

        Returns:
        None
        """
        df["DATE OCC"] = pd.to_datetime(df["DATE OCC"])
        # Extract year and month
        df["Year"] = df["DATE OCC"].dt.year
        df["Month"] = df["DATE OCC"].dt.month
        # Create a histogram using Seaborn
        plt.figure(figsize=(15, 6))
        sns.histplot(
            data=df,
            x="Year",
            hue="Month",
            discrete=(True, False),
            multiple="stack",
            palette="flare",
        )
        plt.title("Reported Times by Year and Month")
        plt.xlabel("Year")
        plt.ylabel("Count of Reported Times")
        plt.show()

    def crime_by_count():
        """
        Identifies and visualizes the top 5 most frequent crimes by description using Matplotlib and Seaborn.

        Returns:
        None
        """
        top_crimes_desc = (
            df.groupby(["Crm Cd Desc"])
            .size()
            .reset_index(name="Count")
            .sort_values(by="Count", ascending=False)
            .head(5)
        )

        # Top 5 Most Frequent Crimes by Description
        plt.figure(figsize=(15, 6))

        # Matplotlib bar plot
        plt.subplot(1, 2, 1)
        plt.bar(
            top_crimes_desc["Crm Cd Desc"],
            top_crimes_desc["Count"],
            color="lightseagreen",
        )
        plt.title("Top 5 Most Frequent Crimes (Description)")
        plt.xlabel("Crime Description")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")

        # Seaborn bar plot
        plt.subplot(1, 2, 2)
        sns.barplot(data=top_crimes_desc, x="Crm Cd Desc", y="Count", palette="Set1")
        plt.title("Top 5 Most Frequent Crimes (Description)")
        plt.xlabel("Crime Description")
        plt.ylabel("Count")
        plt.xticks(rotation=45)

        plt.tight_layout()  # Ensures the plots don't overlap

        plt.show()

    def crime_by_type():
        """
        Analyzes crime types and provides visualizations.

        Args:
        df (DataFrame): Input DataFrame containing crime data.

        Returns:
        None
        """
        # Grouping data by 'Crm Cd Desc'
        crime_type_counts = df["Crm Cd Desc"].value_counts().head(10)

        # Creating a bar plot for top crime types
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x=crime_type_counts.values, y=crime_type_counts.index, palette="viridis"
        )
        plt.xlabel("Count of Crimes")
        plt.ylabel("Crime Type")
        plt.title("Top 10 Crime Types")
        plt.show()

        # Statistical summary for crime types
        print("\nStatistical Summary for Crime Types:")
        print(crime_type_counts)

    def crime_rate_over_time():
        """
        Calculates and visualizes crime rate changes over time.

        Args:
        df (DataFrame): Input DataFrame containing crime data.

        Returns:
        None
        """
        # Extracting year from the 'DATE OCC' column
        df["Year"] = df["DATE OCC"].dt.year

        # Calculating crime counts per year
        crime_counts_per_year = df["Year"].value_counts().sort_index()

        # Calculating crime rate per year
        total_records = len(df)
        crime_rate_per_year = (crime_counts_per_year / total_records) * 100

        # Plotting crime rate changes over time
        plt.figure(figsize=(10, 6))
        crime_rate_per_year.plot(kind="line", marker="o", color="green")
        plt.title("Crime Rate Changes Over Time")
        plt.xlabel("Year")
        plt.ylabel("Crime Rate (%)")
        plt.grid(True)
        plt.show()

        # Statistical summary for crime rate per year
        print("\nStatistical Summary for Crime Rate Changes Over Time:")
        print(crime_rate_per_year)

    def crime_places():
        """
        Calculates and visualizes crime rate changes over time.

        Args:
        df (DataFrame): Input DataFrame containing crime data.

        Returns:
        None
        """
        # Crime Location Analysis

        crime_location_counts = df["Premis Desc"].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        crime_location_counts.plot(kind="bar")
        plt.xlabel("Crime Location")
        plt.ylabel("Number of Crimes")
        plt.title("Crime Location Analysis")
        plt.show()

    def Result_1():
        """
        Generates a countplot visualizing the distribution of reported crimes based on their description across different time zones, utilizing Seaborn.

        Returns:
        None

        Description:
        This function creates a Seaborn countplot to illustrate the distribution of reported crimes categorized by their descriptions and time zones. The x-axis represents the time zones, categorized as 'Early Morning', 'Morning', 'Afternoon', 'Evening', and 'Night'. The count of reported crimes is encoded using color variations, distinguishing various crime descriptions. The function provides a comprehensive visual overview, enabling the comparison of crime distributions across different time segments, aiding in the identification of temporal crime trends.
        """
        time_zone_order = ["Early Morning", "Morning", "Afternoon", "Evening", "Night"]
        # data_frame = The_EDA.eda_Exploration.binning()
        # Setting up the figure size
        plt.figure(figsize=(30, 15))
        # Using the Counterplot with set2 pallette and using hue for colour encoding.
        sns.countplot(
            data=df,
            x="Time Zone",
            hue="Crm Cd Desc",
            order=time_zone_order,
            palette="Set2",
            dodge=True,
        )
        plt.xlabel("Time Zone", size=30)
        plt.ylabel("Count of Reported Crimes", size=30)
        plt.title("Reported Crimes by Time Zone and Crime Description", size=40)
        # with the custom rotation we can tilt out titles to any angle
        plt.xticks(rotation=0)
        # By giving the loc on upper left we are managing not to make the fig soo Conjested.
        plt.legend(
            title="Crime Description",
            title_fontsize="30",
            loc="upper left",
            bbox_to_anchor=(1.15, 1),
        )
        plt.show()
