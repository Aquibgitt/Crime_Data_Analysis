# The_Inference_Class
"""
Importing the required libraries. 
"""
import pandas as pd
import numpy as np
# Importing the Visualization libraries
import matplotlib.pyplot as plt  
import seaborn as sns
pd.set_option("display.max_columns", None)
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



from src.crime_data_analysis import the_eda



# Reading the values and storing it in a dataframe using pandas
df = pd.read_csv(
    "https://media.githubusercontent.com/media/Aquibgitt/Crime_Data_Analysis/main/Crime_Data_from_2020_to_Present.csv"
)


class InferenceAnalysis:
    """
    This class contains the functions to analyse and get the conclusions of the research question.
    """

    def __init__(self):
        pass

    def crimeRateByMonth():
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

    def crimeByAreaTypeCount():
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


    # def crime_by_area_type_count_Test():
    #     """
    #     Examines the areas in Los Angeles with the highest incidence of specific crimes.

    #     Parameters:
    #     df (DataFrame): Crime dataset containing 'AREA' and 'Crm Cd Desc' columns.

    #     Returns:
    #     None (Displays crime counts by area and crime type using Matplotlib)
    #     """
    #     crime_area_counts = (
    #         df.groupby(["AREA", "Crm Cd Desc"]).size().unstack().fillna(0)
    #     )

    #     # Set up the figure and axis using seaborn's heatmap
    #     plt.figure(figsize=(20, 8))
    #     sns.heatmap(crime_area_counts, annot=True, fmt="g", cmap="YlGnBu")

    #     plt.xlabel("Crime Type")
    #     plt.ylabel("Area")
    #     plt.title("Crime Counts by Area and Crime Type")

    #     plt.tight_layout()
    #     plt.show()

    def crimeByAgeGroup():
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

    def crimeByAreaType():
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

    def crimeAreaAgeGroup():
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

    def crimeAreaAgeGroupSome():
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

    def dataModeling():
        """
        Perform KMeans clustering on the dataset and evaluate the model using Pipeline and Cross-validation Techniques.

        Args:
        df (DataFrame): Input DataFrame containing necessary columns, including 'Crm Cd Desc' and 'Vict Age'.

        Returns:
        float, float: Mean squared error for the train and test sets.
    """
        # Custom transformer for encoding 'Crime_Key'
        class EncodeCrimeKey(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self
            
            def transform(self, X):
                X['Crime_Key'] = X['Crm Cd Desc'].str.split().str[0]
                label_encoder = LabelEncoder()
                X['Crime_Key_Encoded'] = label_encoder.fit_transform(X['Crime_Key'])
                return X[['Crime_Key_Encoded', 'Vict Age']]  # Keeping 'Vict Age'

        # Load your dataset into 'df'

        # Define the pipeline for clustering
        pipeline = Pipeline([
            ('encoding', EncodeCrimeKey()),
            ('kmeans', KMeans(n_clusters=5, random_state=42))
        ])

        # Get the features and target for clustering
        X = df  # Using the whole dataframe
        y = df['Vict Age']  # Target

        # Cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_mse = -1 * cv_scores.mean()  # Taking the negative mean as cross_val_score gives negative MSE

        print(f"Cross-Validation Mean Squared Error: {cv_mse}")

        # Splitting the data into train and test sets
        X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

        # Fit the pipeline on the training data
        pipeline.fit(X_train)

        # Get the cluster labels for train and test sets
        train_cluster_labels = pipeline.predict(X_train)
        test_cluster_labels = pipeline.predict(X_test)

        # Calculate mean squared error for train and test
        mse_train = mean_squared_error(X_train['Vict Age'], train_cluster_labels)
        mse_test = mean_squared_error(X_test['Vict Age'], test_cluster_labels)

        print(f"Mean Squared Error for Train: {mse_train}")
        print(f"Mean Squared Error for Test: {mse_test}")

        # Plotting the clusters
        plt.scatter(X_test['Crime_Key_Encoded'], X_test['Vict Age'], c=test_cluster_labels, cmap='viridis')
        plt.title('KMeans Clustering of Victim Age and Crime Key (Test Set)')
        plt.xlabel('Crime_Key_Encoded')
        plt.ylabel('Victim Age')
        plt.colorbar(label='Cluster')
        plt.show()




    def modelQuestion():

        """
        Perform KMeans clustering on the dataset using Latitude, Longitude, and Victim Age and visualize in 3D.
        Args:
        df (DataFrame): Input DataFrame containing 'LAT', 'LON', and 'Vict Age' columns.

        Returns:
        None
    """
        # Selecting features (X) for clustering
        X = df[['LAT', 'LON', 'Vict Age']]

        # Applying KMeans clustering
        kmeans = KMeans(n_clusters=5, random_state=42)  # You can set the number of clusters as needed
        kmeans.fit(X)

        # Getting the cluster labels
        cluster_labels = kmeans.labels_

        # Adding the cluster labels to the dataframe
        df['Cluster_Labels'] = cluster_labels

        # Plotting the clusters using LAT, LON, and Vict Age
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df['LAT'], df['LON'], df['Vict Age'], c=df['Cluster_Labels'], cmap='viridis')
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Longitude')
        ax.set_zlabel('Victim Age')
        plt.title('KMeans Clustering')
        plt.show()



    # def Elbow():
    #     """
    #     Perform KMeans clustering on the dataset and evaluate the model using Pipeline and Cross-validation Techniques.

    #     Args:
    #     df (DataFrame): Input DataFrame containing necessary columns, including 'Crm Cd Desc' and 'Vict Age'.

    #     Returns:
    #     float, float: Mean squared error for the train and test sets.
    # """
    # inertia_values = []
    # for i in range(1, 11):
    #         kmeans = KMeans(n_clusters=i, random_state=42)
    #         data = df[['AREA', 'Crm Cd Desc']]
    #         encoded_df = pd.get_dummies(data, drop_first=True)
    #         kmeans.fit(encoded_df)
    #         inertia_values.append(kmeans.inertia_)

    #     # Plotting the elbow curve for up to 5 clusters
    # plt.figure(figsize=(8, 5))
    # plt.plot(range(1, 6), inertia_values, marker='o', linestyle='--')  # Adjusted range for 5 clusters
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Inertia')
    # plt.title('Elbow Method for 5 Clusters')
    # plt.show()
    

    


    def performClusteringVisualization():
        """
        Perform KMeans clustering on the 'AREA' and 'Crm Cd Desc' columns and visualize the clusters.

        Args:
        df (DataFrame): Input DataFrame containing 'AREA' and 'Crm Cd Desc' columns.

        Returns:
        None
        """
        # Custom transformer for one-hot encoding 'Crime_Key'
        # Here are the two columns where we can perform the clustering on.
        data = df[['AREA', 'Crm Cd Desc']]

        # Perform one-hot encoding for categorical columns
        encoded_df = pd.get_dummies(data, drop_first=True)

        # Initialize and fit KMeans clustering model
        kmeans = KMeans(n_clusters=5, random_state=42)  # You can adjust the number of clusters as needed
        kmeans.fit(encoded_df)

        # Adding the cluster labels to the original dataframe
        data['Cluster'] = kmeans.labels_
        plt.figure(figsize=(30, 20))
        for cluster in data['Cluster'].unique():
            cluster_data = data[data['Cluster'] == cluster]
            plt.scatter(cluster_data['AREA'], cluster_data['Crm Cd Desc'], label=f'Cluster {cluster}')

        plt.xlabel('AREA', fontsize=12)  # Adjusting the font size for X-axis label
        plt.ylabel('Crm Cd Desc', fontsize=12)  # Adjusting the font size for y-axis label
        plt.title('Clustering of Crime Areas and Types', fontsize=14)  # Adjust the font size for the title
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)  # Change legend position and font size
        plt.tight_layout()  # Increase the spacing between axis labels and the plot area
        plt.show()

