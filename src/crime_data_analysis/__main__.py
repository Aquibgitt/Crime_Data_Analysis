"""for main"""
from .the_eda import EdaExploration
from .the_inference import InferenceAnalysis

"""
Here we are calling different functions through main so that it could work with -m command. here we are calling the both things eda and inf.
"""
def main():
    print("hello professor Molin")
    eda()
    inf()

"""Here we are defining the eda part """
def eda():
    print("EDA start---------------------------------------")
    data_cleaned = EdaExploration.Cleaning()
    print(data_cleaned)
    EdaExploration.outlier_check()
    data_binning = EdaExploration.binning()
    print(data_binning)
    EdaExploration.crime_by_time()
    EdaExploration.crime_by_gender_1()
    EdaExploration.crime_by_gender_2()
    EdaExploration.crime_by_area()
    EdaExploration.crime_by_type()
    EdaExploration.crime_by_years()
    EdaExploration.crime_places()
    data_percent = EdaExploration.age_cat_gender()
    print(data_percent)
    EdaExploration.crime_by_count()
    print("EDA ends----------------------------------------")
""" Here we are calling the inference file"""

def inf():
    print("inf start---------------------------------------")
    InferenceAnalysis.crimeRateByMonth()
    InferenceAnalysis.crimeByAgeGroup()
    InferenceAnalysis.crimeByAreaType()
    InferenceAnalysis.crimeByAreaTypeCount()
    InferenceAnalysis.crimeAreaAgeGroup()
    InferenceAnalysis.crimeAreaAgeGroupSome()
    InferenceAnalysis.dataModeling()
    InferenceAnalysis.modelQuestion()
    InferenceAnalysis.performClusteringVisualization()
    print("inf ends----------------------------------------")


main()
