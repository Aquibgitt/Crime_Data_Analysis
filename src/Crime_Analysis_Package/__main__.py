'''for main'''
from .the_eda import eda_Exploration
from .the_inference import inference_Analysis

def main():
    print("hello professor Molin")
    eda()
    inf()


def eda():
    print("EDA start---------------------------------------")
    EDA = eda_Exploration()
    data_cleaned=EDA.Cleaning()
    print("hello aquib")
    print(data_cleaned)
    EDA.outlier_check()
    data_binning=EDA.binning()
    print(data_binning)
    EDA.crime_by_time()
    EDA.crime_by_gender_1()
    EDA.crime_by_gender_2()
    EDA.crime_by_area()
    EDA.crime_by_type()
    EDA.crime_by_years()
    EDA.crime_places()
    data_percent=EDA.age_cat_gender()
    print(data_percent)
    EDA.crime_by_count()
    print("EDA ends----------------------------------------")


def inf():
    print("inf start---------------------------------------")
    inference = inference_Analysis
    inference.crime_rate_by_month()
    inference.crime_by_age_group()
    inference.crime_by_area_type()
    inference.crime_by_area_type_count()
    inference.crime_area_age_group()
    inference.crime_area_age_group_top5()
    print("inf ends----------------------------------------")











main()