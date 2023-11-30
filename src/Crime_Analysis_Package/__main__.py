"""for main"""
from .the_eda import eda_Exploration
from .the_inference import inference_Analysis


def main():
    print("hello professor Molin")
    eda()
    inf()


def eda():
    print("EDA start---------------------------------------")
    data_cleaned = eda_Exploration.Cleaning()
    print(data_cleaned)
    eda_Exploration.outlier_check()
    data_binning = eda_Exploration.binning()
    print(data_binning)
    eda_Exploration.crime_by_time()
    eda_Exploration.crime_by_gender_1()
    eda_Exploration.crime_by_gender_2()
    eda_Exploration.crime_by_area()
    eda_Exploration.crime_by_type()
    eda_Exploration.crime_by_years()
    eda_Exploration.crime_places()
    data_percent = eda_Exploration.age_cat_gender()
    print(data_percent)
    eda_Exploration.crime_by_count()
    print("EDA ends----------------------------------------")


def inf():
    print("inf start---------------------------------------")
    inference_Analysis.crime_rate_by_month()
    inference_Analysis.crime_by_age_group()
    inference_Analysis.crime_by_area_type()
    inference_Analysis.crime_by_area_type_count()
    inference_Analysis.crime_area_age_group()
    inference_Analysis.crime_area_age_group_some()
    print("inf ends----------------------------------------")


main()
