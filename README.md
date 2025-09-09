# IDS706_Week2_assignment
Global Coffee Health Dataset

## Project Overview
This project analyzes the relationship between Coffee Consumption, Sleeping hours, Age, and BMI with Heart rate. The goal of this project is to see if there is there is a positive relationship between these two variables or no. 

## Project Structure

- `analysis.py`: Main analysis script for data processing and visualization.
- `synthetic_coffee_health.csv`: Dataset used for analysis.
- `requirements.txt`: Python dependencies.
- `Makefile`: Common commands for setup and running analysis.
- `.gitignore`: Files and folders excluded from version control.

## Makefile Setup

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	flake8 analysis.py

clean:
	rm -rf __pycache__ .pytest_cache .coverage

all:
	install format lint 

## requirement.txt

pylint
flake8
pytest
click
black
pytest-cov
pandas
numpy
plotly.express
matplotlib
polars
scikit-learn
seaborn

## Dataset
Dataset contains information about Age, Gender, Country, Coffee Intake (in cups), Sleep Hours, Caffeiene consumption in mg, Hours of Physical Activity, Sleep Quality, BMI, Heart Rate, Stress Level, Health Issues, Occupation, Smoking, and Alcohol Consumption varaibles. It has 4057 rows and 16 columns.

<class 'pandas.core.frame.DataFrame'>
Index: 4057 entries, 2 to 9997
Data columns (total 16 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   ID                       4057 non-null   int64  
 1   Age                      4057 non-null   int64  
 2   Gender                   4057 non-null   object 
 3   Country                  4057 non-null   object 
 4   Coffee_Intake            4057 non-null   float64
 5   Caffeine_mg              4057 non-null   float64
 6   Sleep_Hours              4057 non-null   float64
 7   Sleep_Quality            4057 non-null   object 
 8   BMI                      4057 non-null   float64
 9   Heart_Rate               4057 non-null   int64  
 10  Stress_Level             4057 non-null   object 
 11  Physical_Activity_Hours  4057 non-null   float64
 12  Health_Issues            4057 non-null   object 
 13  Occupation               4057 non-null   object 
 14  Smoking                  4057 non-null   int64  
 15  Alcohol_Consumption      4057 non-null   int64  
dtypes: float64(5), int64(5), object(6)
memory usage: 538.8+ KB

## Data Exploration
There were some duplicates and null values which I cleaned. Some insights which I found:

1. More than half of the people in the dataset are above 30 years of age.

2. There are no people who fall under the following categories: 'Having a BMI of less than 25 and heart rate above 50', 'Having a BMI of greater than 25 and heart rate below 50', 'Having a BMI of greater than 40 and heart rate below 50', 'Having a BMI of less than 15 or heart rate below 50', 'Doing less than 25 hours of physical activity and less than 7 hours of sleep', and and 'Having a BMI of less than 70 or heart rate above 50'.

3. Some people of South Korea who answered in this survey (213 people) have a higher percentage of drinking coffee than the Chinese people who responded (233 of them)

4. In the 'Gender' column, there are more women than men and other categories. This could be why the mean values of BMI and alcohol consumption of women are higher than men, but not by a lot (and these numbers are similar). 

5. People who consume high levels of caffeine have higher levels of stress than other people. 

## Machine Learning
Imported scikit-learn and three modules which are LinearRegression, r2_score, train_test_spilt, and mean_squared_error. 

The variables which I used for my x-values are 'Age', 'Caffeine_mg', 'Coffee_Intake', 'BMI', and 'Sleep_Hours', with my y variable being 'Heart_Rate'.
 
Performed Multiple Linear Regression and found out some interesting insights. There is a very high mean_sq_error with a very low r2_score. This shows that the model that I have used shows a very poor correlation between these variables. The coefficients are too small. 

## Plots

Created a scatterplot, bar graph and a few regression plots to visualize the results.

From the scatterplot and regression plots, we can see that the relationship between the variables ploted in those graphs have poor correlation. Created a bar graph to just see the proportion of people (categorized by gender) to see how they consume coffee. 

From my analysis I conclude that there is a poor correlation between Age, Caffeine_mg, Coffee_Intake, BMI, and Sleep_Hours, with Heart_Rate.











