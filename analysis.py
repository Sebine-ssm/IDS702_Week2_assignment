import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

coffee = pd.read_csv("synthetic_coffee_health.csv")

coffee

coffee.shape

coffee.head()

coffee.sample(6)

coffee.info()

coffee.describe()

coffee.isnull().sum()

# dropped null values
coffee.dropna(inplace=True)
coffee["BMI"] = coffee["BMI"].fillna(coffee["BMI"].median())

coffee["BMI"]

coffee["Smoking"]

coffee_raw = coffee.copy()

# created a new dataframe to work with which has no duplicate rows.
coffee_cleaned = coffee.drop_duplicates(keep="first")

coffee_cleaned

coffee_cleaned.info()

coffee_cleaned.describe()

coffee_cleaned.sample(7)

coffee_cleaned.isnull().sum()

# removes out-of-bound values from the following columns
coffee_cleaned = coffee_cleaned[coffee_cleaned["Age"].between(10, 100)]
coffee_cleaned = coffee_cleaned[coffee_cleaned["Sleep_Hours"].between(0, 24)]
coffee_cleaned = coffee_cleaned[coffee_cleaned["BMI"].between(0, 40)]
coffee_cleaned = coffee_cleaned[coffee_cleaned["Heart_Rate"].between(0, 120)]

coffee_cleaned

print(coffee_cleaned[~coffee_cleaned["BMI"].between(0, 40)])
print(coffee_cleaned[~coffee_cleaned["Heart_Rate"].between(0, 120)])

coffee_cleaned["Country"].unique()

# understanding relationships between some variables.
coffee_cleaned[coffee_cleaned["Age"] > 30]

coffee_cleaned[coffee_cleaned["Sleep_Hours"] < 6]

# making a few varibles as a part of refactoring
coff_sleep = coffee_cleaned["Sleep_Hours"]
coff_heart = coffee_cleaned["Heart_Rate"]
coff_age = coffee_cleaned["Age"]
coff_BMI = coffee_cleaned["BMI"]
coff_ph = coffee_cleaned["Physical_Activity_Hours"]

coffee_cleaned[(coff_age > 25) & (coff_sleep < 7)]

coffee_cleaned[(coff_BMI < 70) | (coff_heart > 50)]

coffee_cleaned[(coff_ph > 25) & (coff_sleep < 7)]

coffee_cleaned[(coff_ph > 6) | (coff_sleep < 7)]

coffee_cleaned.groupby("Country")["Smoking"].mean()

coffee_cleaned.groupby("Country")["Smoking"].std()

coffee_cleaned["Country"].value_counts()

coffee_cleaned.groupby("Country")["Alcohol_Consumption"].mean()

coffee_cleaned.groupby("Country")["Alcohol_Consumption"].std()

coffee_cleaned.groupby("Gender")["BMI"].mean()

coffee_cleaned.groupby("Gender")["BMI"].std()

coffee_cleaned["Gender"].value_counts()

coffee_cleaned.groupby("Gender")["Alcohol_Consumption"].mean()

coffee_cleaned.groupby("Gender")["Caffeine_mg"].mean()

coffee_cleaned.groupby("Stress_Level")["Caffeine_mg"].mean()

# created another variable for refactoring and doing ML
x_cols = ["Age", "Caffeine_mg", "Coffee_Intake", "BMI", "Sleep_Hours"]

X = coffee_cleaned[x_cols]
y = coffee_cleaned[["Heart_Rate"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred
print("Mean_sq_error : ", mean_squared_error(y_test, y_pred))
print("r_sqd : ", r2_score(y_test, y_pred))

coeff = pd.DataFrame(
    {
        "Feature": X_train.columns, "Coefficient": model.coef_.flatten()
    })
print(coeff)
# created these functions to use them in the test file
def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(df):
    df = df.copy()
    df["BMI"] = df["BMI"].fillna(df["BMI"].median())
    df = df.drop_duplicates(keep="first")
    df = df[df["Age"].between(10, 100)]
    df = df[df["Sleep_Hours"].between(0, 24)]
    df = df[df["BMI"].between(0, 40)]
    df = df[df["Heart_Rate"].between(0, 120)]
    return df

def analyze_data(df):
    return {
        "average_age": df["Age"].mean(),
        "average_bmi": df["BMI"].mean(),
        "average_sleep_hours": df["Sleep_Hours"].mean(),
        "average_heart_rate": df["Heart_Rate"].mean(),
    }

# refactored the graphing codes to avoid duplication
def plot_relationships(df):
    sns.scatterplot(data=df, x="Coffee_Intake", y="BMI")
    plt.title("BMI vs Coffee_Intake")
    plt.show()

    sns.regplot(data=df, x="Coffee_Intake", y="BMI")
    plt.title("BMI vs Coffee_Intake")
    plt.show()

    sns.regplot(data=df, x="Caffeine_mg", y="BMI")
    plt.title("BMI vs Caffeine_mg")
    plt.show()

    sns.regplot(data=df, x="Coffee_Intake", y="Heart_Rate")
    plt.title("Heart_Rate vs Coffee_Intake")
    plt.show()

    sns.barplot(data=df, x="Gender", y="Coffee_Intake")
    plt.title("Coffee Intake by Gender")
    plt.show()
