import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

coffee = pd.read_csv('synthetic_coffee_health.csv')

coffee 

coffee.shape

coffee.head()

coffee.sample(6)

coffee.info()

coffee.describe()

coffee.isnull().sum()

coffee.dropna(inplace=True)

coffee["BMI"] = coffee["BMI"].fillna(coffee["BMI"].median())

coffee["BMI"]

coffee["Smoking"]

coffee_raw = coffee.copy() 

coffee_cleaned = coffee.drop_duplicates(keep='first')

coffee_cleaned

coffee_cleaned.info()

coffee_cleaned.describe()

coffee_cleaned.sample(7)

coffee_cleaned.isnull().sum()

coffee_cleaned = coffee_cleaned[coffee_cleaned['Age'].between(10,100)]
coffee_cleaned = coffee_cleaned[coffee_cleaned['Sleep_Hours'].between(0,24)]
coffee_cleaned = coffee_cleaned[coffee_cleaned['BMI'].between(0, 40)]
coffee_cleaned = coffee_cleaned[coffee_cleaned['Heart_Rate'].between(0, 120)]

coffee_cleaned 

print(coffee_cleaned[~coffee_cleaned['BMI'].between(0, 40)])
print(coffee_cleaned[~coffee_cleaned['Heart_Rate'].between(0, 120)])

coffee_cleaned['Country'].unique()

coffee_cleaned[coffee_cleaned['Age'] > 30]

coffee_cleaned[coffee_cleaned['Sleep_Hours'] < 6]

coffee_cleaned[(coffee_cleaned['Age'] > 25) & (coffee_cleaned['Sleep_Hours'] < 7)]

coffee_cleaned[(coffee_cleaned['BMI'] < 70) | (coffee_cleaned['Heart_Rate'] > 50)]

coffee_cleaned[(coffee_cleaned['Physical_Activity_Hours'] > 25) & (coffee_cleaned['Sleep_Hours'] < 7)]

coffee_cleaned[(coffee_cleaned['Physical_Activity_Hours'] > 6) | (coffee_cleaned['Sleep_Hours'] < 7)]

coffee_cleaned.groupby('Country')['Smoking'].mean()

coffee_cleaned.groupby('Country')['Smoking'].std()

coffee_cleaned['Country'].value_counts()

coffee_cleaned.groupby('Country')['Alcohol_Consumption'].mean()

coffee_cleaned.groupby('Country')['Alcohol_Consumption'].std()

coffee_cleaned.groupby('Gender')['BMI'].mean()

coffee_cleaned.groupby('Gender')['BMI'].std()

coffee_cleaned['Gender'].value_counts()

coffee_cleaned.groupby('Gender')['Alcohol_Consumption'].mean()

coffee_cleaned.groupby('Gender')['Caffeine_mg'].mean()

coffee_cleaned.groupby('Stress_Level')['Caffeine_mg'].mean()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = coffee_cleaned[['Age', 'Caffeine_mg', 'Coffee_Intake', 'BMI', 'Sleep_Hours']]
y = coffee_cleaned[['Heart_Rate']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred

print('Mean_sq_error : ', mean_squared_error(y_test, y_pred))
print('r_sqd : ', r2_score(y_test, y_pred))

coeff = pd.DataFrame({
    'Feature' : X_train.columns,
    'Coefficient' : model.coef_.flatten()
})

print(coeff)

sns.scatterplot(data=coffee_cleaned, x='Coffee_Intake',y='BMI')
plt.title('BMI vs Coffee_Intake')
plt.show()

sns.regplot(data=coffee_cleaned, x='Coffee_Intake',y='BMI')
plt.title('BMI vs Coffee_Intake')
plt.show()

sns.regplot(data=coffee_cleaned, x='Caffeine_mg',y='BMI')
plt.title('BMI vs Caffeine_mg')
plt.show()

sns.regplot(data=coffee_cleaned, x='Coffee_Intake',y='Heart_Rate')
plt.title('Heart_Rate vs Coffee_Intake')
plt.show()

sns.barplot(data=coffee_cleaned, x='Gender',y='Coffee_Intake')
plt.title('Coffee Intake by Gender')
plt.show()
