
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure the 'visuals' folder exists
os.makedirs('visuals', exist_ok=True)

# Load the dataset
df = pd.read_csv('dataset/titanic.csv')

# Fill missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Gender Distribution
sns.countplot(x='Sex', data=df)
plt.title('Gender Distribution')
plt.savefig('visuals/gender_distribution.png')
plt.clf()

# Survival Count
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.savefig('visuals/survival_count.png')
plt.clf()

# Age Distribution
sns.histplot(df['Age'], bins=10, kde=True)
plt.title('Age Distribution')
plt.savefig('visuals/age_distribution.png')
plt.clf()

# Fare vs Survival
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title('Fare vs Survival')
plt.savefig('visuals/fare_vs_survival.png')
plt.clf()
