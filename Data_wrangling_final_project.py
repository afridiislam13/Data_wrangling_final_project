# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Import the dataset using Pandas from the provided URL
url = "https://raw.githubusercontent.com/SR1608/Datasets/main/covid-data.csv"
df = pd.read_csv(url)

# 2. High Level Data Understanding
# a. Find no. of rows & columns in the dataset
num_rows, num_columns = df.shape

# b. Data types of columns
data_types = df.dtypes

# c. Info & describe of data in dataframe
data_info = df.info()
data_description = df.describe()

# 3. Low Level Data Understanding
# a. Find count of unique values in the location column
unique_location_count = df['location'].nunique()

# b. Find which continent has maximum frequency using value counts
max_continent_frequency = df['continent'].value_counts().idxmax()

# c. Find maximum & mean value in 'total_cases'
max_total_cases = df['total_cases'].max()
mean_total_cases = df['total_cases'].mean()

# d. Find quartile values in 'total_deaths'
quartiles_total_deaths = df['total_deaths'].quantile([0.25, 0.5, 0.75])

# e. Find which continent has maximum 'human_development_index'
max_hdi_continent = df.loc[df['human_development_index'].idxmax()]['continent']

# f. Find which continent has minimum 'gdp_per_capita'
min_gdp_continent = df.loc[df['gdp_per_capita'].idxmin()]['continent']

# 4. Filter the dataframe with only specified columns
selected_columns = ['continent', 'location', 'date', 'total_cases', 'total_deaths', 'gdp_per_capita', 'human_development_index']
df = df[selected_columns]

# 5. Data Cleaning
# a. Remove all duplicate observations
df = df.drop_duplicates()

# b. Find missing values in all columns
missing_values = df.isnull().sum()

# c. Remove all observations where the continent column value is missing
df = df.dropna(subset=['continent'])

# d. Fill all missing values with 0
df = df.fillna(0)

# 6. Date time format
# a. Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# b. Create a new column 'month' after extracting month data from the date column
df['month'] = df['date'].dt.month

# 7. Data Aggregation
# a. Find max value in all columns using groupby function on 'continent' column
df_groupby = df.groupby('continent').max().reset_index()

# 8. Feature Engineering
# a. Create a new feature 'total_deaths_to_total_cases' by the ratio of 'total_deaths' column to 'total_cases'
df_groupby['total_deaths_to_total_cases'] = df_groupby['total_deaths'] / df_groupby['total_cases']

# 9. Data Visualization
# a. Univariate analysis on 'gdp_per_capita' column by plotting histogram
sns.distplot(df_groupby['gdp_per_capita'])
plt.title('Histogram of GDP per Capita')
plt.show()

# b. Scatter plot of 'total_cases' & 'gdp_per_capita'
sns.scatterplot(x='gdp_per_capita', y='total_cases', data=df_groupby)
plt.title('Scatter Plot of Total Cases vs GDP per Capita')
plt.show()

# c. Pairplot on df_groupby dataset
sns.pairplot(df_groupby)
plt.show()

# d. Bar plot of 'continent' column with 'total_cases'
sns.catplot(x='continent', y='total_cases', kind='bar', data=df_groupby)
plt.title('Total Cases by Continent')
plt.show()

# 10. Save the df_groupby dataframe to a CSV file
df_groupby.to_csv('df_groupby.csv', index=False)
