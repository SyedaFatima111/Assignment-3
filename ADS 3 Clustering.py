import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Specify the file path
file_path = 'World_bank.csv'

# Load the CSV file into a dataframe
World_Bank_data = pd.read_csv(World_bank, skiprows=4)

# Display the dataframe
World_Bank_data

# Display the first five rows using iloc
World_Bank_data.iloc[:5]

# Display the columns of the dataframe
print(World_Bank_data.columns)

# Total missing values
# how many total missing values do we have?
total_cells = np.product(World_Bank_data.shape)
print('Total_cells :',total_cells)
total_missing = World_Bank_data.isnull().sum().sum()
print('Total_missing :',total_missing)
# percent of data that is missing
print('Missing value percentage rate :',((total_missing/total_cells) * 100))

# Detect Missing Data Values
World_Bank_data.isnull().sum()

# Total Missing Values
World_Bank_data.isnull().sum().sum()

# Impute missing data values with the mean of the respective columns
World_Bank_data = World_Bank_data.apply(lambda x: x.fillna(x.mean()) if x.dtype != 'object' else x)

# Data overview concisely presented
World_Bank_data.info()

col_year = ['1980', '1981', '1982', '1983', '1984', '1985', '1986',
       '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995',
       '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
       '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
       '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']

#Identify relevant variables for analysis succinctly.
variables = ['GDP (current US$)']

#Subdivide data into smaller sets as required.
World_Bank_data = World_Bank_data[(World_Bank_data['Indicator Name'].isin(variables))]
World_Bank_data

World_Bank_data.columns

from sklearn.preprocessing import StandardScaler

# Specify the file path
file_path = 'World_bank.csv'

# Load the CSV file into a dataframe
World_Bank_data = pd.read_csv(file_path, skiprows=4)

# Define a list of columns to select
cols_to_select = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code',
                  '1991', '1996', '2001', '2006', '2011', '2016', '2021']

# Select the desired columns and create a copy of the resulting dataframe
df = World_Bank_data[cols_to_select].copy()

# Set the index of the dataframe to the 'Country Name' column
df.set_index('Country Name', inplace=True)

# Define a list of columns to normalize
cols_to_normalize = ['1991', '1996', '2001', '2006', '2011', '2016', '2021']

# Create a StandardScaler object to standardize the data
scaler = StandardScaler()

# Normalize the selected columns
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

# Display the resulting dataframe
print(df)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



# Convert each column to a numeric type
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Check for NaN, infinity, and large values
if df.isnull().values.any() or np.isinf(df.values).any() or df.abs().max().max() > 1e9:
    # Replace NaN and infinity values with 0
    df = df.replace([np.inf, -np.inf, np.nan], 0)
    # Check for values that are too large
    if df.abs().max().max() > 1e9:
        # Scale down the values
        df = df / 1e9

# Define a list of columns to normalize
cols_to_normalize = ['1991', '1996', '2001', '2006', '2011', '2016', '2021']

# Create a StandardScaler object to standardize the data
scaler = StandardScaler()

# Normalize the selected columns
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

# Apply KMeans clustering with 3 clusters
kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(df[cols_to_normalize])

# Output clusters' initial values
for i in range(kmeans.n_clusters):
    print(f'Cluster {i}:')
    print(df[df['Cluster'] == i][cols_to_select])

# Calculate the Silhouette score
score = silhouette_score(df[cols_to_normalize], df['Cluster'])
print(f"Silhouette score: {score}")

# Visualize clusters and centers using plots professionally.
fig, ax = plt.subplots(figsize=(6, 6))
scatter = ax.scatter(df[cols_to_normalize[1]], df[cols_to_normalize[-2]], c=df['Cluster'])
centers = scaler.inverse_transform(kmeans.cluster_centers_)
ax.scatter(centers[:, 0], centers[:, -1], marker='*', s=100, linewidths=3, color='g')
ax.set_xlabel('1996')
ax.set_ylabel('2016')
ax.set_title('Clustering of Data')
plt.show()

# Import climate data with proficiency
climate_data = pd.read_csv('World_bank.csv', skiprows=4)
climate_data = climate_data.fillna(climate_data.mean(numeric_only=True))

# select the columns for analysis
cols = ['1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969',
        '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979',
        '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989',
        '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999',
        '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
        '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
data = climate_data[cols]
data

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Run KMeans with 7 clusters
kmeans = KMeans(n_clusters=7, random_state=42).fit(scaled_data)

# Assign cluster labels to data
climate_data['Cluster'] = kmeans.labels_

# Display cluster-wise country count
print(climate_data.groupby('Cluster')['Country Name'].count())

# select one country from each cluster
sample_countries = climate_data.groupby('Cluster').apply(lambda x: x.sample(1))

# Inter-cluster country comparison analysis
cluster_0 = climate_data[climate_data['Cluster'] == 0]
print(cluster_0[cols].mean())

# Cross-Cluster Country Comparison Analysis.
cluster_1 = climate_data[climate_data['Cluster'] == 1]
print(cluster_1[cols].mean())

# Analyze emerging patterns and tendencies
trend_cluster_0 = cluster_0[cols].mean()
trend_cluster_1 = cluster_1[cols].mean()
print('Trend similarity between cluster 0 and cluster 1:', np.corrcoef(trend_cluster_0, trend_cluster_1)[0,1])

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate scatter plot for PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
colors = ['blue', 'green', 'red', 'purple', 'orange']
for i in range(5):
    plt.scatter(pca_data[kmeans.labels_==i,0], pca_data[kmeans.labels_==i,1], color=colors[i])
plt.title('Principal Component Analysis')
plt.xlabel('1st PCA')
plt.ylabel('2nd PCA')
plt.show()
