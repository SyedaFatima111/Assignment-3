import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Specify the file path
file_path = 'World_bank.csv'

# Load the CSV file into a dataframe
World_Bank_data = pd.read_csv(World_bank, skiprows=4)

# Display the dataframe
World_Bank_data

# Display the first five rows using iloc
World_Bank_data.iloc[:5]

World_Bank_data.value_counts('Indicator Name')

def exponential_growth(x, a, b):
    return a * np.exp(b * x)

x = np.array(range(1960, 2022)) 
us_data = World_Bank_data[World_Bank_data["Country Name"] == "Canada"]

y = (np.array(us_data[us_data['Indicator Name']== "GDP (current US$)"]))[0][4:66]

popt, pcov = curve_fit(exponential_growth, x, y)

from scipy import stats
# Specify forecasting timeframe using years.
prediction_years = np.array(range(1960, 2021))

# Generate predictions using language model
predicted_values = exponential_growth(prediction_years, *popt)

#Compute confidence intervals using err_ranges
def err_ranges(func, xdata, ydata, popt, pcov, alpha=0.05):
    perr = np.sqrt(np.diag(pcov))
    n = len(ydata)
    dof = max(0, n - len(popt))
    tval = np.abs(stats.t.ppf(alpha / 2, dof))
    ranges = tval * perr
    return ranges

lower_bounds, upper_bounds = err_ranges(exponential_growth, x, y, popt, pcov)

# Plot optimal function with confidence
plt.plot(x, y, 'o', label='World_Bank_data')
plt.plot(x, y, 'r-', label='fit')
plt.fill_between(prediction_years, predicted_values - upper_bounds, predicted_values + lower_bounds, alpha=0.3)
plt.title('Best Fitting Function Vs Confidence Range')
plt.xlabel('Years')
plt.ylabel('GDP (in billions of dollars)')
plt.title('Best Fitting Function Vs Confidence Range')

plt.legend()
plt.show()