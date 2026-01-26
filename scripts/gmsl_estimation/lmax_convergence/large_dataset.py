import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set plotting theme to resemble theme_minimal
sns.set_theme(style="whitegrid")

# Load the dataset
file_path = "work/6-lmax_issues/outputs/explore_lmax/large_dataset.csv"

try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit(1)

# Preprocessing
# Convert lmax to category for discrete coloring (similar to factor in R)
if 'lmax' in data.columns:
    unique_lmax = sorted(data['lmax'].unique())
    data['lmax'] = pd.Categorical(data['lmax'], categories=unique_lmax, ordered=True)

# Convert SLC standard deviation to mm (assuming input is in meters)
data['slc_std'] = data['slc_std'] * 1000

# Plot 1: Scatter plot
plt.figure(figsize=(10, 6))
p1 = sns.scatterplot(
    data=data,
    x='ice_gmsl_target_std',
    y='slc_std',
    hue='lmax',
    alpha=0.5
)
p1.set_title("Sea Level Change vs Ice GMSL Target Data")
p1.set_xlabel("Ice GMSL Target Data (mm)")
p1.set_ylabel("Sea Level Change Standard Deviation (mm)")

plt.show()

# Preprocessing for second plot
# Calculate difference squared
# Note: slc_std is already converted to mm above, assuming ice_gmsl_target_std is also in mm
data['diff_squared'] = (data['slc_std'] - data['ice_gmsl_target_std']) ** 2

# Plot 2: Violin plot
plt.figure(figsize=(10, 6))
p2 = sns.violinplot(
    data=data,
    x='lmax',
    y='diff_squared'
)
p2.set_title("Difference Squared between SLC Std and Ice GMSL Target Std by Lmax")
p2.set_xlabel("Lmax")
p2.set_ylabel("Difference Squared")

plt.show()
