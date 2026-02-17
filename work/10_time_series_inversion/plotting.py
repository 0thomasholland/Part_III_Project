# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_file = "time_series_inversion_results.csv"
output_data = pd.read_csv(data_file, index_col=0)
print(output_data)

#%%
# plot means vs years with fill between mean +/- 2*std
years = output_data.index.astype(str)
gis_means = output_data["GIS_mean"]
gis_stds = np.sqrt(output_data["GIS_marginal_cov"])
wais_means = output_data["WAIS_mean"]
wais_stds = np.sqrt(output_data["WAIS_marginal_cov"])
eais_means = output_data["EAIS_mean"]
eais_stds = np.sqrt(output_data["EAIS_marginal_cov"])
global_means = output_data["global_mean"]
global_stds = np.sqrt(output_data["global_cov"])

plt.figure(figsize=(10, 6))
plt.plot(years, gis_means, label="GIS Mean", color="blue")
plt.fill_between(
    years, gis_means - 2 * gis_stds, gis_means + 2 * gis_stds, color="blue", alpha=0.2
)
plt.plot(years, wais_means, label="WAIS Mean", color="orange")
plt.fill_between(
    years, wais_means - 2 * wais_stds, wais_means + 2 * wais_stds, color="orange", alpha=0.2
)
plt.plot(years, eais_means, label="EAIS Mean", color="green")
plt.fill_between(
    years, eais_means - 2 * eais_stds, eais_means + 2 * eais_stds, color="green", alpha=0.2 
)
plt.plot(years, global_means, label="Global Mean", color="black")
plt.fill_between(
    years, global_means - 2 * global_stds, global_means + 2 * global_stds, color="black", alpha=0.2
)

plt.xlabel("Year")
# plt x ticks every 5 years
plt.xticks(years[::5])
plt.ylabel("GMSL Change (attribution) (mm)")
plt.title("")
plt.legend()
plt.show()
# %%
