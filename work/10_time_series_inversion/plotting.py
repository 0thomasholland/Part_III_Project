# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_file = "time_series_inversion_results.csv"
output_data = pd.read_csv(data_file, index_col=0)
print(output_data)

plotable_std = 1

#%%
# plot means vs years with fill between mean +/- plotable_stds*std
years = output_data.index.astype(str)
gis_means = output_data["GIS_mean"]
gis_stds = np.sqrt(output_data["GIS_marginal_cov"])
wais_means = output_data["WAIS_mean"]
wais_stds = np.sqrt(output_data["WAIS_marginal_cov"])
eais_means = output_data["EAIS_mean"]
eais_stds = np.sqrt(output_data["EAIS_marginal_cov"])
global_means = output_data["global_mean"]
global_stds = np.sqrt(output_data["global_cov"])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
ax1.plot(years, gis_means, label="GIS Mean", color="blue")
ax1.fill_between(
    years,
    gis_means - plotable_std * gis_stds,
    gis_means + plotable_std * gis_stds,
    color="blue",
    alpha=0.2,
    label=f"GIS ± {plotable_std} Std Dev",
)
ax1.plot(years, wais_means, label="WAIS Mean", color="orange")
ax1.fill_between(
    years,
    wais_means - plotable_std * wais_stds,
    wais_means + plotable_std * wais_stds,
    color="orange",
    alpha=0.2,
    label=f"WAIS ± {plotable_std} Std Dev",
)
ax1.plot(years, eais_means, label="EAIS Mean", color="green")
ax1.fill_between(
    years,
    eais_means - plotable_std * eais_stds,
    eais_means + plotable_std * eais_stds,
    color="green",
    alpha=0.2,
    label=f"EAIS ± {plotable_std} Std Dev",
)
ax1.set_title("Posterior GMSL Contribution Means and Uncertainties for GIS, WAIS, EAIS, using the DUACS Annual Data (1993-2020)")
ax1.set_ylabel("Mass Change (mm/year)")
ax1.legend()

ax2.plot(years, global_means, label="Global Mean", color="red")
ax2.fill_between(
    years,
    global_means - plotable_std * global_stds,
    global_means + plotable_std * global_stds,
    color="red",
    alpha=0.2,
    label=f"Global ± {plotable_std} Std Dev",
)
ax2.set_title("Posterior GMSL Mean and Uncertainty")
ax2.set_xlabel("Year")
ax2.set_ylabel("Mass Change (mm/year)")
ax2.legend()
ax2.set_xticks(years[::5]) 
plt.tight_layout()
plt.show()
# %%
