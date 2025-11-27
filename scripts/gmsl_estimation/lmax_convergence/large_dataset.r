library(ggplot2)
library(dplyr)

# Load the dataset
data <- read.csv("work/6-lmax_issues/outputs/explore_lmax/large_dataset.csv")

# colour by lmax in discrete steps of lmax
data <- data %>%
    mutate(lmax = factor(lmax)) %>%
    mutate(slc_std = slc_std * 1000)
p1 <- ggplot(data, aes(x = ice_gmsl_target_std, y = slc_std, color = lmax)) +
    geom_point(alpha = 0.5) +
    labs(
        title = "Sea Level Change vs Ice GMSL Target Data",
        x = "Ice GMSL Target Data (mm)",
        y = "Sea Level Change Standard Deviation (mm)"
    ) +
    theme_minimal()

print(p1)


# plot the difference between slc_std and ice_gmsl_target_std squared, coloured by lmax as violin plot
data <- data %>%
    mutate(diff_squared = (slc_std - ice_gmsl_target_std)^2)
p2 <- ggplot(data, aes(x = lmax, y = diff_squared)) +
    geom_violin() +
    labs(
        title = "Difference Squared between SLC Std and Ice GMSL Target Std by Lmax",
        x = "Lmax",
        y = "Difference Squared"
    ) +
    theme_minimal()

print(p2)
