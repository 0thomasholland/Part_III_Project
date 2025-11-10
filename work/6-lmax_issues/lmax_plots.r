library(ggplot2)
library(dplyr)

text <- "slc_no_translation_no_gmsl_weighting"
  
data_full <- read.csv(paste0("work/5-distribution_mapping/outputs/lmax_problems/", text, ".csv"))

# plot on same plot space with dual axes that on log scale
p1 <- ggplot(data_full, aes(x = lmax)) +
  geom_point(aes(y = mean, color = "Mean"), size = 2) +
  geom_point(aes(y = std * 1e10, color = "Std Dev (x1e3)"), size = 2) +
  # geom_smooth(aes(y = mean, color = "Mean"), method = "lm") +
  # geom_smooth(aes(y = std * 1e10, color = "Std Dev (x1e3)"), method = "lm") +
  # scale_y_log10(
  #   name = "Mean",
  #   sec.axis = sec_axis(~ . / 1e3, name = "Std Dev")
  # ) +
  # scale_x_log10() +
  scale_y_continuous(
    name = "Mean",
    sec.axis = sec_axis(~ . / 1e10, name = "Std Dev")
  ) +
  scale_x_continuous(
    breaks = seq(0, max(data_full$lmax), by = 50)
  ) +
  labs(x = "lmax", title = "Mean and Std Dev vs lmax") +
  theme_minimal() +
  theme(
    legend.position = "top",
    legend.title = element_blank()
  ) +
  scale_color_manual(values = c("Mean" = "blue", "Std Dev (x1e3)" = "red"))

print(p1)
ggsave(paste0("work/5-distribution_mapping/outputs/lmax_problems/", text, "_lmax_plot.png"), plot = p1, width = 8, height = 6)

p2 <- ggplot(data_full, aes(x = lmax)) +
  geom_point(aes(y = mean, color = "Mean"), size = 2) +
  geom_point(aes(y = std * 1e10, color = "Std Dev (x1e3)"), size = 2) +
  geom_smooth(aes(y = mean, color = "Mean"), method = "lm") +
  geom_smooth(aes(y = std * 1e10, color = "Std Dev (x1e3)"), method = "lm") +
  scale_y_log10(
    name = "Mean",
    sec.axis = sec_axis(~ . / 1e3, name = "Std Dev")
  ) +
  scale_x_log10() +
  # scale_y_continuous(
  #   name = "Mean",
  #   sec.axis = sec_axis(~ . / 1e10, name = "Std Dev")
  # ) +
  # scale_x_continuous(
  #   breaks = seq(0, max(data_full$lmax), by = 50)
  # ) +
  labs(x = "lmax", title = "Mean and Std Dev vs lmax") +
  theme_minimal() +
  theme(
    legend.position = "top",
    legend.title = element_blank()
  ) +
  scale_color_manual(values = c("Mean" = "blue", "Std Dev (x1e3)" = "red"))

print(p2)
ggsave(paste0("work/5-distribution_mapping/outputs/lmax_problems/", text, "_lmax_plot1.png"), plot = p2, width = 8, height = 6)

generate_gaussian <- function(mean, std, lmax) {
  x <- seq(0 - 4*std, 0 + 4*std, length.out = 200)
  y <- dnorm(x, mean = 0, sd = std)
  data.frame(x = x, y = y, lmax = lmax)
}
distribution_data <- data_full %>%
  rowwise() %>%
  do(generate_gaussian(.$mean, .$std, .$lmax)) %>%
  ungroup()
p3 <-ggplot(distribution_data, aes(x = x, y = y, color = lmax, group = lmax)) +
  geom_line(linewidth = 1) +
  scale_color_viridis_c() +
  labs(
    title = "Gaussian Distributions (only by standard deviation, ignores mean) by lmax",
    x = "Value",
    y = "Density",
    color = "lmax"
  ) +
  theme_minimal()

print(p3)
ggsave(paste0("work/5-distribution_mapping/outputs/lmax_problems/", text, "_lmax_gaussian_plot.png"), plot = p3, width = 8, height = 6)