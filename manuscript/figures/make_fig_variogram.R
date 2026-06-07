#!/usr/bin/env Rscript
# Within-class variograms on the real scenes: empirical radial semivariogram of
# the first PC averaged over pure single-class windows, with the fitted model.
# Shows the within-class autocorrelation the calibration relies on is real and
# has a well-defined range on actual imagery.

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(ggplot2)
})

d <- read_csv("../../experiments/results/real_variograms.csv",
              show_col_types = FALSE) |>
  mutate(scene = factor(scene, levels = c("Indian Pines", "Salinas", "Sentinel-2")))

pal <- c("Indian Pines" = "#0072B2", "Salinas" = "#E69F00",
         "Sentinel-2" = "#009E73")

p <- ggplot(d, aes(lag, gamma, colour = scene)) +
  geom_line(aes(y = fit), linewidth = 0.7) +
  geom_point(size = 1.7) +
  scale_colour_manual(values = pal) +
  scale_x_continuous(breaks = seq(0, 8, 2)) +
  scale_y_continuous(limits = c(0, NA), expand = expansion(c(0, 0.05))) +
  labs(x = "lag (pixels)", y = "standardised semivariance",
       colour = NULL,
       title = "Within-class variograms (points: empirical; lines: fitted model)") +
  theme_minimal(base_size = 9) +
  theme(panel.grid.minor = element_blank(),
        axis.line = element_line(linewidth = 0.3, colour = "grey20"),
        axis.ticks = element_line(linewidth = 0.3, colour = "grey20"),
        legend.position = c(0.99, 0.02), legend.justification = c(1, 0),
        legend.background = element_blank(),
        plot.title = element_text(size = 9, face = "bold"))

ggsave("variogram.png", p, width = 4.6, height = 3.2, dpi = 300, bg = "white")
ggsave("variogram.pdf", p, width = 4.6, height = 3.2, bg = "white")
cat("wrote variogram.png / variogram.pdf\n")
