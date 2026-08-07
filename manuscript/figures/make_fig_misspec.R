#!/usr/bin/env Rscript
# Figure (Sec. 4.8): robustness of the calibration to a multiplicative error f
# in the estimated range (exp 37). (a) false-split rate on the null world stays
# at or below alpha for every factor, including a four-fold underestimate of
# the range; (b) recovery on the structured K=4 world survives underestimation
# unharmed and degrades only under overestimation, collapsing to silence (K=1),
# never to invention. Companion to the exact values in the table.

suppressPackageStartupMessages({
  library(readr); library(dplyr)
  library(ggplot2); library(patchwork)
})

d <- read_csv("../../experiments/results/range_misspec.csv",
              show_col_types = FALSE)

base_theme <- theme_minimal(base_size = 9) +
  theme(panel.grid.minor = element_blank(),
        axis.line = element_line(linewidth = 0.3, colour = "grey20"),
        axis.ticks = element_line(linewidth = 0.3, colour = "grey20"),
        legend.position = "none",
        plot.title = element_text(size = 9, face = "bold"))

fbreaks <- c(0.25, 0.5, 0.75, 1, 1.5, 2, 4)
flabels <- c("1/4", "1/2", "3/4", "1", "3/2", "2", "4")

## (a) null world: false-split rate vs factor
da <- d |> filter(world == "null") |>
  group_by(factor) |> summarise(fsr = mean(k_hat > 1), .groups = "drop")

pA <- ggplot(da, aes(factor, fsr)) +
  geom_hline(yintercept = 0.05, linetype = "dashed", colour = "grey40",
             linewidth = 0.3) +
  annotate("text", x = 4, y = 0.062, label = "alpha == 0.05", parse = TRUE,
           hjust = 1, size = 2.7, colour = "grey40") +
  geom_vline(xintercept = 1, linetype = "dotted", colour = "grey60",
             linewidth = 0.3) +
  geom_line(colour = "#D55E00", linewidth = 0.4) +
  geom_point(size = 1.8, colour = "#D55E00") +
  scale_x_log10(breaks = fbreaks, labels = flabels) +
  scale_y_continuous(limits = c(0, 0.27), expand = expansion(c(0, 0.02))) +
  labs(title = "(a) False-split rate (one-class world)",
       x = "range factor f  (estimated range multiplied by f)",
       y = "false-split rate") +
  base_theme

## (b) structured world: mean K-hat vs factor
db <- d |> filter(world == "struct") |>
  group_by(factor) |>
  summarise(mean_k = mean(k_hat), sd_k = sd(k_hat), .groups = "drop")

pB <- ggplot(db, aes(factor, mean_k)) +
  geom_hline(yintercept = 4, linetype = "dashed", colour = "grey40",
             linewidth = 0.3) +
  annotate("text", x = 0.25, y = 4.15, label = "true K = 4", hjust = 0,
           size = 2.7, colour = "grey40") +
  geom_vline(xintercept = 1, linetype = "dotted", colour = "grey60",
             linewidth = 0.3) +
  geom_ribbon(aes(ymin = mean_k - sd_k, ymax = mean_k + sd_k),
              fill = "#0072B2", alpha = 0.15) +
  geom_line(colour = "#0072B2", linewidth = 0.4) +
  geom_point(size = 1.8, colour = "#0072B2") +
  scale_x_log10(breaks = fbreaks, labels = flabels) +
  scale_y_continuous(limits = c(0.8, 4.6), breaks = 1:4) +
  labs(title = "(b) Recovered classes (four-class world)",
       x = "range factor f  (estimated range multiplied by f)",
       y = expression(mean~~hat(K))) +
  base_theme

fig <- pA + pB + plot_layout(widths = c(1, 1))
ggsave("misspec.png", fig, width = 7.0, height = 3.0, dpi = 300, bg = "white")
ggsave("misspec.pdf", fig, width = 7.0, height = 3.0, bg = "white")
cat("wrote misspec.png / misspec.pdf\n")
