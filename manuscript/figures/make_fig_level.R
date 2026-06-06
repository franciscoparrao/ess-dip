#!/usr/bin/env Rscript
# Figure 4: empirical level and power of the calibrated test vs effective sample
# size. (a) false-split rate on single-class (H0) fields, ESS calibration vs
# nominal-n calibration, against the nominal alpha; (b) power on two-class (H1)
# fields. Shows the effective-sample-size calibration controls the level at or
# below alpha and keeps it stable as n_eff varies, while retaining power.

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr)
  library(ggplot2); library(patchwork)
})

d <- read_csv("../../experiments/results/level_neff.csv", show_col_types = FALSE)

base_theme <- theme_minimal(base_size = 9) +
  theme(panel.grid.minor = element_blank(),
        axis.line = element_line(linewidth = 0.3, colour = "grey20"),
        axis.ticks = element_line(linewidth = 0.3, colour = "grey20"),
        legend.position = c(0.98, 0.98), legend.justification = c(1, 1),
        legend.title = element_blank(), legend.background = element_blank(),
        legend.key.size = unit(9, "pt"),
        plot.title = element_text(size = 9, face = "bold"))

## (a) false-split rate vs n_eff
dfa <- d |>
  select(n_eff, `ESS calibration` = fsr_ess, `nominal size` = fsr_nominal) |>
  pivot_longer(-n_eff, names_to = "series", values_to = "fsr")

pA <- ggplot(dfa, aes(n_eff, fsr, colour = series, shape = series)) +
  geom_hline(yintercept = 0.05, linetype = "dashed", colour = "grey40",
             linewidth = 0.3) +
  annotate("text", x = max(dfa$n_eff), y = 0.062, label = "alpha == 0.05",
           parse = TRUE, hjust = 1, size = 2.7, colour = "grey40") +
  geom_point(size = 1.8) +
  scale_colour_manual(values = c("ESS calibration" = "#D55E00",
                                 "nominal size" = "#999999")) +
  scale_shape_manual(values = c("ESS calibration" = 16, "nominal size" = 1)) +
  scale_x_log10() +
  scale_y_continuous(limits = c(0, NA), expand = expansion(c(0, 0.05))) +
  labs(title = "(a) False-split rate (one-class fields)",
       x = expression(effective~sample~size~~n[eff]),
       y = "false-split rate") +
  base_theme

## (b) power vs n_eff
pB <- ggplot(d, aes(n_eff, power)) +
  geom_point(size = 1.8, colour = "#0072B2") +
  scale_x_log10() +
  scale_y_continuous(limits = c(0, 1.001), expand = expansion(c(0, 0.02))) +
  labs(title = "(b) Power (two-class fields)",
       x = expression(effective~sample~size~~n[eff]),
       y = "correct-split rate") +
  base_theme + theme(legend.position = "none")

fig <- pA + pB + plot_layout(widths = c(1, 1))
ggsave("level.png", fig, width = 7.0, height = 3.0, dpi = 300, bg = "white")
ggsave("level.pdf", fig, width = 7.0, height = 3.0, bg = "white")
cat("wrote level.png / level.pdf\n")
