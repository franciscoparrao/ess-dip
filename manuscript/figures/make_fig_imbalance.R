#!/usr/bin/env Rscript
# Figure (Sec. 4.9): detection of a contiguous minority class of proportion pi
# (exp 36, true K = 2, scenes paired across methods). (a) minority-detection
# rate P(K-hat >= 2): ESS-Dip is reliable down to pi = 0.20, ESS-Dip-R extends
# the regime to 0.10; the baselines' apparent detection at extreme imbalance is
# not creditable (their false-split rate on class-free scenes is near one,
# dashed lines). (b) mean K-hat: at pi = 0.02 the silhouette reports ~3 classes
# by fragmenting the majority, while the calibrated family degrades to silence
# (K-hat = 1). Companion to the exact values in the table.

suppressPackageStartupMessages({
  library(readr); library(dplyr)
  library(ggplot2); library(patchwork)
})

d <- read_csv("../../experiments/results/imbalance_sweep.csv",
              show_col_types = FALSE)

lv <- c("ESS-Dip", "ESS-Dip-R", "Gap statistic", "Silhouette")
d <- d |>
  mutate(method = recode(method,
                         ess_dip = "ESS-Dip", ess_dip_R = "ESS-Dip-R",
                         gap = "Gap statistic", silhouette = "Silhouette"),
         method = factor(method, levels = lv))

cols   <- c("ESS-Dip" = "#0072B2", "ESS-Dip-R" = "#D55E00",
            "Gap statistic" = "#999999", "Silhouette" = "#CC79A7")
lts    <- c("ESS-Dip" = "solid", "ESS-Dip-R" = "solid",
            "Gap statistic" = "dashed", "Silhouette" = "dashed")
shapes <- c("ESS-Dip" = 16, "ESS-Dip-R" = 17,
            "Gap statistic" = 1, "Silhouette" = 2)

base_theme <- theme_minimal(base_size = 9) +
  theme(panel.grid.minor = element_blank(),
        axis.line = element_line(linewidth = 0.3, colour = "grey20"),
        axis.ticks = element_line(linewidth = 0.3, colour = "grey20"),
        legend.title = element_blank(), legend.background = element_blank(),
        legend.key.size = unit(9, "pt"),
        plot.title = element_text(size = 9, face = "bold"))

s <- d |>
  group_by(method, pi) |>
  summarise(det = mean(k_hat >= 2), mean_k = mean(k_hat), .groups = "drop")

pbreaks <- c(0.02, 0.05, 0.10, 0.20, 0.30, 0.50)

## (a) minority-detection rate
pA <- ggplot(s, aes(pi, det, colour = method, linetype = method,
                    shape = method)) +
  geom_line(linewidth = 0.4) +
  geom_point(size = 1.8) +
  scale_colour_manual(values = cols) +
  scale_linetype_manual(values = lts) +
  scale_shape_manual(values = shapes) +
  scale_x_log10(breaks = pbreaks,
                labels = c(".02", ".05", ".10", ".20", ".30", ".50")) +
  scale_y_continuous(limits = c(0, 1.02), expand = expansion(c(0, 0.02))) +
  labs(title = "(a) Minority-detection rate",
       x = expression(minority~proportion~~pi),
       y = expression(P(hat(K) >= 2))) +
  base_theme +
  theme(legend.position = c(0.98, 0.05), legend.justification = c(1, 0))

## (b) mean K-hat
pB <- ggplot(s, aes(pi, mean_k, colour = method, linetype = method,
                    shape = method)) +
  geom_hline(yintercept = 2, linetype = "dashed", colour = "grey40",
             linewidth = 0.3) +
  annotate("text", x = 0.5, y = 2.12, label = "true K = 2", hjust = 1,
           size = 2.7, colour = "grey40") +
  geom_line(linewidth = 0.4) +
  geom_point(size = 1.8) +
  scale_colour_manual(values = cols) +
  scale_linetype_manual(values = lts) +
  scale_shape_manual(values = shapes) +
  scale_x_log10(breaks = pbreaks,
                labels = c(".02", ".05", ".10", ".20", ".30", ".50")) +
  scale_y_continuous(limits = c(0.9, 3.1), breaks = 1:3) +
  labs(title = "(b) Estimated number of classes",
       x = expression(minority~proportion~~pi),
       y = expression(mean~~hat(K))) +
  base_theme + theme(legend.position = "none")

fig <- pA + pB + plot_layout(widths = c(1, 1))
ggsave("imbalance.png", fig, width = 7.0, height = 3.0, dpi = 300, bg = "white")
ggsave("imbalance.pdf", fig, width = 7.0, height = 3.0, bg = "white")
cat("wrote imbalance.png / imbalance.pdf\n")
