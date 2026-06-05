#!/usr/bin/env Rscript
# Figure 2: computational performance (reviewer item C2).
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(ggplot2); library(patchwork); library(tidyr)
})

t <- read_csv("../../experiments/results/timing.csv", show_col_types = FALSE)

base_theme <- theme_minimal(base_size = 9) +
  theme(panel.grid.minor = element_blank(),
        axis.line = element_line(linewidth = 0.3, colour = "grey20"),
        axis.ticks = element_line(linewidth = 0.3, colour = "grey20"),
        legend.position = c(0.02, 0.98), legend.justification = c(0, 1),
        legend.title = element_blank(), legend.background = element_blank(),
        legend.key.size = unit(9, "pt"),
        plot.title = element_text(size = 9, face = "bold"))

## panel A: time vs N (log-log)
dA <- t |> filter(axis == "N") |>
  select(N, `ESS-Dip` = ess_dip, `gap statistic` = gap) |>
  pivot_longer(-N, names_to = "method", values_to = "sec") |> filter(!is.na(sec))
slope <- with(filter(dA, method == "ESS-Dip"),
              round(coef(lm(log(sec) ~ log(N)))[2], 2))
pal <- c("ESS-Dip" = "#D55E00", "gap statistic" = "#999999")

pA <- ggplot(dA, aes(N, sec, colour = method)) +
  geom_line(aes(linewidth = method == "ESS-Dip")) + geom_point(size = 1.5) +
  scale_colour_manual(values = pal) +
  scale_linewidth_manual(values = c(`FALSE` = 0.5, `TRUE` = 1.1), guide = "none") +
  scale_x_log10() + scale_y_log10() +
  annotation_logticks(sides = "bl", linewidth = 0.2) +
  annotate("text", x = 6e3, y = 1.0, hjust = 0, size = 2.9,
           label = paste0("ESS-Dip slope = ", slope)) +
  labs(title = "(a) Time vs image size", x = "pixels N", y = "seconds") +
  base_theme

## panel B: time vs p
dB <- t |> filter(axis == "p") |> select(p, sec = ess_dip)
pB <- ggplot(dB, aes(p, sec)) +
  geom_line(colour = "#D55E00", linewidth = 1.0) +
  geom_point(colour = "#D55E00", size = 1.6) +
  scale_y_continuous(limits = c(0, NA), expand = expansion(c(0, 0.05))) +
  labs(title = "(b) Time vs band count", x = "bands p", y = "seconds") +
  base_theme + theme(legend.position = "none")

fig <- pA + pB + plot_layout(widths = c(1.2, 1))
ggsave("timing.png", fig, width = 7.0, height = 2.9, dpi = 300, bg = "white")
ggsave("timing.pdf", fig, width = 7.0, height = 2.9, bg = "white")
cat("wrote timing.png / timing.pdf  (ESS-Dip slope =", slope, ")\n")
