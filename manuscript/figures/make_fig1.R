#!/usr/bin/env Rscript
# Figure 1 for the manuscript: specificity-vs-accuracy and accuracy-vs-K.
# Reads the benchmark CSV, relabels methods to the paper's nomenclature,
# and writes a publication-quality two-panel figure (PNG + PDF).

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr)
  library(ggplot2); library(patchwork)
})

bench <- read_csv("../../experiments/results/bench.csv", show_col_types = FALSE) |>
  mutate(correct = k_hat == truth_k)

lbl <- c(classical_gap = "Gap", silhouette = "Silhouette",
         calinski_harabasz = "Calinski-Harabasz", davies_bouldin = "Davies-Bouldin",
         ess_dip_local = "ESS-Dip", ess_dip_adaptive = "global range",
         ess_dip = "no trend removal", ess_dip_detrend = "forced trend removal",
         decluster_dip = "declustering")
ord <- c("Gap", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin", "ESS-Dip",
         "global range", "no trend removal", "forced trend removal", "declustering")
recode_m <- function(x) factor(unname(lbl[x]), levels = ord)

## ---- panel A: the three regime scores per method ------------------------
specA <- bench |> filter(kind %in% c("null", "trend")) |>
  group_by(method) |> summarise(value = mean(correct), .groups = "drop") |>
  mutate(metric = "Specificity (class-free)")
strA <- bench |> filter(kind == "struct") |>
  group_by(method) |> summarise(value = mean(correct), .groups = "drop") |>
  mutate(metric = "Accuracy (structured)")
mixA <- bench |> filter(kind == "mixed") |>
  group_by(method) |> summarise(value = mean(correct), .groups = "drop") |>
  mutate(metric = "Accuracy (mixed)")
dfA <- bind_rows(specA, strA, mixA) |>
  mutate(method = recode_m(method),
         metric = factor(metric, levels = c("Specificity (class-free)",
                          "Accuracy (structured)", "Accuracy (mixed)")))

pal3 <- c("Specificity (class-free)" = "#0072B2",
          "Accuracy (structured)"    = "#E69F00",
          "Accuracy (mixed)"         = "#009E73")

base_theme <- theme_minimal(base_size = 9) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        axis.line = element_line(linewidth = 0.3, colour = "grey20"),
        axis.ticks = element_line(linewidth = 0.3, colour = "grey20"),
        legend.position = "bottom", legend.title = element_blank(),
        legend.key.size = unit(9, "pt"),
        plot.title = element_text(size = 9, face = "bold"),
        axis.text.x = element_text(angle = 30, hjust = 1))

pA <- ggplot(dfA, aes(method, value, fill = metric)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.72) +
  scale_fill_manual(values = pal3) +
  scale_y_continuous(limits = c(0, 1.001), expand = expansion(c(0, 0.02))) +
  labs(title = "(a) Specificity versus accuracy", x = NULL, y = "rate") +
  base_theme

## ---- panel B: exact-K accuracy vs true K (structured) -------------------
keepB <- c("Gap", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin", "ESS-Dip")
dfB <- bench |> filter(kind == "struct") |>
  group_by(method, k_true) |> summarise(acc = mean(correct), .groups = "drop") |>
  mutate(method = recode_m(method)) |>
  filter(method %in% keepB) |> mutate(method = factor(method, levels = keepB))

palB <- c("Gap" = "#999999", "Silhouette" = "#56B4E9",
          "Calinski-Harabasz" = "#E69F00", "Davies-Bouldin" = "#CC79A7",
          "ESS-Dip" = "#D55E00")

pB <- ggplot(dfB, aes(k_true, acc, colour = method)) +
  geom_line(aes(linewidth = method == "ESS-Dip")) +
  geom_point(size = 1.5) +
  scale_colour_manual(values = palB) +
  scale_linewidth_manual(values = c(`FALSE` = 0.5, `TRUE` = 1.1), guide = "none") +
  scale_y_continuous(limits = c(0, 1.001), expand = expansion(c(0, 0.02))) +
  scale_x_continuous(breaks = 2:5) +
  labs(title = "(b) Accuracy versus number of classes",
       x = "true number of classes", y = "exact-K accuracy") +
  base_theme +
  theme(axis.text.x = element_text(angle = 0),
        legend.position = "right",
        legend.key.height = unit(11, "pt"))

fig <- pA + pB + plot_layout(widths = c(1.05, 1))

ggsave("benchmark.png", fig, width = 7.2, height = 3.2, dpi = 300, bg = "white")
ggsave("benchmark.pdf", fig, width = 7.2, height = 3.2, bg = "white")
cat("wrote benchmark.png / benchmark.pdf\n")
