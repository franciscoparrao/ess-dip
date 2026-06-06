#!/usr/bin/env Rscript
# Qualitative figure: ESS-Dip recovers the correct map on a mixed scene
# (classes + trend) where the gap statistic and the Hennig-Lin spatial null
# over-segment. Ground truth is known (synthetic). Reads tidy CSVs, writes a
# 4-panel map (PNG+PDF). ESS-Dip cluster colours are matched to ground truth.

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(ggplot2); library(patchwork)
})

base <- "../../experiments/results/"
maps <- read_csv(paste0(base, "fig_recover_maps.csv"), show_col_types = FALSE)
meta <- read_csv(paste0(base, "fig_recover_meta.csv"), show_col_types = FALSE)
kv <- function(k) meta$value[meta$key == k]

map_theme <- theme_void(base_size = 9) +
  theme(plot.title = element_text(size = 9, face = "bold", hjust = 0.5,
                                  margin = margin(b = 3)),
        legend.position = "none", plot.margin = margin(4, 4, 4, 4))

panel <- function(name, title, palette) {
  df <- maps |> filter(panel == name) |> mutate(label = factor(label))
  ggplot(df, aes(col, row)) +
    geom_raster(aes(fill = label)) +
    scale_fill_manual(values = palette) +
    scale_y_reverse() + coord_fixed(expand = FALSE) +
    labs(title = title) + map_theme
}

# ground truth & ESS-Dip share a 3-colour palette (Okabe-Ito) so a correct
# recovery looks identical to (a)
truth_pal <- setNames(c("#0072B2", "#E69F00", "#009E73"),
                      c("c0", "c1", "c2"))
# over-segmenting methods: qualitative 12-colour palette
seg12 <- c("#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462",
           "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f")
seg_pal <- function(name) {
  lv <- levels(factor(maps$label[maps$panel == name]))
  setNames(seg12[seq_along(lv)], lv)
}

pA <- panel("truth",  sprintf("(a) Ground truth (K = %d)", kv("k_true")), truth_pal)
pB <- panel("gap",    sprintf("(b) Gap statistic (K = %d)", kv("k_gap")),  seg_pal("gap"))
pC <- panel("hl",     sprintf("(c) Hennig-Lin (K = %d)", kv("k_hl")),      seg_pal("hl"))
pD <- panel("essdip", sprintf("(d) ESS-Dip (K = %d)", kv("k_ess")),        truth_pal)

fig <- (pA | pB) / (pC | pD)
ggsave("recover.png", fig, width = 6.6, height = 6.9, dpi = 300, bg = "white")
ggsave("recover.pdf", fig, width = 6.6, height = 6.9, bg = "white")
cat(sprintf("wrote recover.png / recover.pdf  (gap %d, HL %d, ESS-Dip %d, truth %d)\n",
            kv("k_gap"), kv("k_hl"), kv("k_ess"), kv("k_true")))
