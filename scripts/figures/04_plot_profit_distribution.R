# -----------------------------------------------------------------------------
# 04_plot_profit_distribution.R
#
# This script generates the final profit distribution figure for the manuscript
# (e.g., Figure 5). It uses the detailed, farmer-level profit data to create
# two panels of ridgeline plots:
#   1. The distribution of average profit for all farmers.
#   2. The distribution of average profit for only those farmers equipped
#      for irrigation.
#
# The script is designed to precisely replicate the formatting of the original
# plotting code to ensure publication-quality figures.
# -----------------------------------------------------------------------------

# --- 1. Load Libraries ---
library(readr)
library(dplyr)
library(purrr)
library(ggplot2)
library(ggridges)
library(patchwork)
library(scales)

# --- 2. Set Up Paths and Parameters ---
# Robust detection of project root whether run via Rscript or interactively
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)])

if (length(script_path) > 0) {
  # Running via Rscript: go two levels up from the script location
  # e.g., /.../pychamp_gcp/scripts/figures/04_plot_profit_distribution.R
  #   -> /.../pychamp_gcp
  PROJECT_ROOT <- normalizePath(file.path(dirname(script_path), "..", ".."))
} else {
  # Probably running interactively; assume working dir is scripts/figures
  # e.g., setwd("/.../pychamp_gcp/scripts/figures")
  PROJECT_ROOT <- normalizePath(file.path(getwd(), "..", ".."))
}

DATA_DIR <- file.path(PROJECT_ROOT, "outputs", "data_for_figures")
FIGURES_DIR <- file.path(PROJECT_ROOT, "outputs", "figures")
if (!dir.exists(FIGURES_DIR)) {
  dir.create(FIGURES_DIR, recursive = TRUE)
}

# --- 3. Define Plotting Styles ---
policy_shortnames <- c("BAU", "UR", "FB", "PR-I", "PR-II", "R+PR")

colors <- c(
  "BAU" = "#808080", "UR" = "#C00000", "FB" = "#FF7F0E",
  "PR-I" = "#4169E1", "PR-II" = "#00BFFF", "R+PR" = "#DAA520"
)

# Function to soften colors for the density plot fill
soften_color <- function(color, amount = 0.5) {
  rgb_vals <- col2rgb(color) / 255
  rgb_soft <- rgb_vals + (1 - rgb_vals) * amount
  rgb_soft <- pmin(rgb_soft, 1) * 255
  rgb(rgb_soft[1,], rgb_soft[2,], rgb_soft[3,], maxColorValue = 255)
}
soft_colors <- sapply(colors, soften_color)


# --- 4. Load and Prepare Data ---
print(paste("Loading data from:", DATA_DIR))
file_paths <- list.files(DATA_DIR, full.names = TRUE, pattern = "^profit_distribution_.*\\.csv$")

if (length(file_paths) == 0) {
  stop("Data files not found. Please run '01_prepare_data_for_figures.py' first.")
}

# Read and combine all individual policy CSVs into one DataFrame
all_data <- map_dfr(file_paths, ~read_csv(.x, show_col_types = FALSE))

# Set the policy column as an ordered factor for correct plotting order
all_data$Policy <- factor(all_data$Policy, levels = rev(policy_shortnames))

# --- 5. Define the Plotting Function ---
generate_plot <- function(data, title_text, 
                          x_axis_label = expression(paste("Profit ($", "10"^"4", ")")), 
                          show_y_labels = TRUE) {
  ggplot(data, aes(x = average_profit, y = Policy, fill = Policy)) +
    geom_density_ridges(scale = 0.9, rel_min_height = 0.01, alpha = 0.6, color = NA) +
    geom_boxplot(width = 0.1, outlier.shape = NA, alpha = 0.4, color = "black", fill = "white") +
    labs(title = title_text, x = x_axis_label, y = NULL) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 20),
      axis.title.x = element_text(size = 16),
      axis.text.x = element_text(size = 14),
      axis.text.y = element_text(size = 14),
      legend.position = "none",
      panel.grid.major = element_line(linewidth = 0.5, linetype = 'dashed', color = "grey80"),
      panel.grid.minor = element_blank()
    ) +
    scale_fill_manual(values = soft_colors) +
    scale_x_continuous(
      breaks = scales::breaks_width(2),        # Sets the interval to 2
      labels = scales::comma_format(accuracy = 1), 
      limits = c(NA, NA)
    ) +
    # >>> keep identical y-levels across plots <<<
    scale_y_discrete(limits = rev(policy_shortnames), drop = FALSE) +
    {
      if (!show_y_labels) {
        theme(
          axis.text.y = element_text(color = "transparent"),
          axis.ticks.y = element_line(color = "transparent")
        )
      } else {
        NULL
      }
    }
}

# --- 6. Generate and Combine Plots ---

# First, calculate the average profit per farmer for each bootstrap run
avg_profit_data <- all_data %>%
  group_by(Policy, Bootstrap, AgentID, field_type_rn) %>%
  summarise(average_profit = mean(profit, na.rm = TRUE), .groups = "drop")

# Plot 1: Average profit of all farmers
# --- 6. Generate Plots ---
plot_all_farmers <- generate_plot(
  avg_profit_data,
  "(a) Average Yearly Profit Among\nAll Farmers"
)

# Plot 2: Average profit of irrigators only
data_irrigated <- avg_profit_data %>% filter(field_type_rn == "optimize")

plot_irrigators_only <- generate_plot(
  data_irrigated,
  "(b) Average Yearly Profit Among\nFarmers Equipped for Irrigation",
  show_y_labels = FALSE
)

# Combine the two plots side-by-side using patchwork
combined_plot <- (plot_all_farmers | plot_irrigators_only)

# --- 7. Save the Final Figure ---
save_path <- file.path(FIGURES_DIR, "profit_distribution.png")
ggsave(save_path, plot = combined_plot, width = 16.5, height = 8.3, dpi = 1000)

print(paste("Final figure saved to:", save_path))
print("-----------------------------------------")
print("Figure generation complete.")