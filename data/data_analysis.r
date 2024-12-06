library(ggplot2)
library(dplyr)
library(tidyr)

# Load the data
data <- read.csv("data/sorted_run_table.csv")

# Convert necessary columns to appropriate data types
data$input_size <- as.factor(data$input_size)
data$task_type <- as.factor(data$task_type)
data$model_version <- as.factor(data$model_version)

# Data processing: Remove columns with all null values and clean numeric columns with empty or multiple values
data <- data %>%
  select_if(~ !all(is.na(.))) %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.) | grepl(",", .), NA, as.numeric(.)))) %>%
  drop_na()

# Convert GPU0_ENERGY from mJ to J
data <- data %>%
  mutate(GPU0_ENERGY..J. = GPU0_ENERGY..mJ. / 1000)

# Additional preprocessing: Remove rows with negative values in energy consumption columns
energy_columns <- c("DRAM_ENERGY..J.", "PACKAGE_ENERGY..J.", "GPU0_ENERGY..J.")
data <- data %>%
  filter(across(all_of(energy_columns), ~ . >= 0))

# Function to create plots for each model family
create_plots <- function(data, model_family) {
  filtered_data <- data %>% filter(grepl(paste0("^", model_family), model_version))
  
  # Debug print to check filtered data
  cat("Creating plots for model family:", model_family, "\n")
  cat("Number of rows in filtered data:", nrow(filtered_data), "\n")
  
  if (nrow(filtered_data) == 0) {
    cat("No data available for model family:", model_family, "\n")
    return()
  }
  
  # Plot for GPU utilization
  gpu_plot <- ggplot(filtered_data, aes(x = model_version, y = GPU0_USAGE, fill = task_type)) +
    geom_boxplot() +
    facet_wrap(~ input_size) +
    labs(title = paste("GPU Utilization for", model_family),
         x = "Model Version", y = "GPU Utilization (%)",
         fill = "Task Type") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Plot for CPU utilization
  cpu_plot <- ggplot(filtered_data %>% gather(key = "CPU", value = "Usage", starts_with("CPU_USAGE_")), 
                     aes(x = model_version, y = Usage, fill = task_type)) +
    geom_boxplot() +
    facet_wrap(~ input_size) +
    labs(title = paste("CPU Utilization for", model_family),
         x = "Model Version", y = "CPU Utilization (%)",
         fill = "Task Type") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Plot for VRAM usage
  vram_plot <- ggplot(filtered_data, aes(x = model_version, y = GPU0_MEMORY_USED, fill = task_type)) +
    geom_boxplot() +
    facet_wrap(~ input_size) +
    labs(title = paste("VRAM Usage for", model_family),
         x = "Model Version", y = "VRAM Usage (MB)",
         fill = "Task Type") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Plot for memory usage
  memory_plot <- ggplot(filtered_data %>% gather(key = "Energy_Type", value = "Energy", USED_MEMORY), 
                        aes(x = model_version, y = Energy, fill = task_type)) +
    geom_boxplot() +
    facet_wrap(~ input_size) +
    labs(title = paste("Memory Usage for", model_family),
         x = "Model Version", y = "Memory Usage (MB)",
         fill = "Task Type") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Plot for energy consumption
  energy_plot <- ggplot(filtered_data %>% gather(key = "Energy_Type", value = "Energy", DRAM_ENERGY..J., PACKAGE_ENERGY..J., GPU0_ENERGY..J.), 
                        aes(x = model_version, y = Energy, fill = task_type)) +
    geom_histogram(stat = "identity", position = "dodge") +
    facet_wrap(~ input_size) +
    labs(title = paste("Energy Consumption for", model_family),
         x = "Model Version", y = "Energy Consumption (J)",
         fill = "Task Type") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Combine plots into a single plot
  combined_plot <- gridExtra::grid.arrange(gpu_plot, cpu_plot, vram_plot, memory_plot, energy_plot, ncol = 2)
  
  # Save combined plot with higher resolution
  ggsave(paste("plots/", model_family, ".png", sep = ""), plot = combined_plot, dpi = 300, width = 16, height = 12)
}

# Create directories for plots
if (!dir.exists("plots")) {
  dir.create("plots")
}

# Extract unique model families using a more robust method
model_families <- unique(sub("([a-zA-Z]+).*", "\\1", data$model_version))

# Debug print to check unique model families
cat("Unique model families:", model_families, "\n")

# Generate plots for each model family
for (model_family in model_families) {
  create_plots(data, model_family)
}
