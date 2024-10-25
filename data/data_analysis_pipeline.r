# Function to install and load packages
install_and_load <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE)) {
      install.packages(pkg, dependencies = TRUE)
      library(pkg, character.only = TRUE)
    }
  }
}

# List of required packages
required_packages <- c('tidyverse', 'ggrepel')
install_and_load(required_packages)

# Load the data without converting strings to factors and prevent R from changing column names
data <- read.csv('run_table.csv', stringsAsFactors = FALSE, check.names = FALSE)

# Print number of rows after loading
cat("Number of rows after loading data:", nrow(data), "\n")

# View the first few rows of data to inspect columns
cat("First few rows of data:\n")
print(head(data))

# Columns available in the data
cat("Columns in the dataset:\n")
print(colnames(data))

# Rename columns to remove spaces and make them consistent
data <- data %>%
  rename(
    gpu_power = `GPU Power`,
    cpu_power = `CPU Power`,
    cpu_utilization = `CPU Utilization`,
    run_id = `__run_id`,
    done = `__done`
  )

# Clean numeric columns with empty or multiple values
clean_numeric_column <- function(x) {
  sapply(x, function(value) {
    # Check for empty or missing values
    if (is.na(value) || value == "" || value == "[]" || value == "NA") {
      return(NA)
    } else {
      # Remove square brackets and any non-numeric characters except commas, periods, and minus signs
      value <- gsub("[\\[\\]]", "", value)
      value <- gsub("[^0-9.,-]", "", value)
      # Split on commas or spaces
      value_split <- unlist(strsplit(value, "[, ]+"))
      # Trim whitespace
      value_split <- trimws(value_split)
      # Convert to numeric and remove NAs
      numeric_values <- as.numeric(value_split)
      numeric_values <- numeric_values[!is.na(numeric_values)]
      # Return the mean
      if (length(numeric_values) > 0) {
        return(mean(numeric_values))
      } else {
        return(NA)
      }
    }
  })
}

# Apply the cleaning functions to the relevant columns
data$gpu_utilization <- clean_numeric_column(data$gpu_utilization)
data$vram_usage <- clean_numeric_column(data$vram_usage)
data$gpu_power <- clean_numeric_column(data$gpu_power)
data$cpu_power <- clean_numeric_column(data$cpu_power)
data$cpu_utilization <- clean_numeric_column(data$cpu_utilization)
data$energy_consumption <- as.numeric(as.character(data$energy_consumption))

# View cleaned data
cat("Cleaned data summary:\n")
print(summary(data))

# Handle NAs in 'energy_consumption' if necessary
# data$energy_consumption[is.na(data$energy_consumption)] <- 0

# Convert 'task_type', 'model_version', and 'input_size' to factors
data$task_type <- as.factor(data$task_type)
data$model_version <- as.factor(data$model_version)
data$input_size <- as.factor(data$input_size)

# Descriptive statistics
descriptive_stats <- data %>%
  group_by(model_version, task_type, input_size) %>%
  summarise(
    energy_consumption_mean = mean(energy_consumption, na.rm = TRUE),
    gpu_utilization_mean = mean(gpu_utilization, na.rm = TRUE),
    vram_usage_mean = mean(vram_usage, na.rm = TRUE),
    gpu_power_mean = mean(gpu_power, na.rm = TRUE),
    cpu_power_mean = mean(cpu_power, na.rm = TRUE),
    cpu_utilization_mean = mean(cpu_utilization, na.rm = TRUE),
    .groups = 'drop'
  )

# Print descriptive statistics
cat("Descriptive statistics:\n")
print(descriptive_stats)

# Proceed to generate plots only if data is not empty
if (nrow(data) > 0) {

  # Filter data for plotting to handle NAs in specific columns
  data_gpu <- data %>% drop_na(gpu_utilization)
  data_vram <- data %>% drop_na(vram_usage)
  data_gpu_power <- data %>% drop_na(gpu_power)
  data_cpu_power <- data %>% drop_na(cpu_power)
  data_cpu_utilization <- data %>% drop_na(cpu_utilization)
  data_energy <- data %>% drop_na(energy_consumption)

  # Check number of rows in each dataset
  cat("Number of rows in data_gpu after cleaning:", nrow(data_gpu), "\n")
  cat("Number of rows in data_vram after cleaning:", nrow(data_vram), "\n")
  cat("Number of rows in data_energy after cleaning:", nrow(data_energy), "\n")

  # Bar Plot for Energy Consumption Across LLM Versions
  energy_consumption_plot <- ggplot(descriptive_stats, aes(x = model_version, y = energy_consumption_mean, fill = task_type)) +
    geom_bar(stat = "identity", position = position_dodge()) +
    facet_wrap(~ input_size) +
    labs(
      title = "Average Energy Consumption by Model Version, Task Type, and Input Size",
      x = "Model Version",
      y = "Average Energy Consumption"
    ) +
    theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5),
      legend.position = "bottom"
    )

  # Save the plot
  ggsave('energy_consumption_plot.png', plot = energy_consumption_plot, width = 12, height = 6, dpi = 300)

  # Plot for GPU Utilization
  if (nrow(data_gpu) > 0) {
    gpu_utilization_plot <- ggplot(data_gpu, aes(x = model_version, y = gpu_utilization, color = task_type)) +
      geom_jitter(width = 0.2, size = 2, alpha = 0.7) +
      facet_wrap(~ input_size) +
      labs(
        title = "GPU Utilization by Model Version, Task Type, and Input Size",
        x = "Model Version",
        y = "GPU Utilization (%)"
      ) +
      theme_bw() +
      theme(
        plot.title = element_text(hjust = 0.5),
        legend.position = "bottom"
      )

    # Save the plot
    ggsave('gpu_utilization_plot.png', plot = gpu_utilization_plot, width = 12, height = 6, dpi = 300)
  } else {
    cat("No data available for GPU Utilization plot after cleaning.\n")
  }

  # Plot for VRAM Usage
  if (nrow(data_vram) > 0) {
    vram_usage_plot <- ggplot(data_vram, aes(x = model_version, y = vram_usage, fill = task_type)) +
      geom_boxplot() +
      facet_wrap(~ input_size) +
      labs(
        title = "VRAM Usage by Model Version, Task Type, and Input Size",
        x = "Model Version",
        y = "VRAM Usage"
      ) +
      theme_bw() +
      theme(
        plot.title = element_text(hjust = 0.5),
        legend.position = "bottom"
      )

    # Save the plot
    ggsave('vram_usage_plot.png', plot = vram_usage_plot, width = 12, height = 6, dpi = 300)
  } else {
    cat("No data available for VRAM Usage plot after cleaning.\n")
  }

  # Plot for GPU Power Consumption
  if (nrow(data_gpu_power) > 0) {
    gpu_power_plot <- ggplot(data_gpu_power, aes(x = model_version, y = gpu_power, fill = task_type)) +
      geom_boxplot() +
      facet_wrap(~ input_size) +
      labs(
        title = "GPU Power Consumption by Model Version, Task Type, and Input Size",
        x = "Model Version",
        y = "GPU Power (Watts)"
      ) +
      theme_bw() +
      theme(
        plot.title = element_text(hjust = 0.5),
        legend.position = "bottom"
      )

    # Save the plot
    ggsave('gpu_power_plot.png', plot = gpu_power_plot, width = 12, height = 6, dpi = 300)
  } else {
    cat("No data available for GPU Power Consumption plot after cleaning.\n")
  }

  # Plot for CPU Power Consumption
  if (nrow(data_cpu_power) > 0) {
    cpu_power_plot <- ggplot(data_cpu_power, aes(x = model_version, y = cpu_power, fill = task_type)) +
      geom_boxplot() +
      facet_wrap(~ input_size) +
      labs(
        title = "CPU Power Consumption by Model Version, Task Type, and Input Size",
        x = "Model Version",
        y = "CPU Power (Watts)"
      ) +
      theme_bw() +
      theme(
        plot.title = element_text(hjust = 0.5),
        legend.position = "bottom"
      )

    # Save the plot
    ggsave('cpu_power_plot.png', plot = cpu_power_plot, width = 12, height = 6, dpi = 300)
  } else {
    cat("No data available for CPU Power Consumption plot after cleaning.\n")
  }

  # Plot for CPU Utilization
  if (nrow(data_cpu_utilization) > 0) {
    cpu_utilization_plot <- ggplot(data_cpu_utilization, aes(x = model_version, y = cpu_utilization, color = task_type)) +
      geom_jitter(width = 0.2, size = 2, alpha = 0.7) +
      facet_wrap(~ input_size) +
      labs(
        title = "CPU Utilization by Model Version, Task Type, and Input Size",
        x = "Model Version",
        y = "CPU Utilization (%)"
      ) +
      theme_bw() +
      theme(
        plot.title = element_text(hjust = 0.5),
        legend.position = "bottom"
      )

    # Save the plot
    ggsave('cpu_utilization_plot.png', plot = cpu_utilization_plot, width = 12, height = 6, dpi = 300)
  } else {
    cat("No data available for CPU Utilization plot after cleaning.\n")
  }

  # Correlation Between GPU Power and Energy Consumption
  if (nrow(data_gpu_power) > 0 && nrow(data_energy) > 0) {
    correlation_plot <- ggplot(data_energy, aes(x = gpu_power, y = energy_consumption, color = model_version)) +
      geom_point(size = 3, alpha = 0.7) +
      facet_wrap(~ task_type) +
      labs(
        title = "Correlation Between GPU Power and Energy Consumption",
        x = "GPU Power (Watts)",
        y = "Energy Consumption"
      ) +
      theme_bw() +
      theme(
        plot.title = element_text(hjust = 0.5),
        legend.position = "right"
      )

    # Save the plot
    ggsave('gpu_power_energy_consumption_correlation.png', plot = correlation_plot, width = 12, height = 6, dpi = 300)
  } else {
    cat("Insufficient data for correlation plot.\n")
  }

} else {
  cat("No data available for plotting after cleaning.\n")
}
