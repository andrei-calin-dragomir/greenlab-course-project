#!/bin/bash

# Function to check if a model is already installed
is_model_installed() {
    local model_name="$1"
    if echo "$INSTALLED_MODELS" | grep -qw "^$model_name$"; then
        return 0  # Model is installed
    else
        return 1  # Model is not installed
    fi
}

# Validate input argument
if [ -z "$1" ]; then
    echo "Usage: $0 model1:version,model2:version,model3:version,..."
    exit 1
fi

# Fetch the list of installed models
INSTALLED_MODELS=$(ollama list 2>/dev/null | awk '{print $1}')
if [ -z "$INSTALLED_MODELS" ]; then
    echo "No models installed currently or unable to fetch installed models list."
else
    echo "Currently installed models:"
    echo "$INSTALLED_MODELS"
    echo
fi

# Parse and clean the input models list
INPUT_MODELS=$(echo "$1" | tr ',' '\n' | sort -u)

# Pull models that are not already installed
for model in $INPUT_MODELS; do
    if is_model_installed "$model"; then
        echo "Model '$model' is already installed. Skipping."
    else
        echo "Pulling model '$model'..."
        ollama pull "$model"
        if [ $? -eq 0 ]; then
            echo "Successfully pulled '$model'."
        else
            echo "Failed to pull '$model'."
        fi
    fi
done

echo "Model installation process completed!"
