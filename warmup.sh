# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <duration_in_seconds>"
    exit 1
fi

# Parse the duration argument
DURATION="$1"
if ! [[ "$DURATION" =~ ^[0-9]+$ ]]; then
    echo "Error: Duration must be a positive integer."
    exit 1
fi

# Get the number of CPU cores dynamically
CPU_CORES=$(nproc --all 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null)
if [ -z "$CPU_CORES" ]; then
    echo "Unable to determine the number of CPU cores."
    exit 1
fi
echo "Detected $CPU_CORES CPU cores."
echo "Warm-up duration: $DURATION seconds."

# Start stress-ng
echo "Starting CPU warm-up using stress-ng..."
stress-ng --cpu "$CPU_CORES" --timeout "$DURATION"s &
STRESS_NG_PID=$!

# Start nvidia-smi in parallel
echo "Starting GPU warm-up using nvidia-smi..."
for i in $(seq 1 "$CPU_CORES"); do
    nvidia-smi -l 1 >/dev/null 2>&1 &
done

# Wait for the specified duration
echo "Warming up for $DURATION seconds..."
sleep "$DURATION"

# Cleanup processes
cleanup

echo "Warm-up completed!"
