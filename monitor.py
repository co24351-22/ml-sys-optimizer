import psutil
import time
import csv
from datetime import datetime

file_name = "system_metrics.csv"

# Create file with header (only once)
try:
    with open(file_name, 'x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "CPU_Usage", "Memory_Usage", "Disk_Usage", "Process_Count"])
except FileExistsError:
    pass

print("Monitoring started... Press Ctrl+C to stop.\n")

try:
    while True:
        # Collect system metrics
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        processes = len(psutil.pids())

        # Current time
        current_time = datetime.now().strftime("%H:%M:%S")

        # Print output
        print(f"[{current_time}] CPU: {cpu}% | Memory: {memory}% | Disk: {disk}% | Processes: {processes}")

        # Append data to CSV (NO DUPLICATION)
        with open(file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([current_time, cpu, memory, disk, processes])

        # Wait 2 seconds
        time.sleep(2)

except KeyboardInterrupt:
    print("\nMonitoring stopped. Data saved to system_metrics.csv")
