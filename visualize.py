import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("system_metrics.csv")

# Show only recent data
df = df.tail(50)

# Convert time
df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S")

# CPU Graph
plt.figure()
plt.plot(df["Time"], df["CPU_Usage"])
plt.title("CPU Usage (Recent)")
plt.xlabel("Time")
plt.ylabel("CPU (%)")
plt.xticks(rotation=45)

# Memory Graph
plt.figure()
plt.plot(df["Time"], df["Memory_Usage"])
plt.title("Memory Usage (Recent)")
plt.xlabel("Time")
plt.ylabel("Memory (%)")
plt.xticks(rotation=45)

plt.show()
