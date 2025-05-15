import re
import pandas as pd
import matplotlib.pyplot as plt
import os

output_path = "/home/maryam/SEARCH/speedtest_CCalgs/results"
input_path = "/home/maryam/SEARCH/speedtest_CCalgs/speedtest_search_over_homewifi.txt"
if not os.path.exists(output_path):
    os.makedirs(output_path)

def extract_speedtest_values(block):
    ping = re.search(r'Ping:\s+([\d.]+)', block)
    down = re.search(r'Download:\s+([\d.]+)', block)
    up = re.search(r'Upload:\s+([\d.]+)', block)
    return (
        float(ping.group(1)) if ping else None,
        float(down.group(1)) if down else None,
        float(up.group(1)) if up else None
    )

def extract_iperf3_throughput(block):
    match = re.search(r'([0-9.]+)\s+Mbits/sec\s+receiver', block)
    return float(match.group(1)) if match else None

with open(input_path) as f:
    content = f.read()

runs = content.split("====== Run")[1:]  # Skip header part
data = []

for run in runs:
    run_number = int(run.split()[0])
    parts = run.split("Running")

    pre_speed = extract_speedtest_values(parts[1]) if len(parts) > 1 else (None, None, None)
    iperf = extract_iperf3_throughput(parts[2]) if len(parts) > 2 else None
    post_speed = extract_speedtest_values(parts[3]) if len(parts) > 3 else (None, None, None)

    data.append({
        "Run": run_number,
        "Pre Ping": pre_speed[0],
        "Pre Download": pre_speed[1],
        "Pre Upload": pre_speed[2],
        "iperf3 Throughput": iperf,
        "Post Ping": post_speed[0],
        "Post Download": post_speed[1],
        "Post Upload": post_speed[2],
    })

df = pd.DataFrame(data)
print(df)

# Save table to CSV
csv_path = os.path.join(output_path, "speedtest_results.csv")
df.to_csv(csv_path, index=False)

# Plot
plt.figure()
df.plot(x="Run", y=["Pre Download", "Post Download", "iperf3 Throughput"], marker='o')
plt.ylabel("Mbps")
plt.title("Download & Throughput Over Runs")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "download_throughput_over_runs.png"))
plt.close()

