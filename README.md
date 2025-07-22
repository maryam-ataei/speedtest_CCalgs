# speedtest Congestion Control Algorithms

## 📂 Scripts and Tools Overview

This repository contains a suite of scripts and automation tools designed to evaluate TCP congestion control (especially SEARCH) using popular bandwidth testing tools and packet-level analysis.

### 🔧 Shell Scripts

| Script Name                                   | Description |
|----------------------------------------------|-------------|
| `search_speedtest.sh`                        | Automates Speedtest (Ookla) and iperf3 tests using the top 5 nearby Speedtest servers. Captures packets with `tcpdump` and logs outputs. |
| `search_speedtest_hardcoded_server_list.sh`  | Like `search_speedtest.sh`, but uses a hardcoded list of 5 Speedtest server IDs. Ideal for fixed-server experiments. |
| `search_speedtest_with_ndt7.sh`              | Combines `ndt7-client` (upload only) with `iperf3` to test performance before and after SEARCH. Includes pcap capture. |

#### 🧪 Usage (Shell Scripts)
```bash
./search_speedtest.sh <server_IP> <iperf3_dwnd_time> <sleep_duration> <repeat> <output_file> <interface> <pcap_path>
```

#### 📌 Command Line Arguments

| Argument             | Description                                                  |
|----------------------|--------------------------------------------------------------|
| `<server_IP>`        | IP address of the remote iperf3 server (with SEARCH enabled) |
| `<iperf3_dwnd_time>` | Duration (in seconds) of the iperf3 download test            |
| `<sleep_duration>`   | Time (in seconds) to sleep between each full run             |
| `<repeat>`           | Number of times to repeat the full test sequence             |
| `<output_file>`      | Log file where outputs and errors will be saved              |
| `<interface>`        | Network interface for packet capture (e.g., `wlo1`, `eth0`)  |
| `<pcap_path>`        | Directory to save `.pcap` files for pre- and post-tests      |

---

### 🐍 Python Automation Scripts

| Script Name                    | Description |
|--------------------------------|-------------|
| `launch_speedtest_apps_iperf3.py` | Automates tests across Ookla Speedtest, NDT7, LibreSpeed, and iperf3. Captures `.pcap` files, kernel logs, and test outputs for each. Used in the final evaluation of multiple tools across networks. |
| `launch_ndt7_iperf3.py`           | Automates upload-only `ndt7` tests with packet capture, followed by iperf3 tests to observe SEARCH behavior. |
| `launch_libre_iperf3.py`         | Runs LibreSpeed (download-only) tests with `.pcap` collection and kernel module logging, followed by iperf3 tests using SEARCH. |

---
