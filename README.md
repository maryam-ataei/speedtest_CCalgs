# speedtest Congestion Control Algorithms

1. search_speedtest.sh
- Purpose: Automates download-only Speedtest runs and iperf3 tests using a dynamic list of the top 5 nearby Speedtest servers.

- Features:

  --Uses fallback logic across servers.

 --Captures packets with tcpdump during each Speedtest.

 --Repeats tests and logs output.

- Usage: Requires 7 arguments: <server_IP> <iperf3_dwnd_time> <sleep_duration> <repeat> <output_file> <interface> <pcap_path>

2. search_speedtest_hardcoded_server_list.sh
Purpose: Similar to the above script but uses a hardcoded list of 5 Speedtest server IDs.

Difference: Does not refresh server list dynamically. It retries the same fixed servers each run.

Use case: Useful for controlled experiments with fixed server conditions.

Usage: ./search_speedtest_hardcoded_server_list.sh <server_IP> <iperf3_dwnd_time> <sleep_duration> <repeat> <output_file> <interface> <pcap_path>

3. search_speedtest_with_ndt7.sh
Purpose: Combines ndt7-client (upload tests) with iperf3 to evaluate performance before and after SEARCH.

Features:

Captures .pcap files during ndt7 tests.

Performs repeated runs.

Usage: <server_IP> <iperf3_dwnd_time> <sleep_duration> <repeat> <output_file> <speedtest_server> <interface>

4. speedtest_analysis.py
Purpose: Analyzes results from Speedtest, NDT7, and iperf3 test runs.

Key functions:

Parses speedtest logs and calculates metrics like ping, download/upload speeds.

Extracts throughput from pcap and CSV logs.

Calculates delivery rate per ACK or over fixed intervals.

Identifies SEARCH exit times based on server logs.

5. speedtest_analysis_all_togother.py
Purpose: Comprehensive analysis combining Ookla, NDT7, LibreSpeed, and SEARCH metrics.

Highlights:

Parses logs and pcap files from all tools.

Computes throughput, duration, bytes sent, and exit rates.

Generates comparative graphs (e.g., error vs. duration, error vs. bytes sent) across different networks (4G, Cable, GEO).

Implements advanced analysis like PCA-based grouping.

Used in: Final paper-level evaluation across multiple tools and networks.
