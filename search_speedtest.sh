#!/bin/bash
#
# This script automates repeated network performance tests using:
#  - `speedtest` (download only) with pcap capture
#  - `iperf3` reverse download test from a remote server
#
# Features:
#  - Uses a dynamic list of the top 5 nearby Speedtest servers (refreshed if needed)
#  - Tries each Speedtest server until one succeeds (fallback logic)
#  - Captures packets during each Speedtest with tcpdump on a given interface
#  - Logs output and errors to a specified file
#  - Repeats the test a specified number of times, with configurable sleep between runs
#
# Usage:
#   ./script.sh <server_IP> <iperf3_dwnd_time> <sleep_duration> <repeat> <output_file> <interface> <pcap_path>


display_usage() {
	echo "Usage: $0 <server_IP> <iperf3_dwnd_time> <sleep_duration> <repeat> <output_file> <interface> <pcap_path>"
	echo "Example: $0 130.215.28.249 45 300 5 speed_test.txt wlo1 pcap_dumps"
}

if [ "$#" -ne 7 ]; then
	display_usage
	exit 1
fi

SERVER_IP="$1"
IPERF3_DWND_TIME="$2"
SLEEP_DURATION="$3"
REPEAT="$4"
OUTPUT_FILE="$5"
INTERFACE="$6"
PCAP_DIR_PATH="$7"

PCAP_DIR="./${PCAP_DIR_PATH}"
mkdir -p "$PCAP_DIR"

# Declare global arrays
declare -a CACHED_SPEEDTEST_SERVERS
declare -A TEMP_FAILED_SERVERS

# Function to fetch a new top-5 server list
fetch_speedtest_servers() {
	readarray -t CACHED_SPEEDTEST_SERVERS < <(speedtest --list --secure | grep -Eo '^[ ]*[0-9]+' | head -n 5 | tr -d ' ')
	echo "Fetched new server list: ${CACHED_SPEEDTEST_SERVERS[*]}" | tee -a "$OUTPUT_FILE"
	# Reset failed server list
	TEMP_FAILED_SERVERS=()
}

# Fetch initial server list once
fetch_speedtest_servers

# === Main speedtest function ===
run_speedtest_with_fallback() {
	local output_file="$1"
	local pcap_file="$2"
	local success=0

	sudo tcpdump -i "$INTERFACE" tcp and port 8080 -s 200 -w "$pcap_file" 2>> "$output_file" &
	local tcpdump_pid=$!

	for attempt in {1..2}; do
		for server_id in "${CACHED_SPEEDTEST_SERVERS[@]}"; do
			if [[ -n "${TEMP_FAILED_SERVERS[$server_id]}" ]]; then
				echo "Skipping failed server $server_id" | tee -a "$output_file"
				continue
			fi

			echo "Trying server $server_id..." | tee -a "$output_file"

			tmp_output=$(mktemp)
			speedtest --no-upload --server "$server_id" --secure > "$tmp_output" 2>&1
			exit_code=$?
			cat "$tmp_output" | tee -a "$output_file"
			rm "$tmp_output"

			if [ "$exit_code" -eq 0 ]; then
				success=1
				break 2  # success, break out of both loops
			else
				echo "Server $server_id failed" | tee -a "$output_file"
				TEMP_FAILED_SERVERS["$server_id"]=1
			fi
		done

		# If all failed and it's the first attempt, refresh the list
		if [ "$success" -ne 1 ] && [ "$attempt" -eq 1 ]; then
			echo "All cached servers failed. Fetching a new list..." | tee -a "$output_file"
			fetch_speedtest_servers
		fi
	done

	sudo kill "$tcpdump_pid" >/dev/null 2>&1
	sudo pkill -f "tcpdump.*${INTERFACE}" >/dev/null 2>&1
	sleep 1

	if [ "$success" -ne 1 ]; then
		echo "All speedtest attempts failed." | tee -a "$output_file"
	fi
}

# === Main Loop ===
for i in $(seq 1 "$REPEAT"); do
	echo "====== Run $i ======" | tee -a "$OUTPUT_FILE"

	echo "1. Running pre-SEARCH speedtest-cli (download only with pcap)" | tee -a "$OUTPUT_FILE"
	PCAP_FILE="${PCAP_DIR}/download_pre_run_${i}.pcap"
	run_speedtest_with_fallback "$OUTPUT_FILE" "$PCAP_FILE"

	echo "2. Running iperf3 download test with SEARCH on server" | tee -a "$OUTPUT_FILE"
	iperf3 -c "$SERVER_IP" -t "$IPERF3_DWND_TIME" -R | tee -a "$OUTPUT_FILE"

	echo "3. Running post-SEARCH speedtest-cli (download only with pcap)" | tee -a "$OUTPUT_FILE"
	PCAP_FILE="${PCAP_DIR}/download_post_run_${i}.pcap"
	run_speedtest_with_fallback "$OUTPUT_FILE" "$PCAP_FILE"

	echo "4. Sleeping for $SLEEP_DURATION seconds..." | tee -a "$OUTPUT_FILE"
	sleep "$SLEEP_DURATION"
done
