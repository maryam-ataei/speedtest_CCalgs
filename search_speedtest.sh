#!/bin/bash

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

# Dynamically build initial fallback server list (top 5)
readarray -t SPEEDTEST_SERVERS < <(speedtest --list --secure | grep -Eo '^[ ]*[0-9]+' | head -n 5 | tr -d ' ')
echo "Initial Speedtest server list: ${SPEEDTEST_SERVERS[*]}" | tee -a "$OUTPUT_FILE"

run_speedtest_with_fallback() {
	local output_file="$1"
	local pcap_file="$2"
	local tried_servers=()
	local success=0

	sudo tcpdump -i "$INTERFACE" tcp and port 8080 -w "$pcap_file" &
	local tcpdump_pid=$!

	# Try initial list
	for server_id in "${SPEEDTEST_SERVERS[@]}"; do
		echo "Trying fallback server $server_id..." | tee -a "$output_file"
		speedtest --no-upload --server "$server_id" --secure | tee -a "$output_file"
		tried_servers+=("$server_id")
		if [ "${PIPESTATUS[0]}" -eq 0 ]; then
			success=1
			break
		else
			echo "Server $server_id failed, trying next..." | tee -a "$output_file"
		fi
	done

	# If all fail, try new dynamic list (excluding tried ones)
	if [ "$success" -ne 1 ]; then
		echo "All fallback servers failed. Fetching additional servers..." | tee -a "$output_file"
		readarray -t new_servers < <(speedtest --list --secure | grep -Eo '^[ ]*[0-9]+' | head -n 15 | tr -d ' ')
		for new_id in "${new_servers[@]}"; do
			if [[ " ${tried_servers[*]} " =~ " ${new_id} " ]]; then
				continue
			fi
			echo "Trying dynamic server $new_id..." | tee -a "$output_file"
			speedtest --no-upload --server "$new_id" --secure | tee -a "$output_file"
			if [ "${PIPESTATUS[0]}" -eq 0 ]; then
				success=1
				break
			else
				echo "Server $new_id also failed." | tee -a "$output_file"
			fi
		done
	fi

	sudo kill "$tcpdump_pid"
	sudo pkill tcpdump
	sleep 1

	if [ "$success" -ne 1 ]; then
		echo "All speedtest servers failed (initial + dynamic)." | tee -a "$output_file"
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
