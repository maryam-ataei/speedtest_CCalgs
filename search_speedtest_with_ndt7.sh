#!/bin/bash

# Display usage message
display_usage() {
	echo "Usage: $0 <server_IP> <iperf3_dwnd_time> <sleep_duration> <repeat> <output_file> <speedtest_server> <interface>"
	echo "Example: $0 130.215.28.249 45 300 5 speed_test_over_viasat.txt 16614 wlo1" 
}

# Check for correct num of arguments
if [ "$#" -ne 7 ]; then
	display_usage
	exit 1
fi

SERVER_IP="$1"
IPERF3_DWND_TIME="$2"
SLEEP_DURATION="$3"
REPEAT="$4"
OUTPUT_FILE="$5"
SPEEDTEST_SERVER="$6"
INTERFACE="$7"

PCAP_DIR="./pcap_dumps_ndt7"
mkdir -p "$PCAP_DIR"

echo "====== Full speedtest for context ======" | tee "$OUTPUT_FILE"
ndt7-client | tee -a "$OUTPUT_FILE"

for i in $(seq 1 "$REPEAT"); do
	echo "====== Run $i ======" | tee -a "$OUTPUT_FILE"

	# ---------- Pre-SEARCH Upload ----------
	echo "1. Running pre-SEARCH speedtest-cli (upload only with pcap)" | tee -a "$OUTPUT_FILE"
	PCAP_FILE="${PCAP_DIR}/upload_pre_run_${i}.pcap"
	sudo tcpdump -i "$INTERFACE" -w "$PCAP_FILE" &
	TCPDUMP_PID=$!
	ndt7-client | tee -a "$OUTPUT_FILE"
	sudo kill "$TCPDUMP_PID"
	sleep 1

	# ---------- iPerf Test ----------
	echo "2. Running iperf3 download test with SEARCH on server" | tee -a "$OUTPUT_FILE"
	iperf3 -c "$SERVER_IP" -t "$IPERF3_DWND_TIME" -R | tee -a "$OUTPUT_FILE"

	# ---------- Post-SEARCH Upload ----------
	echo "3. Running post-SEARCH speedtest (upload only with pcap)" | tee -a "$OUTPUT_FILE"
	PCAP_FILE="${PCAP_DIR}/upload_post_run_${i}.pcap"
	sudo tcpdump -i "$INTERFACE" -w "$PCAP_FILE" &
	TCPDUMP_PID=$!
	ndt7-client | tee -a "$OUTPUT_FILE"
	sudo kill "$TCPDUMP_PID"
	sleep 1

	echo "4. Sleeping for $SLEEP_DURATION seconds..." | tee -a "$OUTPUT_FILE"
	sleep "$SLEEP_DURATION"
done
