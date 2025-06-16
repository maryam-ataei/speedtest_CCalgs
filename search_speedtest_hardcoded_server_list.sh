#!/bin/bash
#
# This script performs repeated network tests using a **fixed list of Speedtest server IDs**.
# Unlike the dynamic version, this version:
#   - Does NOT refresh the Speedtest server list
#   - Walks through a fixed set of 5 server IDs (hardcoded)
#   - Tries each in order for each Speedtest
#   - Skips Speedtest if all fail, but still proceeds with iperf3 and sleeps
#
# Usage:
#   ./script_fixed.sh <server_IP> <iperf3_dwnd_time> <sleep_duration> <repeat> <output_file> <interface> <pcap_path>

if [ "$#" -ne 7 ]; then
  echo "Usage: $0 <server_IP> <iperf3_dwnd_time> <sleep_duration> <repeat> <output_file> <interface> <pcap_path>"
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

# Fixed list of Speedtest server IDs
CACHED_SPEEDTEST_SERVERS=(4920 2405 13429 45728 27111)

run_speedtest_with_fixed_list() {
  local output_file="$1"
  local pcap_file="$2"
  local success=0

  sudo tcpdump -i "$INTERFACE" tcp and port 8080 -s 200 -w "$pcap_file" 2>> "$output_file" &
  local tcpdump_pid=$!

  for server_id in "${CACHED_SPEEDTEST_SERVERS[@]}"; do
    echo "Trying fixed server $server_id..." | tee -a "$output_file"

    tmp_output=$(mktemp)
    speedtest --no-upload --server "$server_id" --secure > "$tmp_output" 2>&1
    exit_code=$?
    cat "$tmp_output" | tee -a "$output_file"
    rm "$tmp_output"

    if [ "$exit_code" -eq 0 ]; then
      success=1
      break
    else
      echo "Fixed server $server_id failed" | tee -a "$output_file"
    fi
  done

  sudo kill "$tcpdump_pid" >/dev/null 2>&1
  sudo pkill -f "tcpdump.*${INTERFACE}" >/dev/null 2>&1
  sleep 1

  if [ "$success" -ne 1 ]; then
    echo "All fixed speedtest servers failed. Skipping this speedtest." | tee -a "$output_file"
  fi
}

# === Main Loop ===
for i in $(seq 1 "$REPEAT"); do
  echo "====== Run $i ======" | tee -a "$OUTPUT_FILE"

  echo "1. Running pre-SEARCH speedtest-cli (download only with pcap)" | tee -a "$OUTPUT_FILE"
  PCAP_FILE="${PCAP_DIR}/download_pre_run_${i}.pcap"
  run_speedtest_with_fixed_list "$OUTPUT_FILE" "$PCAP_FILE"

  echo "2. Running iperf3 download test with SEARCH on server" | tee -a "$OUTPUT_FILE"
  iperf3 -c "$SERVER_IP" -t "$IPERF3_DWND_TIME" -R | tee -a "$OUTPUT_FILE"

  echo "3. Running post-SEARCH speedtest-cli (download only with pcap)" | tee -a "$OUTPUT_FILE"
  PCAP_FILE="${PCAP_DIR}/download_post_run_${i}.pcap"
  run_speedtest_with_fixed_list "$OUTPUT_FILE" "$PCAP_FILE"

  echo "4. Sleeping for $SLEEP_DURATION seconds..." | tee -a "$OUTPUT_FILE"
  sleep "$SLEEP_DURATION"
done
