#! /bin/bash

# Display usage message
display_usage() {
	echo "Usage: $0 <server_IP> <iperf3_dwnd_time> <sleep_duration> <repeat> <output_file>"
	echo "Example: $0 130.215.28.249 45 300 5 speed_test_over_viasat.txt" 
}

# check for correct num of argument
if [ "$#" -ne 5 ]; then
	display_usage
	exit 1
fi

# Assign input values to variables
SERVER_IP="$1"
IPERF3_DWND_TIME="$2"
SLEEP_DURATION="$3"
REPEAT="$4"
OUTPUT_FILE="$5"

echo "====== Full speedtest for context ======" | tee "$OUTPUT_FILE"
speedtest-cli | tee -a "$OUTPUT_FILE"


for i in $(seq 1 "$REPEAT"); do

	echo "====== Run $i ======" | tee -a "$OUTPUT_FILE"
	echo "1. Running pre-SEARCH speedtest-cli" | tee -a "$OUTPUT_FILE"
	speedtest-cli --simple | tee -a "$OUTPUT_FILE"
	
	echo "2. Running iperf3 download test with SEARCH on server" | tee -a "$OUTPUT_FILE"
	iperf3 -c "$SERVER_IP" -t "$IPERF3_DWND_TIME" -R | tee -a "$OUTPUT_FILE"
	
	echo "3. Running post-SEARCH speedtest-cli" | tee -a "$OUTPUT_FILE"
  	speedtest-cli --simple | tee -a "$OUTPUT_FILE"
  	
  	echo "4. Sleeping for $SLEEP_DURATION seconds..." | tee -a "$OUTPUT_FILE"
  	sleep "$SLEEP_DURATION"
done
