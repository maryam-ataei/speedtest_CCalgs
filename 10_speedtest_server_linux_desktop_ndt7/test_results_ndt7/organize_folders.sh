#!/bin/bash

# Create destination folders
mkdir -p all_data/server/log
mkdir -p all_data/server/server_pcap
mkdir -p all_data/server/speedtest_server_pcap
mkdir -p all_data/client/speedtest_result
mkdir -p all_data/client/speedtest_client_pcap
mkdir -p all_data/client/client_pcap

# Loop index
index=1

# Sort folders and loop through them
for folder in $(ls -d test_* | sort -V); do
    [ -d "$folder" ] || continue

    # Extract all kernel logs and rename to .log
    for kfile in "$folder"/kernel_ccrg_log_*.txt; do
        [ -f "$kfile" ] || continue
        base=$(basename "$kfile" .txt)
        cp "$kfile" "all_data/server/log/${folder}_${base}.log"
    done
    
    # 2. tcpdump_iperf3_server.pcap → all_data/server/server_pcap/..._N.pcap
    if [ -f "$folder/tcpdump_iperf3_server.pcap" ]; then
        cp "$folder/tcpdump_iperf3_server.pcap" "all_data/server/server_pcap/tcpdump_iperf3_server_${index}.pcap"
    fi

    # 3. tcpdump_ndt7_server.pcap → all_data/server/speedtest_server_pcap/..._N.pcap
    if [ -f "$folder/tcpdump_ndt7_server.pcap" ]; then
        cp "$folder/tcpdump_ndt7_server.pcap" "all_data/server/speedtest_server_pcap/tcpdump_ndt7_server_${index}.pcap"
    fi

    # 4. ndt7_client_output.txt → all_data/client/speedtest_result/..._N.txt
    if [ -f "$folder/ndt7_client_output.txt" ]; then
        cp "$folder/ndt7_client_output.txt" "all_data/client/speedtest_result/ndt7_client_output_${index}.txt"
    fi

    # 5. tcpdump_ndt7_client.pcap → all_data/client/speedtest_client_pcap/..._N.pcap
    if [ -f "$folder/tcpdump_ndt7_client.pcap" ]; then
        cp "$folder/tcpdump_ndt7_client.pcap" "all_data/client/speedtest_client_pcap/tcpdump_ndt7_client_${index}.pcap"
    fi

    # 6. tcpdump_iperf3_client.pcap → all_data/client/client_pcap/..._N.pcap
    if [ -f "$folder/tcpdump_iperf3_client.pcap" ]; then
        cp "$folder/tcpdump_iperf3_client.pcap" "all_data/client/client_pcap/tcpdump_iperf3_client_${index}.pcap"
    fi

    # Increment counter
    index=$((index + 1))
done

