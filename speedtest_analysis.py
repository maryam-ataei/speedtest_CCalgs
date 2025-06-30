import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import bisect
import numpy as np

input_path = "/home/maryam/SEARCH/speedtest_CCalgs/9_speedtest_homecable_24hrs/client/speedtest_homecable_24hrs.txt"
output_path = "/home/maryam/SEARCH/speedtest_CCalgs/9_speedtest_homecable_24hrs/result"
if not os.path.exists(output_path):
    os.makedirs(output_path)

server_log_path = "/home/maryam/SEARCH/speedtest_CCalgs/9_speedtest_homecable_24hrs/server/data/log_search"
server_pcap_path = "/home/maryam/SEARCH/speedtest_CCalgs/9_speedtest_homecable_24hrs/server/data/pcap_server"

client_pcap_path = "/home/maryam/SEARCH/speedtest_CCalgs/9_speedtest_homecable_24hrs/client/pcap"



TEST_WITH_SPEEDTEST = True
TEST_WITH_NDT7 = False

CALC_THROUGHPUT = True

#################################### Functions ####################################
# Extract speedtest values from a block of text
def extract_speedtest_values(block):
    ping = re.search(r'Ping:\s+([\d.]+)', block)
    down = re.search(r'Download:\s+([\d.]+)', block)
    up = re.search(r'Upload:\s+([\d.]+)', block)
    return (
        float(ping.group(1)) if ping else None,
        float(down.group(1)) if down else None,
        float(up.group(1)) if up else None
    )
# Extract speedtest values from a block of text
def extract_speedtest_values_ndt7(block):
    down_speed = None
    up_speed = None
    current_section = None

    for line in block.splitlines():
        line = line.strip()
        if "Download" in line:
            current_section = "download"
        elif "Upload" in line:
            current_section = "upload"
        elif line.startswith("Throughput:"):
            match = re.search(r'Throughput:\s+([\d.]+)', line)
            if match:
                speed = float(match.group(1))
                if current_section == "download":
                    down_speed = speed
                elif current_section == "upload":
                    up_speed = speed

    return down_speed, up_speed

# Extract iperf3 throughput from a block of text
def extract_iperf3_throughput(block):
    match = re.search(r'([0-9.]+)\s+Mbits/sec\s+receiver', block)
    return float(match.group(1)) if match else None

# Calculate delivery rate per ACK
def calculate_delivery_rate_per_ack(bytes_acked, now, rtt):
    delivery_rates = []
    time_cal_delv_rates = []
    start_index = 0

    for i, (t_now, r) in enumerate(zip(now, rtt)):
        target_time = t_now - r
        if target_time <= 0:
            start_index = i + 1
            continue

        j = bisect.bisect_right(now, target_time, 0, i) - 1
        if j < 0 or j + 1 >= len(now):
            start_index = i + 1
            continue

        # Linear interpolation between j and j+1
        t0, t1 = now[j], now[j + 1]
        b0, b1 = bytes_acked[j], bytes_acked[j + 1]

        if t1 == t0:
            bytes_acked_at_target = b0  # fallback if timestamps duplicated
        else:
            frac = (target_time - t0) / (t1 - t0)
            bytes_acked_at_target = b0 + frac * (b1 - b0)

        # Calculate rate using difference
        delta_bytes = bytes_acked[i] - bytes_acked_at_target
        delivery_rate = delta_bytes / r
        delivery_rates.append(delivery_rate)
        time_cal_delv_rates.append(t_now)

    return delivery_rates, start_index, time_cal_delv_rates if delivery_rates else None

# Calculate CDF from a list of data
def calculate_cdf(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    cdf = [(i + 1) / n for i in range(n)]
    return sorted_data, cdf

# Calculate delivery rate per fixed-length time intewrval (based on initial rtt)
def calculate_delivery_rate_by_interval(bytes_acked, now, rtt):

    if not bytes_acked or not now or not rtt:
        return [], []

    interval_len = rtt[0]
    interval_start = now[0]
    interval_end = interval_start + interval_len

    interval_rates = []
    interval_times = []

    i = 0
    while interval_end <= now[-1]:
        # Find indices within this interval
        start_bytes = None
        end_bytes = None

        # Scan forward to cover the interval
        while i < len(now) and now[i] < interval_end:
            if start_bytes is None and now[i] >= interval_start:
                start_bytes = bytes_acked[i]
            end_bytes = bytes_acked[i]
            i += 1

        if start_bytes is not None and end_bytes is not None:
            delta_bytes = end_bytes - start_bytes
            delivery_rate = delta_bytes / interval_len
            interval_rates.append(delivery_rate)
            interval_times.append(interval_end)
        
        # Advance interval
        interval_start = interval_end
        interval_end += interval_len

    return interval_rates, interval_times

# Calculate delivery rate aat exit
def rate_at_exit(exit_time, times, rates):  
    
    # ----- edge cases -------------------------------------------------------
    if exit_time <= times[0]:
        return rates[0]                     # or None if you’d rather skip
    if exit_time >= times[-1]:
        return rates[-1]

    # ----- locate the bracketing indices ------------------------------------
    # idx is the first index whose time >= exit_time
    idx = bisect.bisect_left(times, exit_time)

    t0, t1 = times[idx - 1], times[idx]
    r0, r1 = rates[idx - 1], rates[idx]

    # ----- linear interpolation --------------------------------------------
    frac = (exit_time - t0) / (t1 - t0)     # in (0,1]
    delivery_rate_at_exit = r0 + frac * (r1 - r0)
    return delivery_rate_at_exit

# Calculate delivery rate aat exit
def total_byte_acked_at_exit(exit_time, times, total_bytes):  
    
    # ----- edge cases -------------------------------------------------------
    if exit_time <= times[0]:
        return total_bytes[0]                     # or None if you’d rather skip
    if exit_time >= times[-1]:
        return total_bytes[-1]

    # ----- locate the bracketing indices ------------------------------------
    # idx is the first index whose time >= exit_time
    idx = bisect.bisect_left(times, exit_time)

    t0, t1 = times[idx - 1], times[idx]
    b0, b1 = total_bytes[idx - 1], total_bytes[idx]

    # ----- linear interpolation --------------------------------------------
    frac = (exit_time - t0) / (t1 - t0)     # in (0,1]
    total_bytes_acked_at_exit = b0 + frac * (b1 - b0)
    return total_bytes_acked_at_exit
############################################################
# calculate delivery rate based on log files
avg_delivery_rates = []
median_delivery_rates = []
delivery_rate_at_exit = []
search_exit_time_list = []
not_search_exit_time_list = []
all_ss_exit_time_list = []
total_bits_acked_at_exit_list = []
exit_time = None
skip_run_indices = set()

if not os.path.exists(server_log_path):
    print(f"Server log path {server_log_path} does not exist")
else:
    log_csv_files = [f for f in os.listdir(server_log_path) if f.endswith('.csv')]
    if not log_csv_files:
        print(f"No CSV files found in {server_log_path}")
        exit()  
    num_log_files = len(log_csv_files)  # Get the last file

    for j in range(num_log_files+1):
        log_csv_file_path = os.path.join(server_log_path, f"log_data{j+1}.csv")
        if not os.path.exists(log_csv_file_path):
            print(f"File {log_csv_file_path} does not exist")
            continue

        df_log = pd.read_csv(log_csv_file_path)

        # Extract bytes acked and now
        bytes_acked_list = df_log["total_byte_acked"].tolist()
        now_list = df_log["start_time_zero_s"].tolist()
        rtt_s_list = df_log["rtt_s"].tolist()
        search_exit_time = df_log["search_ex_time_s"].tolist()
        sstresh_list = df_log["ssthresh_pkt"].tolist()

        # if we have negative value in now_list, we limit the list to positive values
        bytes_acked_list = [b for b, n in zip(bytes_acked_list, now_list) if n >= 0]
        now_list = [n for n in now_list if n >= 0]
        rtt_s_list = [r for r, n in zip(rtt_s_list, now_list) if n >= 0]
        search_exit_time = [s for s, n in zip(search_exit_time, now_list) if n >= 0]
        sstresh_list = [s for s, n in zip(sstresh_list, now_list) if n >= 0]
        
        positive_indices = [i for i, n in enumerate(now_list) if n >= 0]
        now_list          = [now_list[i] for i in positive_indices]
        bytes_acked_list  = [bytes_acked_list[i] for i in positive_indices]
        rtt_s_list        = [rtt_s_list[i] for i in positive_indices]
        sstresh_list      = [sstresh_list[i] for i in positive_indices]

        if len(bytes_acked_list) == 0 or len(now_list) == 0 or len(rtt_s_list) == 0:
            print(f"No valid data in {log_csv_file_path}")
            continue

        # find the time sstresh is changed from first value to another value
        initial_value = sstresh_list[0]
        last_change_index = None

        for i in range(1, len(sstresh_list)):
            if sstresh_list[i - 1] == initial_value and sstresh_list[i] != initial_value:
                last_change_index = i

        if last_change_index is not None:
            notsearch_ss_exit_time = [now_list[last_change_index]]
        else:
            notsearch_ss_exit_time = None


        # find exit time reported by search algorithm
        if not search_exit_time or search_exit_time[0] == 0:
            search_exit_time = None        
        else:
            search_exit_time = [search_exit_time[0]]

        # If both search and notsearch exit times are available, choose the earlier one
        if notsearch_ss_exit_time is not None and search_exit_time is not None:
            if notsearch_ss_exit_time < search_exit_time:
                not_search_exit_time_list.append(notsearch_ss_exit_time[0])
                all_ss_exit_time_list.append(notsearch_ss_exit_time[0])
                exit_time = notsearch_ss_exit_time[0]
            else:
                search_exit_time_list.append(search_exit_time[0])
                all_ss_exit_time_list.append(search_exit_time[0])
                exit_time = search_exit_time[0]

        elif search_exit_time is not None:
            search_exit_time_list.append(search_exit_time[0])
            all_ss_exit_time_list.append(search_exit_time[0])
            exit_time = search_exit_time[0]

        elif notsearch_ss_exit_time is not None:
            not_search_exit_time_list.append(notsearch_ss_exit_time[0])
            all_ss_exit_time_list.append(notsearch_ss_exit_time[0])
            exit_time = notsearch_ss_exit_time[0]

        # Calculate delivery rates
        # delivery_rates_calculated, start_index_to_cal_delv_rate, time_cal_delv_rates = \
        # calculate_delivery_rate_per_ack(bytes_acked_list, now_list, rtt_s_list)

        delivery_rates_calculated_per_fixed_interval, time_cal_delv_rates = \
            calculate_delivery_rate_by_interval(bytes_acked_list, now_list, rtt_s_list)

        # convert delivery rates from MB/s to Mb/s
        delivery_rates_calculated_per_fixed_interval = [rate * 8 for rate in delivery_rates_calculated_per_fixed_interval]

        if delivery_rates_calculated_per_fixed_interval is not None:
            avg_delivery_rates.append(np.average(delivery_rates_calculated_per_fixed_interval))
            median_delivery_rates.append(np.median(delivery_rates_calculated_per_fixed_interval))
            
            # Find the delivery rate at the exit time
            if exit_time is not None and time_cal_delv_rates is not None:
                if exit_time < time_cal_delv_rates[-1]:
                    delivery_rate_exit_ = rate_at_exit(exit_time, time_cal_delv_rates, delivery_rates_calculated_per_fixed_interval)
                    delivery_rate_at_exit.append(delivery_rate_exit_)
                else:
                    delivery_rate_at_exit.append(None)
            else:
                delivery_rate_at_exit.append(None)
        else:
            avg_delivery_rates.append(None)
            median_delivery_rates.append(None)
            delivery_rate_at_exit.append(None)


        # find total bytes acked at the exit time
        if exit_time is not None and now_list:
            if exit_time < now_list[-1]:
                total_bytes_acked_at_exit = total_byte_acked_at_exit(exit_time, now_list, bytes_acked_list)
            else:
                total_bytes_acked_at_exit = None
        else:
            total_bytes_acked_at_exit = None

        total_bits_acked_at_exit_list.append(total_bytes_acked_at_exit * 8 if total_bytes_acked_at_exit is not None else None)

        # skip runs if exit time is None
        if exit_time is None:
            skip_run_indices.add(j + 1)

#############################################################################################
with open(input_path) as f:
    content = f.read()

runs = content.split("====== Run")[1:]  # Skip header part
data = []

for run in runs:
    run_number = int(run.split()[0])
    parts = run.split("Running")

    # if run_number in skip_run_indices:
    #     continue

    if TEST_WITH_SPEEDTEST:
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
    elif TEST_WITH_NDT7:
        pre_speed = extract_speedtest_values_ndt7(parts[1]) if len(parts) > 1 else (None, None, None)
        iperf = extract_iperf3_throughput(parts[2]) if len(parts) > 2 else None
        post_speed = extract_speedtest_values_ndt7(parts[3]) if len(parts) > 3 else (None, None, None)

        data.append({
            "Run": run_number,
            "Pre Download": pre_speed[0],
            "Pre Upload": pre_speed[1],
            "iperf3 Throughput": iperf,
            "Post Download": post_speed[0],
            "Post Upload": post_speed[1],
        })



df_client = pd.DataFrame(data)

# Save table to CSV
csv_path = os.path.join(output_path, "speedtest_results.csv")
df_client.to_csv(csv_path, index=False)

#####################################################
if CALC_THROUGHPUT:
    # calculate throughput from pcap files on server
    avg_throughputs = []
    median_throughputs = []
    throughput_all = {}
    time_throughput_all = {}

    if not os.path.exists(server_pcap_path):
        print(f"Server pcap path {server_pcap_path} does not exist")
    else:
        SERVER_IP = "130.215.28.249"
        INTERVAL = 1  # 20 ms interval for throughput calculation

        # find number of csv files in server_pcap_path
        csv_files = [f for f in os.listdir(server_pcap_path) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV files found in {server_pcap_path}")
            exit()

        num = len(csv_files)  # Get the last file

        for i in range(num+1):

            # if i + 1 in skip_run_indices:
            #     continue

            throughputs = []
            timestamps_thput = []

            pcap_csv_file_path = os.path.join(server_pcap_path, f"tcp_run_{i+1}.csv")
            if not os.path.exists(pcap_csv_file_path):
                print(f"File {pcap_csv_file_path} does not exist")
                continue

            if os.path.exists(pcap_csv_file_path):
                df = pd.read_csv(pcap_csv_file_path)

                # Find the first row where the ack number is greater than 1000 (sync the time of pcap file and log file)
                first_row = df[df['Ack number'] > 1000].iloc[0]

                # Get the time value from the first row
                time_first_ack = first_row['Time']

                # remove the times before time_first_ack
                df = df[df['Time'] >= time_first_ack]

                df['Time'] = df['Time'] - time_first_ack

                df_valid = df[(df["Source"] == SERVER_IP) & (df["retransmission"].isna())]
                df_valid = df_valid.sort_values("Time")

                start_time = df_valid["Time"].iloc[0]
                end_time = start_time + INTERVAL

                # Compute throughput in fixed intervals
                while end_time <= df_valid["Time"].iloc[-1]:
                    window_data = df_valid.loc[(df_valid["Time"] >= start_time) & (df_valid["Time"] < end_time)]
                    if not window_data.empty:
                        total_bytes = window_data["Length"].sum() * 8 * 1e-6
                        throughput = total_bytes / INTERVAL
                        throughputs.append(throughput)
                        timestamps_thput.append(end_time)

                    # Move to next window
                    start_time = end_time
                    end_time = start_time + INTERVAL
            else:
                print(f"File {pcap_csv_file_path} does not exist")

            if throughputs:
                throughput_all[i+1]= throughputs
                time_throughput_all[i+1] = timestamps_thput
                avg_throughputs.append(np.mean(throughputs))
                median_throughputs.append(np.median(throughputs))
            else:
                avg_throughputs.append(None)
                median_throughputs.append(None)

            # plot throughput over time
            plt.figure(figsize=(10, 5))
            plt.plot(timestamps_thput, throughputs, marker='o', label=f'Run {i+1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Throughput (Mb/s)')
            plt.title(f'Throughput Over Time for Run {i+1}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"throughput_run_{i+1}.png"))
            plt.close()

#########################################################
download_duration_post_list = []
download_bits_sent_post_list = []
download_duration_pre_list = []
download_bits_sent_pre_list = []
client_csv_file_path_list = []
throughput_df_pre_list = []
throughput_time_pre_list = []
throughput_df_post_list = []
throughput_time_post_list = []



if not os.path.exists(client_pcap_path):
    print(f"Client pcap path {client_pcap_path} does not exist")
else:
    # find number of pcap files in client_pcap_path
    client_pcap_files = [f for f in os.listdir(client_pcap_path) if f.endswith('.pcap')]
    if not client_pcap_files:
        print(f"No PCAP files found in {client_pcap_path}")
        exit()
    num_client_pcap_files = len(client_pcap_files)

    num_client_pcap_files_half = num_client_pcap_files // 2

    for k in range(num_client_pcap_files_half):
        # if k + 1 in skip_run_indices:
        #     continue
        client_pcap_file_postspeed_path = os.path.join(client_pcap_path, f"download_post_run_{k+1}.pcap")
        if not os.path.exists(client_pcap_file_postspeed_path):
            print(f"File {client_pcap_file_postspeed_path} does not exist")
            
        
        client_pcap_file_prespeed_path = os.path.join(client_pcap_path, f"download_pre_run_{k+1}.pcap")
        if not os.path.exists(client_pcap_file_prespeed_path):
            print(f"File {client_pcap_file_prespeed_path} does not exist")
            
        
        client_csv_file_path_list = []
        # if the pcap files exist, convert them to csv files:
        if os.path.exists(client_pcap_file_postspeed_path):
            client_csv_file_path_list.append(client_pcap_file_postspeed_path.replace('.pcap', '.csv'))
        if os.path.exists(client_pcap_file_prespeed_path):    
            client_csv_file_path_list.append(client_pcap_file_prespeed_path.replace('.pcap', '.csv'))

        for client_csv_file_path in client_csv_file_path_list:
            tshark_command = f"tshark -r {client_csv_file_path.replace('.csv', '.pcap')} -T fields -e frame.time_relative -e ip.src \
            -e ip.dst -e frame.len  -E header=y -E separator=, -E quote=n > {client_csv_file_path}"

            os.system(tshark_command)

            # Read the pcap file and extract pre and post download speeds
            df_client_pcap = pd.read_csv(client_csv_file_path)
            # df_client_pcap = df_client_pcap[(df_client_pcap["ip.src"] == '74.42.170.5')]


            # if df_client_pcap.empty:
            #     print(f"No data in {client_pcap_file_postspeed_path}")
            #     download_duration_post_list.append(None)
            #     download_bits_sent_post_list.append(None)
            #     download_duration_pre_list.append(None)
            #     download_bits_sent_pre_list.append(None)
            #     continue
            
            df_client_pcap = df_client_pcap.dropna()

            df_client_pcap["frame.len"] = df_client_pcap["frame.len"].astype(int)
            df_client_pcap["frame.time_relative"] = df_client_pcap["frame.time_relative"].astype(float)

            download_duration = df_client_pcap["frame.time_relative"].max() - df_client_pcap["frame.time_relative"].min()
            download_bits_sent = df_client_pcap["frame.len"].sum() * 8
            if download_bits_sent == 0:
                download_bits_sent = None
            if download_duration == 0:
                download_duration = None

            bin_width = 1  # seconds

            # Create time bin column
            df_client_pcap["time_bin"] = (df_client_pcap["frame.time_relative"] // bin_width) * bin_width

            # Group by each bin and sum frame lengths
            throughput_df = df_client_pcap.groupby("time_bin")["frame.len"].sum().reset_index()

            # Convert bytes to bits and divide by interval to get throughput in bps (or Mbps)
            throughput_df["throughput_Mbps"] = (throughput_df["frame.len"] * 8) / (bin_width * 1_000_000)

            if client_csv_file_path.endswith(f"post_run_{k+1}.csv"):
                download_duration_post_list.append(download_duration)
                download_bits_sent_post_list.append(download_bits_sent)
                throughput_df_post_list.append(throughput_df)
                throughput_time_post_list.append(throughput_df["time_bin"].tolist())
            else:
                download_duration_pre_list.append(download_duration)
                download_bits_sent_pre_list.append(download_bits_sent)
                throughput_df_pre_list.append(throughput_df)
                throughput_time_pre_list.append(throughput_df["time_bin"].tolist())


#################################### PLOT ####################################
# plot pre and post download speeds
if not df_client.empty:
    plt.figure()
    df_client.plot(x="Run", y=["Pre Download", "Post Download"], marker='o')
    plt.ylabel("Mb/s", fontsize=14)
    plt.title("Download Over Runs", fontsize=14)
    # plt.ylim([0, 110]) 
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "download_over_runs.png"))
    plt.close()

# plot cdf pre and post download speeds
if not df_client.empty and "Pre Download" in df_client.columns and "Post Download" in df_client.columns:
    pre_downloads = df_client["Pre Download"].dropna().tolist()
    post_downloads = df_client["Post Download"].dropna().tolist()

    cdf_pre_downloads, cdf_values_pre = calculate_cdf(pre_downloads)
    cdf_post_downloads, cdf_values_post = calculate_cdf(post_downloads)

    plt.figure()
    plt.plot(cdf_pre_downloads, cdf_values_pre, marker='o', label='CDF of Pre Download')
    plt.plot(cdf_post_downloads, cdf_values_post, marker='o', label='CDF of Post Download')
    plt.xlabel("Mb/s", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Pre and Post Download Speeds")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.xlim([0, 100])  # Set x-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_pre_post_download.png"))
    plt.close()

# plot average throughput 
if CALC_THROUGHPUT:
    if avg_throughputs and median_throughputs:
        plt.figure()
        plt.plot(range(1, num + 1), avg_throughputs, marker='o', label='Average Throughput')
        plt.plot(range(1, num + 1), median_throughputs, marker='o', label='Median Throughput')
        plt.xlabel("Run", fontsize=14)
        plt.ylabel("Mb/s", fontsize=14)
        plt.title("Average and Median Throughput Over Runs")
        plt.legend()
        # plt.ylim([0, 100]) 
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "avg_median_throughput_over_runs.png"))
        plt.close()

        # plot cdf of average throughput
        cdf_avg_throughputs, cdf_values_avg_throughputs = calculate_cdf(avg_throughputs)

        plt.figure()
        plt.plot(cdf_avg_throughputs, cdf_values_avg_throughputs, marker='o', label='CDF of Average Throughput')
        plt.xlabel("Mb/s", fontsize=14)
        plt.ylabel("CDF", fontsize=14)
        plt.title("CDF of Average Throughput")
        plt.legend()
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(left=0)  # Set x-axis to start from 0
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "cdf_avg_throughput.png"))
        plt.close()

        # plot cdf of median throughput
        cdf_median_throughputs, cdf_values_median_throughputs = calculate_cdf(median_throughputs)
        plt.figure()
        plt.plot(cdf_median_throughputs, cdf_values_median_throughputs, marker='o', label='CDF of Median Throughput')
        plt.xlabel("Mb/s", fontsize=14)
        plt.ylabel("CDF", fontsize=14)
        plt.title("CDF of Median Throughput")
        plt.legend()
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(left=0)  # Set x-axis to start from 0
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "cdf_median_throughput.png"))
        plt.close()


# plot average delivery rate
if avg_delivery_rates and median_delivery_rates:
    plt.figure()
    plt.plot(range(1, num_log_files + 1), avg_delivery_rates, marker='o', label='Average Delivery Rate')
    plt.plot(range(1, num_log_files + 1), median_delivery_rates, marker='o', label='Median Delivery Rate')
    plt.xlabel("Run", fontsize=14)
    plt.ylabel("Mb/s", fontsize=14)
    plt.title("Average and Median Delivery Rate Over Runs")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(bottom=0)  # Set y-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "avg_median_delivery_rate_over_runs.png"))
    plt.close()

# plot delivery rate at exit
if delivery_rate_at_exit:
    plt.figure()
    plt.plot(range(1, num_log_files + 1), delivery_rate_at_exit, marker='o', label='Delivery Rate at Exit')
    plt.xlabel("Run", fontsize=14)
    plt.ylabel("Mb/s", fontsize=14)
    plt.title("Delivery Rate at Exit Over Runs")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(bottom=0)  # Set y-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "delivery_rate_at_exit_over_runs.png"))
    plt.close()

# plot exit time
if search_exit_time_list:
    plt.figure()
    plt.plot(range(1, len(search_exit_time_list)+1), search_exit_time_list, marker='o', label='Exit Time')    
    plt.xlabel("Sample", fontsize=14)
    plt.ylabel("Time (s)", fontsize=14)
    plt.title("Exit Time Over Runs")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(bottom=0)  # Set y-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "exit_time_over_runs.png"))
    plt.close()

    # plot cdf of exit time
    search_exit_time_list = [time for time in search_exit_time_list if time is not None]  # Remove None value
    cdf_exit_time, cdf_values_exit_time = calculate_cdf(search_exit_time_list)
    plt.figure()
    plt.plot(cdf_exit_time, cdf_values_exit_time, marker='o', label='CDF of Exit Time')
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Exit Time")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=0)  # Set x-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_search_exit_time.png"))
    plt.close()

if not_search_exit_time_list:
    plt.figure()
    plt.plot(range(1, len(not_search_exit_time_list)+1), not_search_exit_time_list, marker='o', label='Non-Search Exit Time')
    plt.xlabel("Run", fontsize=14)
    plt.ylabel("Time (s)", fontsize=14)
    plt.title("Non-Search Exit Time Over Runs")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(bottom=0)  # Set y-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "non_search_exit_time_over_runs.png"))
    plt.close()
    
    # plot cdf of not search exit time
    not_search_exit_time_list = [time for time in not_search_exit_time_list if time is not None]  # Remove None value
    cdf_not_search_exit_time, cdf_values_not_search_exit_time = calculate_cdf(not_search_exit_time_list)
    plt.figure()
    plt.plot(cdf_not_search_exit_time, cdf_values_not_search_exit_time, marker='o', label='CDF of Non-Search Exit Time')
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Non-Search Exit Time")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=0)  # Set x-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_not_search_exit_time.png"))
    plt.close()

# combine seaerch and not search exit times (but for each one hase seperate color) and then also make cdf plot
if search_exit_time_list or not_search_exit_time_list:
    plt.figure()
    if search_exit_time_list:
        plt.plot(range(1, len(search_exit_time_list)+1), search_exit_time_list, marker='*', label='Search Exit Time', color='b')
    if not_search_exit_time_list:
        plt.plot(range(1, len(not_search_exit_time_list)+1), not_search_exit_time_list, marker='o', label='Loss Exit Time', color='r')
    plt.xlabel("Run", fontsize=14)
    plt.ylabel("Time (s)", fontsize=14)
    plt.title("Search and Non-Search Exit Time Over Runs")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(bottom=0)  # Set y-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "search_not_search_exit_time_over_runs.png"))
    plt.close()

if search_exit_time_list or not_search_exit_time_list:
    # Combine all with labels
    combined = [(x, 'SEARCH') for x in search_exit_time_list] + \
            [(x, 'NON_SEARCH') for x in not_search_exit_time_list]

    # Sort by exit time
    combined.sort(key=lambda x: x[0])

    # Separate sorted values and tags
    values = [x[0] for x in combined]
    tags = [x[1] for x in combined]

    # Compute CDF
    cdf = np.arange(1, len(values) + 1) / len(values)

    # Plot the combined CDF as dots with color based on tag
    for x, y, label in zip(values, cdf, tags):
        color = 'blue' if label == 'SEARCH' else 'red'
        marker = '*' if label == 'SEARCH' else 'o'
        plt.plot(x, y, marker=marker, linestyle='None', color=color, label=label)

    # Fix legend (avoid duplicates)
    handles = [
        plt.Line2D([0], [0], marker='*', markersize=10, color='w', label='SEARCH Exit', markerfacecolor='blue'),
        plt.Line2D([0], [0], marker='o', color='w', label='LOSS Exit', markerfacecolor='red')
    ]
    plt.legend(handles=handles)
    plt.xlabel('Exit Time (s)')
    plt.ylabel('CDF')
    plt.title('Combined CDF with Category-colored Points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "combined_cdf_search_not_search_exit_time.png"))
    plt.close()

# plot cdf all ss exit times
if all_ss_exit_time_list:
    all_ss_exit_time_list = [time for time in all_ss_exit_time_list if time is not None]  # Remove None value
    cdf_all_ss_exit_time, cdf_values_all_ss_exit_time = calculate_cdf(all_ss_exit_time_list)
    plt.figure()
    plt.plot(cdf_all_ss_exit_time, cdf_values_all_ss_exit_time, marker='o', label='CDF of All SS Exit Time')
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of All SS Exit Time")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=0)  # Set x-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_all_ss_exit_time.png"))
    plt.close()

# calculate cdf of AVG delivery rates and then plot it
if avg_delivery_rates:
    cdf_avg_delivery_rates, cdf_values_avg = calculate_cdf(avg_delivery_rates)

    plt.figure()
    plt.plot(cdf_avg_delivery_rates, cdf_values_avg, marker='o', label='CDF of Average Delivery Rates')
    plt.xlabel("Mb/s", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Average Delivery Rates")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=0)  # Set x-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_avg_delivery_rates.png"))
    plt.close()

# calculate cdf of delivery rate at exit and then plot it
if delivery_rate_at_exit:
    # Filter out None values from delivery_rate_at_exit
    delivery_rate_at_exit = [rate for rate in delivery_rate_at_exit if rate is not None]
    cdf_delivery_rate_exit, cdf_values_exit = calculate_cdf(delivery_rate_at_exit)

    plt.figure()
    plt.plot(cdf_delivery_rate_exit, cdf_values_exit, marker='o', color='m', label='CDF of Delivery Rate at Exit')
    plt.xlabel("Mb/s", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Delivery Rate at Exit")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=0)  # Set x-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_delivery_rate_at_exit.png"))
    plt.close()

# Plot the cdf of delivery rate and delivery rate at exit in one figure
if avg_delivery_rates and delivery_rate_at_exit:
    plt.figure()
    plt.plot(cdf_avg_delivery_rates, cdf_values_avg, marker='o', color='b', label='CDF of Average Delivery Rates')
    plt.plot(cdf_delivery_rate_exit, cdf_values_exit, marker='o', color='m', label='CDF of Delivery Rate at Exit')
    plt.xlabel("Mb/s", fontsize=14)  
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Delivery Rates")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=0)  # Set x-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_delivery_rates_combined.png"))
    plt.close()

# plot the cdf of median delivery rates and delivery rate at exit in one figure
if median_delivery_rates and delivery_rate_at_exit:
    cdf_median_delivery_rates, cdf_values_median = calculate_cdf(median_delivery_rates)

    plt.figure()
    plt.plot(cdf_median_delivery_rates, cdf_values_median, marker='o', color='c', label='CDF of Median Delivery Rates')
    plt.plot(cdf_delivery_rate_exit, cdf_values_exit, marker='o', color='m', label='CDF of Delivery Rate at Exit')
    plt.xlabel("Mb/s", fontsize=14)  
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Median Delivery Rates and Delivery Rate at Exit")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=-0.05)  # Set x-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_median_delivery_rates_combined.png"))
    plt.close()

# plot the cdf of median throughput and delivery rate at exit in one figure
if median_throughputs and delivery_rate_at_exit:
    plt.figure()
    plt.plot(cdf_median_throughputs, cdf_values_median_throughputs, marker='o', color='c', label='CDF of Median Throughput')
    plt.plot(cdf_delivery_rate_exit, cdf_values_exit, marker='o', color='m', label='CDF of Delivery Rate at Exit')
    plt.xlabel("Mb/s", fontsize=14)  
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Median Throughput and Delivery Rate at Exit")    
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=-0.05)  # Set x-axis to start from
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_median_throughput_delivery_rate_exit_combined.png"))
    plt.close()
     
# plot diff between median throughput and delivery rate at exit
if median_throughputs and delivery_rate_at_exit:
    # Filter out None values from delivery_rate_at_exit
    delivery_rate_at_exit = [rate for rate in delivery_rate_at_exit if rate is not None]

    # Create a Series for median throughput
    median_throughput_series = pd.Series(median_throughputs, index=range(1, len(median_throughputs) + 1))

    # Create a Series for delivery rate at exit
    delivery_rate_exit_series = pd.Series(delivery_rate_at_exit, index=range(1, len(delivery_rate_at_exit) + 1))

    # Only compute when both values are not None (i.e., not NaN in pandas)
    valid_mask = median_throughput_series.notna() & delivery_rate_exit_series.notna()

    normalized_dif_median_throughput_exit_percent = pd.Series(index=median_throughput_series.index, dtype="float")
    normalized_dif_median_throughput_exit_percent[valid_mask] = (
        (median_throughput_series[valid_mask] - delivery_rate_exit_series[valid_mask]) /
        median_throughput_series[valid_mask]
    ) * 100

    # plot normalized difference
    plt.figure()
    plt.plot(range(1, len(normalized_dif_median_throughput_exit_percent) + 1), normalized_dif_median_throughput_exit_percent, marker='o', color='brown', label='Normalized Difference')
    plt.xlabel("Run", fontsize=14)
    plt.ylabel("Normalized Difference (%)", fontsize=14)
    plt.title("Normalized Difference Between Median Throughput and Delivery Rate at Exit")    
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "normalized_difference_median_throughput_vs_delivery_rate_exit.png"))
    plt.close()

    # plot cdf of normalized difference
    cdf_normalized_dif_median_throughput_exit, cdf_values_normalized_median_throughput_exit = calculate_cdf(normalized_dif_median_throughput_exit_percent)

    plt.figure()
    plt.plot(cdf_normalized_dif_median_throughput_exit, cdf_values_normalized_median_throughput_exit, marker='o', color='brown', linestyle="", label='CDF of Normalized Difference')
    # plot v line on 0
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Normalized Difference (%)", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Normalized Difference Between Median Throughput and Delivery Rate at Exit")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.xlim(left=-10, right=20)  # Set x-axis to

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_normalized_difference_median_throughput_vs_delivery_rate_exit.png"))
    plt.close()


    # plot just difference between median throughput and delivery rate at exit (median throughput - delivery rate at exit)
    plt.figure()
    plt.plot(range(1, len(median_throughputs) + 1), np.array(median_throughputs) - np.array(delivery_rate_at_exit), marker='o', color='brown', label='Difference')
    plt.xlabel("Run", fontsize=14)
    plt.ylabel("Difference (Mb/s)", fontsize=14)
    plt.title("Difference Between Median Throughput and Delivery Rate at Exit")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "difference_median_throughput_vs_delivery_rate_exit.png"))
    plt.close()

    # plot cdf of difference between median throughput and delivery rate at exit
    difference_median_throughput_exit = np.array(median_throughputs) - np.array(delivery_rate_at_exit)
    difference_median_throughput_exit = [diff for diff in difference_median_throughput_exit if diff is not None]  # Remove None values
    cdf_difference_median_throughput_exit, cdf_values_difference_median_throughput_exit = calculate_cdf(difference_median_throughput_exit)

    plt.figure()
    plt.plot(cdf_difference_median_throughput_exit, cdf_values_difference_median_throughput_exit, marker='o', color='brown', linestyle="", label='CDF of Difference')
    # plot v line on 0
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Difference (Mb/s)", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Difference Between Median Throughput and Delivery Rate at Exit")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.xlim(left=-10, right=20)  # Set x-axis to start from -100 to 100
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_difference_median_throughput_vs_delivery_rate_exit.png"))
    plt.close()

# scatter plot of speed test pre download (x) and search exit delivery rate (y-axis), one dot for each run
if not df_client.empty and delivery_rate_at_exit:
    delivery_rate_at_exit = [rate for rate in delivery_rate_at_exit if rate is not None]
    plt.figure()
    plt.scatter(df_client["Pre Download"], delivery_rate_at_exit, marker='o', color='g')
    plt.xlabel("Pre Download Speed (Mb/s)", fontsize=14)
    plt.ylabel("Delivery Rate at Exit (Mb/s)", fontsize=14)
    plt.title("Pre Download Speed vs Delivery Rate at Exit")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=-0.05)
    plt.ylim(bottom=-5)  # Set y-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "pre_download_vs_delivery_rate_exit.png"))
    plt.close()

# scatter plot of speed test pre download (x) and search exit delivery rate (y-axis), one dot for each run
if not df_client.empty and delivery_rate_at_exit:
    # Filter out None values from delivery_rate_at_exit
    delivery_rate_at_exit = [rate for rate in delivery_rate_at_exit if rate is not None]
    plt.figure()
    plt.scatter(df_client["Post Download"], delivery_rate_at_exit, marker='o', color='brown')
    plt.xlabel("Post Download Speed (Mb/s)", fontsize=14)
    plt.ylabel("Delivery Rate at Exit (Mb/s)", fontsize=14)
    plt.title("Post Download Speed vs Delivery Rate at Exit")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=-0.05)
    plt.ylim(bottom=-5)   # Set y-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "post_download_vs_delivery_rate_exit.png"))
    plt.close()

# compute normalized difference (pst download - delivery rate at exit) / post download * 100
if not df_client.empty and delivery_rate_at_exit:

    delivery_rate_series = pd.Series(delivery_rate_at_exit, index=df_client.index)

    # Only compute when both values are not None (i.e., not NaN in pandas)
    valid_mask = df_client["Post Download"].notna() & delivery_rate_series.notna()

    normalized_dif_post_exit_percent = pd.Series(index=df_client.index, dtype="float")
    normalized_dif_post_exit_percent[valid_mask] = (
        (df_client["Post Download"][valid_mask] - delivery_rate_series[valid_mask]) /
        df_client["Post Download"][valid_mask]
    ) * 100


    # plot normalized difference
    plt.figure()
    plt.plot(range(1, len(normalized_dif_post_exit_percent)+1), normalized_dif_post_exit_percent, marker='o', color='g', label='Normalized Difference')
    plt.xlabel("Run", fontsize=14)
    plt.ylabel("Normalized Difference (%)", fontsize=14)
    plt.title("Normalized Difference Between Post Download Speed and Delivery Rate at Exit")    
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "normalized_difference_post_download_vs_delivery_rate_exit.png"))
    plt.close()

    # plot cdf of normalized difference
    cdf_normalized_dif_post_exit, cdf_values_normalized_post = calculate_cdf(normalized_dif_post_exit_percent)
    plt.figure()
    plt.plot(cdf_normalized_dif_post_exit, cdf_values_normalized_post, marker='o', color='g',linestyle="",  label='CDF of Normalized Difference')
    #plot v line on 0
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Normalized Difference (%)", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Normalized Difference Between Post Download Speed and Delivery Rate at Exit")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.xlim(left=-10, right=20)  # Set x-axis to start from -100 to 100
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_normalized_difference_post_download_vs_delivery_rate_exit.png"))
    plt.close()


# compute normalized difference (pre download - delivery rate at exit) / pre download * 100
if not df_client.empty and delivery_rate_at_exit:

    delivery_rate_series = pd.Series(delivery_rate_at_exit, index=df_client.index)

    # Only compute when both values are not None (i.e., not NaN in pandas)
    valid_mask = df_client["Pre Download"].notna() & delivery_rate_series.notna()

    normalized_dif_pre_exit_percent = pd.Series(index=df_client.index, dtype="float")
    normalized_dif_pre_exit_percent[valid_mask] = (
        (df_client["Pre Download"][valid_mask] - delivery_rate_series[valid_mask]) /
        df_client["Pre Download"][valid_mask]
    ) * 100

    # plot normalized difference
    plt.figure()
    plt.plot(range(1, len(normalized_dif_pre_exit_percent) + 1), normalized_dif_pre_exit_percent, marker='o', color='brown', label='Normalized Difference')
    plt.xlabel("Run", fontsize=14)
    plt.ylabel("Normalized Difference (%)", fontsize=14)
    plt.title("Normalized Difference Between Pre Download Speed and Delivery Rate at Exit")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "normalized_difference_pre_download_vs_delivery_rate_exit.png"))
    plt.close()

    # plot cdf of normalized difference
    # remove nan values from normalized_dif_pre_exit_percent
    normalized_dif_pre_exit_percent = normalized_dif_pre_exit_percent.dropna()

    cdf_normalized_dif_pre_exit, cdf_values_normalized_pre = calculate_cdf(normalized_dif_pre_exit_percent)
    plt.figure()
    plt.plot(cdf_normalized_dif_pre_exit, cdf_values_normalized_pre, marker='o', color='brown',linestyle="-", label='CDF of Normalized Difference')
        #plot v line on 0
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Normalized Difference (%)", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Normalized Difference Between Pre Download Speed and Delivery Rate at Exit")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.xlim(left=-10, right=20)  # Set x-axis to start from -100 to 100
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_normalized_difference_pre_download_vs_delivery_rate_exit.png"))
    plt.close()


# plot cdf time for speedtest download pre and put SEARCH exit time on it
if download_duration_pre_list and all_ss_exit_time_list:
    # REMOVE NONE VALUES FROM BOTH LISTS
    download_duration_pre_list = [x for x in download_duration_pre_list if x > 0]
    all_ss_exit_time_list = [x for x in all_ss_exit_time_list if x is not None]

    # Calculate CDF for download duration pre
    cdf_download_duration_pre, cdf_values_download_pre = calculate_cdf(download_duration_pre_list)
    cdf_exit_time, cdf_values_exit_time = calculate_cdf(all_ss_exit_time_list)

    plt.figure()
    plt.plot(cdf_download_duration_pre, cdf_values_download_pre, marker='o', label='CDF of Download Duration Pre')
    plt.plot(cdf_exit_time, cdf_values_exit_time, marker='o', color='m', label='CDF of Exit Time')
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Download Duration Pre and Exit Time")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=-1)  # Set x-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_download_duration_pre_and_exit_time.png"))
    plt.close()

# plot cdf time for speedtest download post and put SEARCH exit time on it
if download_duration_post_list and all_ss_exit_time_list:
    # REMOVE NONE VALUES FROM BOTH LISTS
    download_duration_post_list = [x for x in download_duration_post_list if x>0]
    all_ss_exit_time_list = [x for x in all_ss_exit_time_list if x is not None]

    # Calculate CDF for download duration post
    cdf_download_duration_post, cdf_values_download_post = calculate_cdf(download_duration_post_list)
    cdf_exit_time, cdf_values_exit_time = calculate_cdf(all_ss_exit_time_list)

    plt.figure()
    plt.plot(cdf_download_duration_post, cdf_values_download_post, marker='o', label='CDF of Download Duration Post')
    plt.plot(cdf_exit_time, cdf_values_exit_time, marker='o', color='m', label='CDF of Exit Time')
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Download Duration Post and Exit Time")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=-1)  # Set x-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_download_duration_post_and_exit_time.png"))
    plt.close()

# plot download bits sent pre and search bit acked at exit
if download_bits_sent_pre_list and total_bits_acked_at_exit_list:
    # remove None values from both lists
    download_bits_sent_pre_list_Mb = [x / 1e6 for x in download_bits_sent_pre_list if x is not None] # Convert to Mb
    total_bits_acked_at_exit_list_Mb_for_pre = [x * 8 for x in total_bits_acked_at_exit_list if x is not None] # Convert to Mb
    # Calculate CDF for download bits sent pre
    cdf_download_bits_sent_pre, cdf_values_download_pre = calculate_cdf(download_bits_sent_pre_list_Mb)
    cdf_total_bits_acked_exit, cdf_values_total_bits_acked_exit = calculate_cdf(total_bits_acked_at_exit_list_Mb_for_pre)

    plt.figure()
    plt.plot(cdf_download_bits_sent_pre, cdf_values_download_pre, marker='o', label='CDF of Download Bits Sent Pre')
    plt.plot(cdf_total_bits_acked_exit, cdf_values_total_bits_acked_exit, marker='o', color='m', label='CDF of Total Bits Acked at Exit')
    plt.xlabel("Mb", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Download Bits Sent Pre and Total Bits Acked at Exit")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=0)  # Set x-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_download_bits_sent_pre_and_total_bits_acked_at_exit.png"))
    plt.close()

# plot download bits sent post and search bit acked at exit
if download_bits_sent_post_list and total_bits_acked_at_exit_list:
    # remove none values from download_bits_sent_post_list and total_bits_acked_at_exit_list
    download_bits_sent_post_list_Mb = [x / 1e6 for x in download_bits_sent_post_list if x is not None] # Convert to Mb
    total_bits_acked_at_exit_list_Mb_for_post = [x * 8 for x in total_bits_acked_at_exit_list if x is not None] # Convert to Mb

    # Calculate CDF for download bits sent post
    cdf_download_bits_sent_post, cdf_values_download_post = calculate_cdf(download_bits_sent_post_list_Mb)
    cdf_total_bits_acked_exit, cdf_values_total_bits_acked_exit = calculate_cdf(total_bits_acked_at_exit_list_Mb_for_post)

    plt.figure()
    plt.plot(cdf_download_bits_sent_post, cdf_values_download_post, marker='o', label='CDF of Download Bits Sent Post')
    plt.plot(cdf_total_bits_acked_exit, cdf_values_total_bits_acked_exit, marker='o', color='m', label='CDF of Total Bits Acked at Exit')
    plt.xlabel("Mb", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Download Bits Sent Post and Total Bits Acked at Exit")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=0)  # Set x-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_download_bits_sent_post_and_total_bits_acked_at_exit.png"))
    plt.close()

# plot throughput of search and throughput of pre download on one graph
if CALC_THROUGHPUT:
    if throughput_df_pre_list and throughput_all:
        for i in range(len(throughput_df_pre_list)):
            if i+1 in time_throughput_all:
                
                plt.figure(figsize=(10, 5))
                plt.plot(throughput_time_pre_list[i], throughput_df_pre_list[i]["throughput_Mbps"], marker='o', label=f'Pre Download Run {i+1}')
                plt.plot(time_throughput_all[i+1], throughput_all[i+1], marker='x', label=f'Search Throughput Run {i+1}')
                plt.xlabel('Time (s)', fontsize=14)
                plt.ylabel('Throughput (Mb/s)', fontsize=14)
                plt.title('Throughput of Pre Download and Search Over Time')
                plt.legend()
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.xlim([-1, 11])
                plt.ylim(bottom=0)  # Set y-axis to start from 0
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, f"throughput_pre_download_and_search_{i+1}.png"))
                plt.close()

    # plot throughput of search and throughput of post download on one graph
    if throughput_df_post_list and throughput_all and throughput_df_pre_list:
        for i in range(len(throughput_all)):
            # if throughput_time_post_list[i] or throughput_time_pre_list[i] does not exist, skip this iteration
            if i+1 in time_throughput_all:
                plt.figure(figsize=(10, 5))
                plt.plot(throughput_time_post_list[i], throughput_df_post_list[i]["throughput_Mbps"], marker='o', label=f'Post Download Run {i+1}')
                plt.plot(time_throughput_all[i+1], throughput_all[i+1], marker='x', label=f'Search Throughput Run {i+1}')
                plt.plot(throughput_time_pre_list[i], throughput_df_pre_list[i]["throughput_Mbps"], marker='o', label=f'Pre Download Run {i+1}')
                plt.xlabel('Time (s)', fontsize=14)
                plt.ylabel('Throughput (Mb/s)', fontsize=14)
                plt.title('Throughput of Post and pre Download and Search Over Time')
                plt.legend()
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.xlim([-1, 11])
                plt.ylim(bottom=0)  # Set y-axis to start from 0
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, f"throughput_post_pre_download_and_search_{i+1}.png"))
                plt.close()

    # plot throughput of search and throughput of post download on one graph
    if throughput_df_post_list and throughput_all:
        for i in range(len(throughput_df_post_list)):
            if i+1 in time_throughput_all:
                plt.figure(figsize=(10, 5))
                plt.plot(throughput_time_post_list[i], throughput_df_post_list[i]["throughput_Mbps"], marker='o', label=f'Post Download Run {i+1}')
                plt.plot(time_throughput_all[i+1], throughput_all[i+1], marker='x', label=f'Search Throughput Run {i+1}')
                plt.xlabel('Time (s)', fontsize=14)
                plt.ylabel('Throughput (Mb/s)', fontsize=14)
                plt.title('Throughput of Post Download and Search Over Time')
                plt.legend()
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.xlim([-1, 11])
                plt.ylim(bottom=0)  # Set y-axis to start from 0
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, f"throughput_post_download_and_search_{i+1}.png"))
                plt.close()

# Save all data to a CSV file
if avg_throughputs and median_throughputs and avg_delivery_rates and median_delivery_rates and delivery_rate_at_exit and all_ss_exit_time_list:
    df_results = pd.DataFrame({
        "Run": range(1, num + 2),
        "Average Throughput (Mb/s)": avg_throughputs,
        "Median Throughput (Mb/s)": median_throughputs,
        "Average Delivery Rate (Mb/s)": avg_delivery_rates,
        "Median Delivery Rate (Mb/s)": median_delivery_rates,
        "Delivery Rate at Exit (Mb/s)": delivery_rate_at_exit,
        "Exit Time (s)": all_ss_exit_time_list
    })
    results_csv_path = os.path.join(output_path, "speedtest_analysis_results.csv")
    df_results.to_csv(results_csv_path, index=False)
