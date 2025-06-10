import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import bisect

input_path = "/home/maryam/SEARCH/speedtest_CCalgs/6-speedtest_just_download_secure/speedtest_modemcable_justdownload2.txt"
output_path = "/home/maryam/SEARCH/speedtest_CCalgs/6-speedtest_just_download_secure/result"
if not os.path.exists(output_path):
    os.makedirs(output_path)

server_log_path = "/home/maryam/SEARCH/speedtest_CCalgs/6-speedtest_just_download_secure/server_speedtest_homecablemodem_just_download2/data/log_search"
server_pcap_path = "/home/maryam/SEARCH/speedtest_CCalgs/6-speedtest_just_download_secure/server_speedtest_homecablemodem_just_download2/data/pcap_server"

client_pcap_path = "/home/maryam/SEARCH/speedtest_CCalgs/6-speedtest_just_download_secure/pcap_dumps"



TEST_WITH_SPEEDTEST = True
TEST_WITH_NDT7 = False

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
    for i in range(len(now)):
        target_time = now[i] - rtt[i]
        if target_time <= 0:
            start_index = i+1
            continue
        # Find rightmost index j where now[j] <= target_time
        j = bisect.bisect_right(now, target_time, 0, i) - 1
        if j >= 0:
            delta_bytes = bytes_acked[i] - bytes_acked[j]
            rate = delta_bytes / rtt[i]
            delivery_rates.append(rate)
            time_cal_delv_rates.append(now[i])
            
    return delivery_rates, start_index, time_cal_delv_rates if delivery_rates else None

# Calculate CDF from a list of data
def calculate_cdf(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    cdf = [(i + 1) / n for i in range(n)]
    return sorted_data, cdf
#############################################################################################
with open(input_path) as f:
    content = f.read()

runs = content.split("====== Run")[1:]  # Skip header part
data = []

for run in runs:
    run_number = int(run.split()[0])
    parts = run.split("Running")

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
# calculate throughput from pcap files on server
avg_throughputs = []
median_throughputs = []
throughput_all = {}
time_throughput_all = {}

if not os.path.exists(server_pcap_path):
    print(f"Server pcap path {server_pcap_path} does not exist")
else:
    SERVER_IP = "130.215.28.249"
    INTERVAL = 0.2

    # find number of csv files in server_pcap_path
    csv_files = [f for f in os.listdir(server_pcap_path) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {server_pcap_path}")
        exit()

    num = len(csv_files) - 1  # Get the last file

    for i in range(num + 1):
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
            avg_throughputs.append(sum(throughputs) / len(throughputs))
            median_throughputs.append(sorted(throughputs)[len(throughputs) // 2])
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

############################################################
# calculate delivery rate based on log files
avg_delivery_rates = []
median_delivery_rates = []
delivery_rate_at_exit = []
exit_time_list = []
total_bits_acked_at_exit_list = []

if not os.path.exists(server_log_path):
    print(f"Server log path {server_log_path} does not exist")
else:
    log_csv_files = [f for f in os.listdir(server_log_path) if f.endswith('.csv')]
    if not log_csv_files:
        print(f"No CSV files found in {server_log_path}")
        exit()  
    num_log_files = len(log_csv_files) - 1  # Get the last file

    for j in range(num_log_files + 1):
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

        exit_time_list.append(search_exit_time[0] if search_exit_time else None)

        if len(bytes_acked_list) == 0 or len(now_list) == 0 or len(rtt_s_list) == 0:
            print(f"No valid data in {log_csv_file_path}")
            continue

        # Calculate delivery rates
        delivery_rates_calculated, start_index_to_cal_delv_rate, time_cal_delv_rates = \
        calculate_delivery_rate_per_ack(bytes_acked_list, now_list, rtt_s_list)

        # convert delivery rates from MB/s to Mb/s
        delivery_rates_calculated = [rate * 8 for rate in delivery_rates_calculated]

        if delivery_rates_calculated is not None:
            avg_delivery_rates.append(sum(delivery_rates_calculated) / len(delivery_rates_calculated))
            median_delivery_rates.append(sorted(delivery_rates_calculated)[len(delivery_rates_calculated) // 2])
            
            # Find the delivery rate at the exit time
            if search_exit_time:
                exit_time = search_exit_time[0]
                if exit_time < time_cal_delv_rates[-1]:
                    index_at_exit = bisect.bisect_right(time_cal_delv_rates, exit_time) - 1
                    delivery_rate_at_exit.append(delivery_rates_calculated[index_at_exit])
                else:
                    delivery_rate_at_exit.append(None)
            else:
                delivery_rate_at_exit.append(None)
        else:
            avg_delivery_rates.append(None)
            median_delivery_rates.append(None)
            delivery_rate_at_exit.append(None)


        # find total bytes acked at the exit time
        if search_exit_time:
            exit_time = search_exit_time[0]
            if exit_time < now_list[-1]:
                index_at_exit = bisect.bisect_right(now_list, exit_time) - 1
                total_bytes_acked_at_exit = bytes_acked_list[index_at_exit]
            else:
                total_bytes_acked_at_exit = bytes_acked_list[-1]
        else:
            total_bytes_acked_at_exit = None
        total_bits_acked_at_exit_list.append(total_bytes_acked_at_exit * 8 if total_bytes_acked_at_exit is not None else None)

#########################################################
upload_duration_post_list = []
upload_bits_sent_post_list = []
upload_duration_pre_list = []
upload_bits_sent_pre_list = []
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

    for k in range((num_client_pcap_files)// 2):
        client_pcap_file_postspeed_path = os.path.join(client_pcap_path, f"download_post_run_{k+1}.pcap")
        if not os.path.exists(client_pcap_file_postspeed_path):
            print(f"File {client_pcap_file_postspeed_path} does not exist")
            continue
        
        client_pcap_file_prespeed_path = os.path.join(client_pcap_path, f"download_pre_run_{k+1}.pcap")
        if not os.path.exists(client_pcap_file_prespeed_path):
            print(f"File {client_pcap_file_prespeed_path} does not exist")
            continue
        
        client_csv_file_path_list = []
        client_csv_file_path_list.append(client_pcap_file_postspeed_path.replace('.pcap', '.csv'))
        client_csv_file_path_list.append(client_pcap_file_prespeed_path.replace('.pcap', '.csv'))

        for client_csv_file_path in client_csv_file_path_list:
            tshark_command = f"tshark -r {client_csv_file_path.replace('.csv', '.pcap')} -T fields -e frame.time_relative -e ip.src \
            -e ip.dst -e frame.len  -E header=y -E separator=, -E quote=n > {client_csv_file_path}"

            os.system(tshark_command)

            # Read the pcap file and extract pre and post download speeds
            df_client_pcap = pd.read_csv(client_csv_file_path)
            # df_client_pcap = df_client_pcap[(df_client_pcap["ip.src"] == '74.42.170.5')]


            if df_client_pcap.empty:
                print(f"No data in {client_pcap_file_postspeed_path}")
                upload_duration_post_list.append(None)
                upload_bits_sent_post_list.append(None)
                upload_duration_pre_list.append(None)
                upload_bits_sent_pre_list.append(None)
                continue
            
            df_client_pcap = df_client_pcap.dropna()

            df_client_pcap["frame.len"] = df_client_pcap["frame.len"].astype(int)
            df_client_pcap["frame.time_relative"] = df_client_pcap["frame.time_relative"].astype(float)

            upload_duration = df_client_pcap["frame.time_relative"].max() - df_client_pcap["frame.time_relative"].min()
            upload_bits_sent = df_client_pcap["frame.len"].sum() * 8

            bin_width = 0.2  # seconds

            # Create time bin column
            df_client_pcap["time_bin"] = (df_client_pcap["frame.time_relative"] // bin_width) * bin_width

            # Group by each bin and sum frame lengths
            throughput_df = df_client_pcap.groupby("time_bin")["frame.len"].sum().reset_index()

            # Convert bytes to bits and divide by interval to get throughput in bps (or Mbps)
            throughput_df["throughput_Mbps"] = (throughput_df["frame.len"] * 8) / (bin_width * 1_000_000)

            if client_csv_file_path.endswith(f"post_run_{k+1}.csv"):
                upload_duration_post_list.append(upload_duration)
                upload_bits_sent_post_list.append(upload_bits_sent)
                throughput_df_post_list.append(throughput_df)
                throughput_time_post_list.append(throughput_df["time_bin"].tolist())
            else:
                upload_duration_pre_list.append(upload_duration)
                upload_bits_sent_pre_list.append(upload_bits_sent)
                throughput_df_pre_list.append(throughput_df)
                throughput_time_pre_list.append(throughput_df["time_bin"].tolist())


#################################### PLOT ####################################
# plot pre and post download speeds
if not df_client.empty:
    plt.figure()
    df_client.plot(x="Run", y=["Pre Download", "Post Download"], marker='o')
    plt.ylabel("Mb/s", fontsize=14)
    plt.title("Download Over Runs", fontsize=14)
    plt.ylim([0, 110]) 
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
    plt.xlim([0, 100])  # Set x-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_pre_post_download.png"))
    plt.close()

# plot average throughput 
if avg_throughputs and median_throughputs:
    plt.figure()
    plt.plot(range(1, num + 2), avg_throughputs, marker='o', label='Average Throughput')
    plt.plot(range(1, num + 2), median_throughputs, marker='o', label='Median Throughput')
    plt.xlabel("Run", fontsize=14)
    plt.ylabel("Mb/s", fontsize=14)
    plt.title("Average and Median Throughput Over Runs")
    plt.legend()
    plt.ylim([0, 100]) 
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

# plot average delivery rate
if avg_delivery_rates and median_delivery_rates:
    plt.figure()
    plt.plot(range(1, num_log_files + 2), avg_delivery_rates, marker='o', label='Average Delivery Rate')
    plt.plot(range(1, num_log_files + 2), median_delivery_rates, marker='o', label='Median Delivery Rate')
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
    plt.plot(range(1, num_log_files + 2), delivery_rate_at_exit, marker='o', label='Delivery Rate at Exit')
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
if exit_time_list:
    plt.figure()
    plt.plot(range(1, num_log_files + 2), exit_time_list, marker='o', label='Exit Time')    
    plt.xlabel("Run", fontsize=14)
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
    cdf_exit_time, cdf_values_exit_time = calculate_cdf(exit_time_list)
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
    plt.savefig(os.path.join(output_path, "cdf_exit_time.png"))
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

# scatter plot of speed test pre download (x) and search exit delivery rate (y-axis), one dot for each run
if not df_client.empty and delivery_rate_at_exit:
    plt.figure()
    plt.scatter(df_client["Pre Download"], delivery_rate_at_exit, marker='o', color='g')
    plt.xlabel("Pre Download Speed (Mb/s)", fontsize=14)
    plt.ylabel("Delivery Rate at Exit (Mb/s)", fontsize=14)
    plt.title("Pre Download Speed vs Delivery Rate at Exit")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.xlim(left=0)
    # plt.ylim(bottom=0)  # Set y-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "pre_download_vs_delivery_rate_exit.png"))
    plt.close()

# scatter plot of speed test pre download (x) and search exit delivery rate (y-axis), one dot for each run
if not df_client.empty and delivery_rate_at_exit:
    plt.figure()
    plt.scatter(df_client["Post Download"], delivery_rate_at_exit, marker='o', color='brown')
    plt.xlabel("Post Download Speed (Mb/s)", fontsize=14)
    plt.ylabel("Delivery Rate at Exit (Mb/s)", fontsize=14)
    plt.title("Post Download Speed vs Delivery Rate at Exit")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.xlim([0, 105])
    # plt.ylim(bottom=0)  # Set y-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "post_download_vs_delivery_rate_exit.png"))
    plt.close()

# compute normalized difference (pst download - delivery rate at exit) / post download * 100
if not df_client.empty and delivery_rate_at_exit:
    normalized_dif_post_exit_percent = (
        (df_client["Post Download"] - pd.Series(delivery_rate_at_exit)) / df_client["Post Download"]
    ) * 100

    # plot normalized difference
    plt.figure()
    plt.plot(range(1, num_log_files + 2), normalized_dif_post_exit_percent, marker='o', color='g', label='Normalized Difference')
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
    plt.plot(cdf_normalized_dif_post_exit, cdf_values_normalized_post, marker='o', color='g', label='CDF of Normalized Difference')
    plt.xlabel("Normalized Difference (%)", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Normalized Difference Between Post Download Speed and Delivery Rate at Exit")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=-10, right=20)  # Set x-axis to start from -100 to 100
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_normalized_difference_post_download_vs_delivery_rate_exit.png"))
    plt.close()


# compute normalized difference (pre download - delivery rate at exit) / pre download * 100
if not df_client.empty and delivery_rate_at_exit:
    normalized_dif_pre_exit_percent = (
        (df_client["Pre Download"] - pd.Series(delivery_rate_at_exit)) / df_client["Pre Download"]
    ) * 100
    # plot normalized difference
    plt.figure()
    plt.plot(range(1, num_log_files + 2), normalized_dif_pre_exit_percent, marker='o', color='brown', label='Normalized Difference')
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
    cdf_normalized_dif_pre_exit, cdf_values_normalized_pre = calculate_cdf(normalized_dif_pre_exit_percent)
    plt.figure()
    plt.plot(cdf_normalized_dif_pre_exit, cdf_values_normalized_pre, marker='o', color='brown', label='CDF of Normalized Difference')
    plt.xlabel("Normalized Difference (%)", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Normalized Difference Between Pre Download Speed and Delivery Rate at Exit")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=-10, right=20)  # Set x-axis to start from -100 to 100
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_normalized_difference_pre_download_vs_delivery_rate_exit.png"))
    plt.close()


# plot cdf time for speedtest upload pre and put SEARCH exit time on it
if upload_duration_pre_list and exit_time_list:
    # REMOVE NONE VALUES FROM BOTH LISTS
    upload_duration_pre_list = [x for x in upload_duration_pre_list if x is not None]
    exit_time_list = [x for x in exit_time_list if x is not None]

    # Calculate CDF for upload duration pre
    cdf_upload_duration_pre, cdf_values_upload_pre = calculate_cdf(upload_duration_pre_list)
    cdf_exit_time, cdf_values_exit_time = calculate_cdf(exit_time_list)

    plt.figure()
    plt.plot(cdf_upload_duration_pre, cdf_values_upload_pre, marker='o', label='CDF of Upload Duration Pre')
    plt.plot(cdf_exit_time, cdf_values_exit_time, marker='o', color='m', label='CDF of Exit Time')
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Upload Duration Pre and Exit Time")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=-1)  # Set x-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_upload_duration_pre_and_exit_time.png"))
    plt.close()

# plot cdf time for speedtest upload post and put SEARCH exit time on it
if upload_duration_post_list and exit_time_list:
    # REMOVE NONE VALUES FROM BOTH LISTS
    upload_duration_post_list = [x for x in upload_duration_post_list if x is not None]
    exit_time_list = [x for x in exit_time_list if x is not None]

    # Calculate CDF for upload duration post
    cdf_upload_duration_post, cdf_values_upload_post = calculate_cdf(upload_duration_post_list)
    cdf_exit_time, cdf_values_exit_time = calculate_cdf(exit_time_list)

    plt.figure()
    plt.plot(cdf_upload_duration_post, cdf_values_upload_post, marker='o', label='CDF of Upload Duration Post')
    plt.plot(cdf_exit_time, cdf_values_exit_time, marker='o', color='m', label='CDF of Exit Time')
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Upload Duration Post and Exit Time")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=-1)  # Set x-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_upload_duration_post_and_exit_time.png"))
    plt.close()

# plot upload bits sent pre and search bit acked at exit
if upload_bits_sent_pre_list and total_bits_acked_at_exit_list:
    # remove None values from both lists
    upload_bits_sent_pre_list_Mb = [x / 1e6 for x in upload_bits_sent_pre_list if x is not None] # Convert to Mb
    total_bits_acked_at_exit_list_Mb_for_pre = [x * 8 for x in total_bits_acked_at_exit_list if x is not None] # Convert to Mb
    # Calculate CDF for upload bits sent pre
    cdf_upload_bits_sent_pre, cdf_values_upload_pre = calculate_cdf(upload_bits_sent_pre_list_Mb)
    cdf_total_bits_acked_exit, cdf_values_total_bits_acked_exit = calculate_cdf(total_bits_acked_at_exit_list_Mb_for_pre)

    plt.figure()
    plt.plot(cdf_upload_bits_sent_pre, cdf_values_upload_pre, marker='o', label='CDF of Upload Bits Sent Pre')
    plt.plot(cdf_total_bits_acked_exit, cdf_values_total_bits_acked_exit, marker='o', color='m', label='CDF of Total Bits Acked at Exit')
    plt.xlabel("Mb", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Upload Bits Sent Pre and Total Bits Acked at Exit")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=0)  # Set x-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_upload_bits_sent_pre_and_total_bits_acked_at_exit.png"))
    plt.close()

# plot upload bits sent post and search bit acked at exit
if upload_bits_sent_post_list and total_bits_acked_at_exit_list:
    # remove none values from upload_bits_sent_post_list and total_bits_acked_at_exit_list
    upload_bits_sent_post_list_Mb = [x / 1e6 for x in upload_bits_sent_post_list if x is not None] # Convert to Mb
    total_bits_acked_at_exit_list_Mb_for_post = [x * 8 for x in total_bits_acked_at_exit_list if x is not None] # Convert to Mb

    # Calculate CDF for upload bits sent post
    cdf_upload_bits_sent_post, cdf_values_upload_post = calculate_cdf(upload_bits_sent_post_list_Mb)
    cdf_total_bits_acked_exit, cdf_values_total_bits_acked_exit = calculate_cdf(total_bits_acked_at_exit_list_Mb_for_post)

    plt.figure()
    plt.plot(cdf_upload_bits_sent_post, cdf_values_upload_post, marker='o', label='CDF of Upload Bits Sent Post')
    plt.plot(cdf_total_bits_acked_exit, cdf_values_total_bits_acked_exit, marker='o', color='m', label='CDF of Total Bits Acked at Exit')
    plt.xlabel("Mb", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF of Upload Bits Sent Post and Total Bits Acked at Exit")
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=0)  # Set x-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_upload_bits_sent_post_and_total_bits_acked_at_exit.png"))
    plt.close()

# plot throughput of search and throughput of pre download on one graph
if throughput_df_pre_list and throughput_all:
    for i in range(len(throughput_df_pre_list)):
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
    for i in range(len(throughput_df_post_list)):
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
if avg_throughputs and median_throughputs and avg_delivery_rates and median_delivery_rates and delivery_rate_at_exit and exit_time_list:
    df_results = pd.DataFrame({
        "Run": range(1, num + 2),
        "Average Throughput (Mb/s)": avg_throughputs,
        "Median Throughput (Mb/s)": median_throughputs,
        "Average Delivery Rate (Mb/s)": avg_delivery_rates,
        "Median Delivery Rate (Mb/s)": median_delivery_rates,
        "Delivery Rate at Exit (Mb/s)": delivery_rate_at_exit,
        "Exit Time (s)": exit_time_list
    })
    results_csv_path = os.path.join(output_path, "speedtest_analysis_results.csv")
    df_results.to_csv(results_csv_path, index=False)
