import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import bisect
import numpy as np
import json
from natsort import natsorted
from matplotlib.ticker import FormatStrFormatter


ookla_speedtest_result_path = "/home/maryam/SEARCH/speedtest_CCalgs/14_speedtest_server_linux_desktop_starbucks_4g/speedtest_results_4g_starbucks/all_data/client/speedtest_result_ookla"
ndt7_speedtest_result_path = "/home/maryam/SEARCH/speedtest_CCalgs/14_speedtest_server_linux_desktop_starbucks_4g/speedtest_results_4g_starbucks/all_data/client/speedtest_result_ndt7"
libre_speedtest_result_path = "/home/maryam/SEARCH/speedtest_CCalgs/14_speedtest_server_linux_desktop_starbucks_4g/speedtest_results_4g_starbucks/all_data/client/speedtest_result_libre"

server_log_path = "/home/maryam/SEARCH/speedtest_CCalgs/14_speedtest_server_linux_desktop_starbucks_4g/speedtest_results_4g_starbucks/all_data/server/data/log_search"
server_iperf3_pcap_path = "/home/maryam/SEARCH/speedtest_CCalgs/14_speedtest_server_linux_desktop_starbucks_4g/speedtest_results_4g_starbucks/all_data/server/data/pcap_server"

ookla_client_pcap_path = "/home/maryam/SEARCH/speedtest_CCalgs/14_speedtest_server_linux_desktop_starbucks_4g/speedtest_results_4g_starbucks/all_data/client/speedtest_client_pcap_ookla"
ndt7_client_pcap_path = "/home/maryam/SEARCH/speedtest_CCalgs/14_speedtest_server_linux_desktop_starbucks_4g/speedtest_results_4g_starbucks/all_data/client/speedtest_client_pcap_ndt7"
libre_client_pcap_path = "/home/maryam/SEARCH/speedtest_CCalgs/14_speedtest_server_linux_desktop_starbucks_4g/speedtest_results_4g_starbucks/all_data/client/speedtest_client_pcap_libre"

output_path = "/home/maryam/SEARCH/speedtest_CCalgs/14_speedtest_server_linux_desktop_starbucks_4g/paper_result_new"
if not os.path.exists(output_path):
    os.makedirs(output_path)

THROUGHPUT_WINDOW_BINS = 0.4 # 20 ms
client_ip =  '172.20.10.4'    #"192.168.1.100"           #"192.168.1.107"
ookla_dest_port = "8080"
ndt7_dest_port = "3001"
libre_dest_port = "8989"
LIST_OF_THPUT_WND_BINS = [] # in seconds

#################################### Functions ####################################
# Extract speedtest values from a block of text ookla
def extract_speedtest_values(block):
    ping = re.search(r'Ping:\s+([\d.]+)', block)
    down = re.search(r'Download:\s+([\d.]+)', block)
    up = re.search(r'Upload:\s+([\d.]+)', block)
    return (
        float(ping.group(1)) if ping else None,
        float(down.group(1)) if down else None,
        float(up.group(1)) if up else None
    ) 
# Extract speedtest values from a block of text ndt7
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

# Extract speedtest values from a block of text LibreSpeed
def extract_speedtest_values_libre(block):
    """
    Extract download and upload speeds from a LibreSpeed text block that contains a JSON string.
    """
    try:
        data = json.loads(block)
        if isinstance(data, list) and len(data) > 0:
            record = data[0]
            download = float(record.get("download", 0))
            upload = float(record.get("upload", 0))
            return download, upload
    except Exception as e:
        print(f"Error parsing LibreSpeed txt file: {e}")
    return None, None


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

# Calculate delivery rate per fixed-length time intewrval
def calculate_delivery_rate_by_interval(bytes_acked, now, interval_len):

    if not bytes_acked or not now:
        return [], []

    # Convert to NumPy arrays for interpolation
    now = np.array(now)
    bytes_acked = np.array(bytes_acked)

    # Generate evenly spaced interval edges
    interval_start = now[0]
    interval_end = now[-1]
    interval_edges = np.arange(interval_start, interval_end, interval_len)

    # Interpolate bytes_acked at each interval edge
    interpolated_bytes = np.interp(interval_edges, now, bytes_acked)

    # Compute rate between each interval
    interval_rates = []
    interval_times = []

    for i in range(1, len(interval_edges)):
        delta_bytes = interpolated_bytes[i] - interpolated_bytes[i - 1]
        rate = delta_bytes / interval_len
        interval_rates.append(rate)
        interval_times.append(interval_edges[i])

    return interval_rates, interval_times

# Calculate delivery rate at exit based on delivery rate list values
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

# Calculate total byte acked at exit
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

# extract info from speedtest pcap files
def extract_speedtest_pcap_metrics(pcap_path, bin_width, client_ip, dst_port):
    """
    Extracts download duration, bits sent, and throughput time series from a PCAP file using tshark.
    Returns a dictionary of metrics for all files in the directory.
    """

    data = {}

    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"{pcap_path} does not exist")
    
    # sort all pcap files in the directory
    pcap_files = natsorted([f for f in os.listdir(pcap_path) if f.endswith(".pcap")])

    # iterate through each pcap file
    for i, pcap_file in enumerate(pcap_files, start=1):
        full_pcap_path = os.path.join(pcap_path, pcap_file)
        if not os.path.exists(full_pcap_path):
            raise FileNotFoundError(f"{full_pcap_path} does not exist")

        csv_path = full_pcap_path.replace(".pcap", ".csv")

        tshark_cmd = (
            f"tshark -r '{full_pcap_path}' -Y 'tcp and not tcp.port == 22' "
            f"-T fields "
            f"-e frame.time_relative "
            f"-e ip.src "
            f"-e ip.dst "
            f"-e tcp.srcport "
            f"-e tcp.dstport "
            f"-e frame.len "
            f"-E header=y -E separator=, -E quote=n -E occurrence=f > '{csv_path}'"
        )

        os.system(tshark_cmd)

        df = pd.read_csv(csv_path).dropna()
        df["frame.len"] = df["frame.len"].astype(int)
        df["frame.time_relative"] = df["frame.time_relative"].astype(float)

        # Remove SSH traffic, if it wrongly captured (either src or dst port is 22)
        df = df[~((df["tcp.srcport"] == 22) | (df["tcp.dstport"] == 22))]

        dst_port = int(dst_port)
        df = df[(df["ip.dst"] == client_ip) & (df["tcp.srcport"] == dst_port)]

        # Drop any malformed entries
        df = df.dropna(subset=["ip.src", "ip.dst", "tcp.dstport", "frame.len"])

        duration = df["frame.time_relative"].max() - df["frame.time_relative"].min()
        bytes_sent = df["frame.len"].sum()

        df["time_bin"] = (df["frame.time_relative"] // bin_width) * bin_width
        throughput_df = df.groupby("time_bin")["frame.len"].sum().reset_index()
        throughput_df["throughput_Mbps"] = (throughput_df["frame.len"] * 8) / (bin_width * 1_000_000)

        # put this info in dictionnary 
        data[i] = {
            "download_duration": duration,
            "bytes_sent": bytes_sent,
            "throughput_time_series": throughput_df[["time_bin", "throughput_Mbps"]].values.tolist()
        }

    return data

# def extract_speedtest_pcap_metrics(pcap_path, bin_width=0.02, client_ip=None, exclude_ips=["130.215.28.181"]):
#     """
#     Extracts download duration, bits sent, and throughput time series from PCAP files using tshark.

#     If client_ip is provided, selects the ip.src (server) that sends the most data to that client.
#     If not, auto-selects the top (ip.src, ip.dst) pair by traffic.

#     Parameters:
#         pcap_path (str): Path to directory containing .pcap files.
#         bin_width (float): Time bin width in seconds.
#         client_ip (str, optional): Fixed client IP address. If None, auto-detects top IP pair.
#         exclude_ips (list, optional): List of source IPs to ignore when auto-selecting server.

#     Returns:
#         dict: Mapping file index to extracted metrics.
#     """
#     data = {}
#     exclude_ips = set(exclude_ips or [])

#     if not os.path.exists(pcap_path):
#         raise FileNotFoundError(f"{pcap_path} does not exist")

#     pcap_files = natsorted([f for f in os.listdir(pcap_path) if f.endswith(".pcap")])

#     for i, pcap_file in enumerate(pcap_files, start=1):
#         full_pcap_path = os.path.join(pcap_path, pcap_file)
#         csv_path = full_pcap_path.replace(".pcap", ".csv")

#         # Extract fields using tshark
#         tshark_cmd = (
#             f"tshark -r '{full_pcap_path}' -Y 'ip and tcp' "
#             f"-T fields -e frame.time_relative -e ip.src -e ip.dst -e frame.len "
#             f"-E header=y -E separator=, -E quote=n -E occurrence=f > '{csv_path}'"
#         )
#         os.system(tshark_cmd)

#         df = pd.read_csv(csv_path).dropna()
#         df["frame.len"] = df["frame.len"].astype(int)
#         df["frame.time_relative"] = df["frame.time_relative"].astype(float)

#         if df.empty:
#             print(f"[Warning] No data in {pcap_file}")
#             continue

#         if client_ip:
#             # Filter to packets destined to fixed client
#             client_df = df[df["ip.dst"] == client_ip]
#             if client_df.empty:
#                 print(f"[Warning] No traffic to client {client_ip} in {pcap_file}")
#                 continue

#             # Remove excluded server IPs
#             client_df = client_df[~client_df["ip.src"].isin(exclude_ips)]
#             if client_df.empty:
#                 print(f"[Warning] All traffic to {client_ip} is from excluded IPs in {pcap_file}")
#                 continue

#             # Pick server IP sending most bytes to client
#             ip_counts = client_df.groupby("ip.src")["frame.len"].sum().reset_index()
#             ip_counts = ip_counts.sort_values("frame.len", ascending=False)
#             server_ip = ip_counts.iloc[0]["ip.src"]
#             print(f"[{pcap_file}] Client fixed: {client_ip}, Auto-selected Server: {server_ip}")
#         else:
#             # Auto-select top (ip.src, ip.dst) pair
#             ip_pair_counts = df.groupby(["ip.src", "ip.dst"])["frame.len"].sum().reset_index()
#             ip_pair_counts = ip_pair_counts[~ip_pair_counts["ip.src"].isin(exclude_ips)]
#             ip_pair_counts = ip_pair_counts.sort_values("frame.len", ascending=False)
#             if ip_pair_counts.empty:
#                 print(f"[Warning] No valid traffic in {pcap_file} after excluding IPs.")
#                 continue
#             top_pair = ip_pair_counts.iloc[0]
#             server_ip = top_pair["ip.src"]
#             client_ip = top_pair["ip.dst"]
#             print(f"[{pcap_file}] Auto-selected IPs → Server: {server_ip}, Client: {client_ip}")

#         # Filter download traffic
#         download_df = df[(df["ip.src"] == server_ip) & (df["ip.dst"] == client_ip)].copy()

#         if download_df.empty:
#             print(f"[Warning] No server-to-client traffic from {server_ip} to {client_ip} in {pcap_file}")
#             continue

#         duration = download_df["frame.time_relative"].max() - download_df["frame.time_relative"].min()
#         bytes_sent = download_df["frame.len"].sum() * 8

#         download_df["time_bin"] = (download_df["frame.time_relative"] // bin_width) * bin_width
#         throughput_df = download_df.groupby("time_bin")["frame.len"].sum().reset_index()
#         throughput_df["throughput_Mbps"] = (throughput_df["frame.len"] * 8) / (bin_width * 1_000_000)

#         data[i] = {
#             "pcap_file": pcap_file,
#             "server_ip": server_ip,
#             "client_ip": client_ip,
#             "download_duration": duration,
#             "bytes_sent": bytes_sent,
#             "throughput_time_series": throughput_df[["time_bin", "throughput_Mbps"]].values.tolist()
#         }

#     return data


# find avg RTT until exit over all num_log_file to set THROUGHPUT_WINDOW_BINS based on it
def find_avg_rtt_until_exit_over_all_log_files(server_log_path, num_log_files):
    avg_rtt_until_exit_list = []
    if not os.path.exists(server_log_path):
        print(f"Server log path {server_log_path} does not exist")
        return avg_rtt_until_exit_list

    for j in range(num_log_files+1):
        log_csv_file_path = os.path.join(server_log_path, f"log_data{j+1}.csv")
        if not os.path.exists(log_csv_file_path):
            print(f"File {log_csv_file_path} does not exist")
            continue

        df_log = pd.read_csv(log_csv_file_path)
        rtt_s_list = df_log["rtt_s"].tolist()
        search_exit_time = df_log["search_ex_time_s"].tolist()

        if not search_exit_time or search_exit_time[0] == 0:
            search_exit_time = None        
        else:
            search_exit_time = [search_exit_time[0]]

        if search_exit_time is not None and len(rtt_s_list) > 0:
            exit_index = bisect.bisect_left(df_log["start_time_zero_s"].tolist(), search_exit_time[0])
            if exit_index < len(rtt_s_list):
                avg_rtt_until_exit_list.append(np.mean(rtt_s_list[:exit_index]))
            else:
                avg_rtt_until_exit_list.append(None)
        else:
            avg_rtt_until_exit_list.append(None)

    return avg_rtt_until_exit_list

# calculate delivery rate at exit based on bytes acked
def calculate_delivery_rate_at_exit(exit_time, now_list, bytes_acked_list, rtt_s_list):
    """
    Calculate the delivery rate at the time of exit using interpolation.
    
    Parameters:
        exit_time (float): The time to calculate delivery rate for.
        now_list (List[float]): Timestamps of acknowledged packets.
        bytes_acked_list (List[float]): Cumulative bytes acknowledged over time.
        rtt_s_list (List[float]): RTTs corresponding to the timestamps.
        
    Returns:
        float or None: Interpolated delivery rate at exit_time, or None if not computable.
    """
    if exit_time is None or exit_time >= now_list[-1]:
        return None

    exit_time_index = bisect.bisect_left(now_list, exit_time)
    if exit_time_index >= len(now_list):
        return None

    current_byted_acked = bytes_acked_list[exit_time_index]
    time_pre_rtt = now_list[exit_time_index] - rtt_s_list[exit_time_index]

    if time_pre_rtt < 0:
        return None

    pre_rtt_index = bisect.bisect_left(now_list, time_pre_rtt)
    if pre_rtt_index < len(now_list) and pre_rtt_index > 0:
        t0, t1 = now_list[pre_rtt_index - 1], now_list[pre_rtt_index]
        b0, b1 = bytes_acked_list[pre_rtt_index - 1], bytes_acked_list[pre_rtt_index]

        if t1 == t0:
            pre_byte_acked = b0
        else:
            frac = (time_pre_rtt - t0) / (t1 - t0)
            pre_byte_acked = b0 + frac * (b1 - b0)

        delta_bytes = (current_byted_acked - pre_byte_acked) * 8 # Convert to bits
        delta_time = now_list[exit_time_index] - time_pre_rtt
        if delta_time > 0:
            return delta_bytes / delta_time
    return None
#################### IPERF3 LOG INFO SERVER_SIDE ##########################
# calculate delivery rate based on log files
avg_delivery_rates = []
median_delivery_rates = []
# delivery_rate_at_exit = []
delivery_rate_at_exit_on_acked_bytes_list = []
delivery_rate_two_pre_rtt_of_exit = []
delivery_rate_at_exit_based_on_window = []
delivery_are_at_search_exit_time = []
delivery_rate_at_non_search_exit_time = []
search_exit_time_list = []
not_search_exit_time_list = []
all_ss_exit_time_list = []
total_bytes_acked_at_exit_list = []
exit_time = None
skip_run_indices = set()
avg_rtt_until_exit_list = []

if not os.path.exists(server_log_path):
    print(f"Server log path {server_log_path} does not exist")
else:
    log_csv_files = [f for f in os.listdir(server_log_path) if f.endswith('.csv')]
    if not log_csv_files:
        print(f"No CSV files found in {server_log_path}")
        exit()  
    num_log_files = len(log_csv_files)  # Get the last file

    # find avg RTT until exit over all log files to set THROUGHPUT_WINDOW_BINS
    avg_rtt_until_exit_list = find_avg_rtt_until_exit_over_all_log_files(server_log_path, num_log_files)
    if avg_rtt_until_exit_list:
        # set THROUGHPUT_WINDOW_BINS based on avg RTT until exit
        avg_rtt_until_exit_list = [rtt for rtt in avg_rtt_until_exit_list if rtt is not None]
        THROUGHPUT_WINDOW_BINS = np.mean(avg_rtt_until_exit_list) if np.mean(avg_rtt_until_exit_list) > 0 else 0.4
        
        # Round THROUGHPUT_WINDOW_BINS to 2 decimal places
        THROUGHPUT_WINDOW_BINS = round(THROUGHPUT_WINDOW_BINS, 2)

        print(f"THROUGHPUT_WINDOW_BINS set to {THROUGHPUT_WINDOW_BINS} seconds based on avg RTT until exit")

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
        pre_acked_window_MB = df_log["prev_wind_MB"].tolist()
        curr_acked_window_MB = df_log["current_wind_MB"].tolist()
        search_times_s = df_log["search_time_s"].tolist()

        # if we have negative value in now_list, we limit the list to positive values
        # bytes_acked_list = [b for b, n in zip(bytes_acked_list, now_list) if n >= 0]
        # now_list = [n for n in now_list if n >= 0]
        # rtt_s_list = [r for r, n in zip(rtt_s_list, now_list) if n >= 0]
        # search_exit_time = [s for s, n in zip(search_exit_time, now_list) if n >= 0]
        # sstresh_list = [s for s, n in zip(sstresh_list, now_list) if n >= 0]
        # pre_acked_window_MB = [p for p, n in zip(pre_acked_window_MB, now_list) if n >= 0]
        # curr_acked_window_MB = [c for c, n in zip(curr_acked_window_MB, now_list) if n >= 0]
        # search_times_s = [s for s, n in zip(search_times_s, now_list) if n >= 0]
        
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
        type_of_exit = None
        if notsearch_ss_exit_time is not None and search_exit_time is not None:
            if notsearch_ss_exit_time < search_exit_time:
                not_search_exit_time_list.append(notsearch_ss_exit_time[0])
                all_ss_exit_time_list.append(notsearch_ss_exit_time[0])
                exit_time = notsearch_ss_exit_time[0]
                type_of_exit = "notsearch"
            else:
                notsearch_ss_exit_time = None
                search_exit_time_list.append(search_exit_time[0])
                all_ss_exit_time_list.append(search_exit_time[0])
                exit_time = search_exit_time[0]
                type_of_exit = "search"

        elif search_exit_time is not None:
            search_exit_time_list.append(search_exit_time[0])
            all_ss_exit_time_list.append(search_exit_time[0])
            exit_time = search_exit_time[0]
            type_of_exit = "search"

        elif notsearch_ss_exit_time is not None:
            not_search_exit_time_list.append(notsearch_ss_exit_time[0])
            all_ss_exit_time_list.append(notsearch_ss_exit_time[0])
            exit_time = notsearch_ss_exit_time[0]
            type_of_exit = "notsearch"

        ########################## Calculate delivery rates

        # delivery_rates_calculated, start_index_to_cal_delv_rate, time_cal_delv_rates = \
        delivery_rates_calculated_per_fixed_interval, time_cal_delv_rates = \
            calculate_delivery_rate_by_interval(bytes_acked_list, now_list, THROUGHPUT_WINDOW_BINS)
        #             calculate_delivery_rate_by_interval(bytes_acked_list, now_list, rtt_s_list[0])
            
        # convert delivery rates from MB/s to Mb/s
        delivery_rates_calculated_per_fixed_interval = [rate * 8 for rate in delivery_rates_calculated_per_fixed_interval]

        if delivery_rates_calculated_per_fixed_interval is not None:
            avg_delivery_rates.append(np.average(delivery_rates_calculated_per_fixed_interval))
            median_delivery_rates.append(np.median(delivery_rates_calculated_per_fixed_interval))
            
        #     # Find the delivery rate at the exit time
        #     if exit_time is not None and time_cal_delv_rates is not None:
        #         if exit_time < time_cal_delv_rates[-1]:
        #             delivery_rate_exit_ = rate_at_exit(exit_time, time_cal_delv_rates, delivery_rates_calculated_per_fixed_interval)
        #             delivery_rate_two_pre_rtt_of_exit_ = rate_at_exit(exit_time - 2 * rtt_s_list[0], time_cal_delv_rates, delivery_rates_calculated_per_fixed_interval)
        #             delivery_rate_at_exit.append(delivery_rate_exit_)
        #             delivery_rate_two_pre_rtt_of_exit.append(delivery_rate_two_pre_rtt_of_exit_)
        #         else:
        #             delivery_rate_at_exit.append(None)
        #             delivery_rate_two_pre_rtt_of_exit.append(None)
        #     else:
        #         delivery_rate_at_exit.append(None)
        #         delivery_rate_two_pre_rtt_of_exit.append(None)
        # else:
        #     delivery_rate_at_exit.append(None)
        #     delivery_rate_two_pre_rtt_of_exit.append(None)

        # Calculate delivery rate at exit time with bytes acked
        if exit_time is not None:
            delivery_rate = calculate_delivery_rate_at_exit(exit_time,now_list, bytes_acked_list,rtt_s_list) #Mb/s
            if delivery_rate is not None:
                delivery_rate_at_exit_on_acked_bytes_list.append(delivery_rate)
                if type_of_exit == "search":
                    delivery_are_at_search_exit_time.append(delivery_rate)
                elif type_of_exit == "notsearch":
                    delivery_rate_at_non_search_exit_time.append(delivery_rate)
            else:
                delivery_rate_at_exit_on_acked_bytes_list.append(None)
                delivery_are_at_search_exit_time.append(None)
                delivery_rate_at_non_search_exit_time.append(None)
        else:
            delivery_rate_at_exit_on_acked_bytes_list.append(None)
            delivery_are_at_search_exit_time.append(None)
            delivery_rate_at_non_search_exit_time.append(None)

        # Calculate delivery rate at 2 rtt before exit time
        if exit_time is not None and rtt_s_list:
            two_rtt_before_exit_time = exit_time - 2 * rtt_s_list[0]
            if two_rtt_before_exit_time >= 0:
                delivery_rate_two_pre_rtt_of_exit_ = calculate_delivery_rate_at_exit(two_rtt_before_exit_time, now_list, bytes_acked_list, rtt_s_list) #Mb/s
                delivery_rate_two_pre_rtt_of_exit.append(delivery_rate_two_pre_rtt_of_exit_)
            else:
                delivery_rate_two_pre_rtt_of_exit.append(None)

        # Calculate delivery rate at exit based on window
        if search_exit_time is not None and pre_acked_window_MB and curr_acked_window_MB:    
            # find the index where exit_time is equal with now_list
            if isinstance(search_exit_time, list):
                search_exit_time = search_exit_time[0]

            exit_time_index = bisect.bisect_left(search_times_s, search_exit_time)
            time_on_now_in_exit_index = bisect.bisect_left(now_list, search_exit_time)

            if time_on_now_in_exit_index < len(now_list):
                rtt_at_exit = rtt_s_list[time_on_now_in_exit_index]
            
            if exit_time_index < len(pre_acked_window_MB) and rtt_at_exit > 0:
                # Calculate the delivery rate at exit based on window
                if exit_time_index > 0:
                    delivery_rate_at_exit_based_on_window_ = (curr_acked_window_MB[exit_time_index] - pre_acked_window_MB[exit_time_index]) * 8/ rtt_at_exit

                delivery_rate_at_exit_based_on_window.append(delivery_rate_at_exit_based_on_window_)
        else:
            delivery_rate_at_exit_based_on_window.append(None)          
    
        # find total bytes acked at the exit time
        if exit_time is not None and now_list:
            if exit_time < now_list[-1]:
                total_bytes_acked_at_exit = total_byte_acked_at_exit(exit_time, now_list, bytes_acked_list)
            else:
                total_bytes_acked_at_exit = None
        else:
            total_bytes_acked_at_exit = None

        total_bytes_acked_at_exit_list.append(total_bytes_acked_at_exit if total_bytes_acked_at_exit is not None else None)

        # skip runs if exit time is None
        if exit_time is None:
            skip_run_indices.add(j + 1)

print("Log processing is done.")
######################## IPERF3 PCAP INFO SERVER SIDE ##########################
# calculate throughput from pcap files on server
avg_throughputs = []
median_throughputs = []
throughput_all = {}
time_throughput_all = {}
throughput_for_diff_windows = {}
time_related_diff_windows = {}
avg_throughputs_diff_windows = {}
median_throughputs_diff_windows = {}

if LIST_OF_THPUT_WND_BINS:
    for wnd_bin in LIST_OF_THPUT_WND_BINS:
        throughput_for_diff_windows[wnd_bin] = {}
        time_related_diff_windows[wnd_bin] = {}
        avg_throughputs_diff_windows[wnd_bin] = {}
        median_throughputs_diff_windows[wnd_bin] = {}

if not os.path.exists(server_iperf3_pcap_path):
    print(f"Server pcap path {server_iperf3_pcap_path} does not exist")
else:
    SERVER_IP = "130.215.28.249"
    INTERVAL = THROUGHPUT_WINDOW_BINS
    # find number of csv files in server_pcap_path
    csv_files = [f for f in os.listdir(server_iperf3_pcap_path) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {server_iperf3_pcap_path}")
        exit()

    num = len(csv_files)  # Get the last file

    for i in range(num+1):

        # if i + 1 in skip_run_indices:
        #     continue

        throughputs = []
        timestamps_thput = []

        pcap_csv_file_path = os.path.join(server_iperf3_pcap_path, f"tcp_run_{i+1}.csv")
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

        # # plot throughput over time
        # throughput_fig_path = os.path.join(output_path, "Throughput_over_time")
        # # make this path if it does not exist
        # if not os.path.exists(throughput_fig_path):
        #     os.makedirs(throughput_fig_path)
            
        # plt.figure(figsize=(10, 5))
        # plt.plot(timestamps_thput, throughputs, marker='o', label=f'Run {i+1}')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Throughput (Mb/s)')
        # plt.title(f'Throughput Over Time for Run {i+1}')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(os.path.join(throughput_fig_path, f"throughput_run_{i+1}.png"))
        # plt.close()

    
        if LIST_OF_THPUT_WND_BINS:
            for wnd_bin in LIST_OF_THPUT_WND_BINS:
                start_time_new = df_valid["Time"].iloc[0]
                end_time_new = start_time_new + wnd_bin 
                throughputs_new = []
                timestamps_thput_new = [] 

                # Compute throughput in fixed intervals
                while end_time_new <= df_valid["Time"].iloc[-1]:
                    window_data = df_valid.loc[(df_valid["Time"] >= start_time_new) & (df_valid["Time"] < end_time_new)]
                    if not window_data.empty:
                        total_bytes = window_data["Length"].sum() * 8 * 1e-6
                        throughput_new = total_bytes / wnd_bin
                        throughputs_new.append(throughput_new)
                        timestamps_thput_new.append(end_time_new)

                    # Move to next window
                    start_time_new = end_time_new
                    end_time_new = start_time_new + wnd_bin 

                    # save throughput, avg, median in dictionary and time 
                if throughputs_new:
                    throughput_for_diff_windows[wnd_bin][i+1] = throughputs_new
                    time_related_diff_windows[wnd_bin][i+1] = timestamps_thput_new
                    avg_throughputs_diff_windows[wnd_bin][i+1] = np.mean(throughputs_new)
                    median_throughputs_diff_windows[wnd_bin][i+1] = np.median(throughputs_new)
                else:
                    throughput_for_diff_windows[wnd_bin][i+1] = None
                    time_related_diff_windows[wnd_bin][i+1] = None
                    avg_throughputs_diff_windows[wnd_bin] = None
                    median_throughputs_diff_windows[wnd_bin] = None

print("PCAP processing is done.")
##################### Speedtest output client side ##########################
ookla_data = []
files = sorted(os.listdir(ookla_speedtest_result_path))
sorted_files = sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else 0)

for i, filename in enumerate(sorted_files, start=1):
    if not filename.endswith(".txt"):
        continue

    file_path = os.path.join(ookla_speedtest_result_path, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()


    pre_speed = extract_speedtest_values(content)
    post_speed = extract_speedtest_values(content)
    ookla_data.append({
        "Run": i,
        "Download": pre_speed[1],
        "Upload": pre_speed[2],
    })        


ookla_df_client = pd.DataFrame(ookla_data)

# Save table to CSV
ookla_csv_path = os.path.join(output_path, "ookla_speedtest_results.csv")
ookla_df_client.to_csv(ookla_csv_path, index=False)

print("Speedtest results for ookla processing is done.")

ndt7_data = []
files = sorted(os.listdir(ndt7_speedtest_result_path))
sorted_files = sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else 0)

for i, filename in enumerate(sorted_files, start=1):
    if not filename.endswith(".txt"):
        continue

    file_path = os.path.join(ndt7_speedtest_result_path, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    download_speed = extract_speedtest_values_ndt7(content)
    run_data = {
        "Run": i,
        "Download": download_speed[0],
        "Upload": None,
    }
    ndt7_data.append(run_data)

ndt7_df_client = pd.DataFrame(ndt7_data)
# Save table to CSV
ndt7_csv_path = os.path.join(output_path, "ndt7_speedtest_results.csv")
ndt7_df_client.to_csv(ndt7_csv_path, index=False)

print("Speedtest results for ndt7 processing is done.")

libre_data = []
files = sorted(os.listdir(libre_speedtest_result_path))
sorted_files = sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else 0)

for i, filename in enumerate(sorted_files, start=1):
    if not filename.endswith(".txt"):
        continue

    file_path = os.path.join(libre_speedtest_result_path, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    down_speed, up_speed = extract_speedtest_values_libre(content)
    run_data = {
        "Run": i,
        "Download": down_speed,
        "Upload": up_speed,
    }
    libre_data.append(run_data)

# Save to CSV
libre_df_client = pd.DataFrame(libre_data)
libre_csv_path = os.path.join(output_path, "librespeed_results.csv")
libre_df_client.to_csv(libre_csv_path, index=False)

print("Speedtest results for librespeed processing is done.")
#################### Speedtest pcap info client side ##########################
window_bins = THROUGHPUT_WINDOW_BINS

# === OOKLA ===
ookla_metrics = extract_speedtest_pcap_metrics(ookla_client_pcap_path, window_bins, client_ip, ookla_dest_port)
print("OOKLA metrics extracted from PCAP files.")

# === NDT7 ===
ndt7_metrics = extract_speedtest_pcap_metrics(ndt7_client_pcap_path, window_bins, client_ip, ndt7_dest_port)
print("NDT7 metrics extracted from PCAP files.")

# === LIBRE ===
libre_metrics = extract_speedtest_pcap_metrics(libre_client_pcap_path, window_bins, client_ip, libre_dest_port)
print("LibreSpeed metrics extracted from PCAP files.")

#################################### PLOT ####################################
if median_throughputs:
    cdf_median_throughputs, cdf_values_median_throughputs = calculate_cdf(median_throughputs)

    # plot cdf delivery rate at exit and cdf delivery rate at two RTTs before exit and delivery rate 
    # at exit_bytes_acked and median throughput on one graph
    # remove non values from delivery_rate_two_pre_rtt_of_exit 
    delivery_rate_two_pre_rtt_of_exit_list = [rate for rate in delivery_rate_two_pre_rtt_of_exit if rate is not None]
    if delivery_rate_two_pre_rtt_of_exit_list and delivery_rate_at_exit_on_acked_bytes_list:
        cdf_delivery_rate_two_pre_rtt_of_exit, cdf_values_delivery_rate_two_pre_rtt_of_exit = calculate_cdf(delivery_rate_two_pre_rtt_of_exit_list)
        #
        #  remove none values of delivery_rate_at_exit_on_acked_bytes_list
        delivery_rate_at_exit_on_acked_bytes_list_ = [rate for rate in delivery_rate_at_exit_on_acked_bytes_list if rate is not None]
        cdf_delivery_rate_at_exit_on_acked_bytes, cdf_values_delivery_rate_at_exit_on_acked_bytes = calculate_cdf(delivery_rate_at_exit_on_acked_bytes_list_)

        plt.figure(figsize=(8, 5))
        plt.plot(cdf_delivery_rate_two_pre_rtt_of_exit, cdf_values_delivery_rate_two_pre_rtt_of_exit, marker='o', color='orange', label=' DR Two RTTs Before Exit')
        plt.plot(cdf_median_throughputs, cdf_values_median_throughputs, marker='o', label=' Median Throughput')
        plt.plot(cdf_delivery_rate_at_exit_on_acked_bytes, cdf_values_delivery_rate_at_exit_on_acked_bytes, marker='o', color='green', label=' DR at Exit')
        plt.xlabel("Rate (Mb/s)", fontsize=19)
        plt.ylabel("Cumulative Distribution", fontsize=19)
        # plt.title(" Delivery Rate at Exit and Two RTTs Before Exit")
        plt.legend(fontsize=16)
        # plt.xlim(left=-0.5)  # Set x-axis to start from 0
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "cdf_delivery_rates_and_median_throughput.png"))
        plt.close()

        # plot cdf median throughput and cdf delivery rate at two RTTs before exit on one graph
        plt.figure(figsize=(8, 5))
        plt.plot(cdf_median_throughputs, cdf_values_median_throughputs, marker='o', color='b', label=' Median Throughput')
        plt.plot(cdf_delivery_rate_two_pre_rtt_of_exit, cdf_values_delivery_rate_two_pre_rtt_of_exit, marker='o', color='m', label=' DR Two RTTs Before Exit')
        plt.xlabel("Throughput (Mb/s)", fontsize=19)
        plt.ylabel("Cumulative Distribution", fontsize=19)
        # plt.title("CDF of Median Throughput and DR Two RTTs Before Exit")
        plt.legend(fontsize=16)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(left=-0.5)  # Set x-axis to start from 0
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "cdf_median_throughput_and_delivery_rate_two_rtts_before_exit.png"))
        plt.close()

        # plot cdf median throughput and cdf delivery rate at exit (bytes acked) on one graph
        plt.figure(figsize=(8, 5))
        plt.plot(cdf_median_throughputs, cdf_values_median_throughputs, marker='o', color='b', label=' Median Throughput')
        plt.plot(cdf_delivery_rate_at_exit_on_acked_bytes, cdf_values_delivery_rate_at_exit_on_acked_bytes, marker='o', color='m', label=' DR at Exit')
        plt.xlabel("Throughput (Mb/s)", fontsize=19)
        plt.ylabel("Cumulative Distribution", fontsize=19)
        # plt.title("CDF of Median Throughput and Delivery Rate at Exit")
        plt.legend(fontsize=16)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(left=-0.5)  # Set x-axis to start from 0
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "cdf_median_throughput_and_delivery_rate_at_exit_bytes_acked.png"))
        plt.close()

# plot cdf diff between median throughput and delivery rate at two RTTs before exit and cdf norm diff
if median_throughputs and delivery_rate_two_pre_rtt_of_exit:

    diff_median_throughput_delivery_rate_two_rtt = [
        (median - delivery_rate) if delivery_rate is not None else None
        for median, delivery_rate in zip(median_throughputs, delivery_rate_two_pre_rtt_of_exit)
    ]

    # remove non values from diff_median_throughput_delivery_rate_two_rtt
    diff_median_throughput_delivery_rate_two_rtt = [diff for diff in diff_median_throughput_delivery_rate_two_rtt if diff is not None]
    cdf_diff_median_throughput_delivery_rate_two_rtt, cdf_values_diff_median_throughput_delivery_rate_two_rtt = calculate_cdf(diff_median_throughput_delivery_rate_two_rtt)

    plt.figure(figsize=(8, 5))
    # plot vertical dashed line on zero
    plt.axvline(x=0, color='gray', linestyle='--', label='Zero Line')
    plt.plot(cdf_diff_median_throughput_delivery_rate_two_rtt, cdf_values_diff_median_throughput_delivery_rate_two_rtt, marker='o', color='purple', label='CDF of Diff (Median Throughput - DR Two RTTs Before Exit)')
    plt.xlabel("Difference (Mb/s)", fontsize=19)
    plt.ylabel("Cumulative Distribution", fontsize=19)
    # plt.title("CDF of Difference between Median Throughput and Delivery Rate Two RTTs Before Exit")
    # plt.legend(fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.xlim(left=-0.5)  # Set x-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_diff_median_throughput_and_delivery_rate_two_rtts_before_exit.png"))
    plt.close()

    # normalized diff
    norm_diff_median_throughput_dr_two_pre_rtt_exit = [
        (median - delivery_rate) / median if median is not None and delivery_rate is not None else None
        for median, delivery_rate in zip(median_throughputs, delivery_rate_two_pre_rtt_of_exit)
    ]

    norm_diff_median_throughput_dr_two_pre_rtt_exit = [diff for diff in norm_diff_median_throughput_dr_two_pre_rtt_exit if diff is not None]
    cdf_norm_diff_median_throughput_dr_two_pre_rtt_exit, cdf_values_norm_diff_median_throughput_dr_two_pre_rtt_exit = calculate_cdf(norm_diff_median_throughput_dr_two_pre_rtt_exit)    

    plt.figure(figsize=(8, 5))
    # plot vertical dashed line on zero
    plt.axvline(x=0, color='gray', linestyle='--', label='Zero Line')
    plt.plot(cdf_norm_diff_median_throughput_dr_two_pre_rtt_exit, cdf_values_norm_diff_median_throughput_dr_two_pre_rtt_exit, marker='o', color='purple', label='CDF of Norm Diff (Median Throughput - DR Two RTTs Before Exit)')
    plt.xlabel("Normalized Difference", fontsize=19)
    plt.ylabel("Cumulative Distribution", fontsize=19)  
    # plt.title("CDF of Normalized Difference between Median Throughput and Delivery Rate Two RTTs Before Exit")
    # plt.legend(fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # Set x-axis tick format to one decimal
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # plt.xlim(left=-0.5)  # Set x-axis to start from
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_norm_diff_median_throughput_and delivery_rate_two_rtts_before_exit.png"))
    plt.close()

# plot cdf diff between median throughput and delivery rate at exit (bytes acked) and cdf norm diff
if median_throughputs and delivery_rate_at_exit_on_acked_bytes_list:
    
    diff_median_throughput_delivery_rate_exit_bytes = [
        (median - delivery_rate) if delivery_rate is not None else None
        for median, delivery_rate in zip(median_throughputs, delivery_rate_at_exit_on_acked_bytes_list)
    ]

    diff_median_throughput_delivery_rate_exit_bytes = [diff for diff in diff_median_throughput_delivery_rate_exit_bytes if diff is not None]
    cdf_diff_median_throughput_delivery_rate_exit_bytes, cdf_values_diff_median_throughput_delivery_rate_exit_bytes = calculate_cdf(diff_median_throughput_delivery_rate_exit_bytes)

    plt.figure(figsize=(8, 5))
    # plot vertical dashed line on zero
    plt.axvline(x=0, color='gray', linestyle='--', label='Zero Line')
    plt.plot(cdf_diff_median_throughput_delivery_rate_exit_bytes, cdf_values_diff_median_throughput_delivery_rate_exit_bytes, marker='o', color='purple', label='CDF of Diff (Median Throughput - DR at Exit Bytes Acked)')
    plt.xlabel("Difference (Mb/s)", fontsize=19)
    plt.ylabel("Cumulative Distribution", fontsize=19)
    # plt.title("CDF of Normalized Difference between Median Throughput and Delivery Rate at Exit Bytes Acked")
    # plt.legend(fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.xlim(left=-0.5)  # Set x-axis to start from
    plt.tight_layout()
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.savefig(os.path.join(output_path, "cdf_diff_median_throughput_and_delivery_rate_at_exit_bytes_acked.png"))
    plt.close() 

    # normalized diff
    norm_diff_median_throughput_dr_exit_bytes = [
        (median - delivery_rate) / median if median is not None and delivery_rate is not None else None
        for median, delivery_rate in zip(median_throughputs, delivery_rate_at_exit_on_acked_bytes_list)
    ]

    norm_diff_median_throughput_dr_exit_bytes = [diff for diff in norm_diff_median_throughput_dr_exit_bytes if diff is not None]
    cdf_norm_diff_median_throughput_dr_exit_bytes, cdf_values_norm_diff_median_throughput_dr_exit_bytes = calculate_cdf(norm_diff_median_throughput_dr_exit_bytes)  

    plt.figure(figsize=(8, 5))
    # plot vertical dashed line on zero
    plt.axvline(x=0, color='gray', linestyle='--', label='Zero Line')
    plt.plot(cdf_norm_diff_median_throughput_dr_exit_bytes, cdf_values_norm_diff_median_throughput_dr_exit_bytes, marker='o', color='purple', label='CDF of Norm Diff (Median Throughput - DR at Exit Bytes Acked)')
    plt.xlabel("Normalized Difference", fontsize=19)
    plt.ylabel("Cumulative Distribution", fontsize=19)
    # plt.title("CDF of Normalized Difference between Median Throughput and Delivery Rate at Exit Bytes Acked")
    # plt.legend(fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.xlim(left=-0.5)  # Set x-axis to start from
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_norm_diff_median_throughput_and_delivery_rate_at_exit_bytes_acked.png"))
    plt.close() 


# plot cdf diff download speed for all tools with delivery rate at exit and norm diff
if ookla_data and ndt7_data and libre_data and delivery_rate_at_exit_on_acked_bytes_list:
    diff_ookla_ndt7_libre_delivery_rate_exit = []
    for i in range(len(ookla_data)):
        ookla_speed = ookla_data[i]["Download"]
        ndt7_speed = ndt7_data[i]["Download"]
        libre_speed = libre_data[i]["Download"]
        if i < len(delivery_rate_at_exit_on_acked_bytes_list) and delivery_rate_at_exit_on_acked_bytes_list[i] is not None:
            delivery_rate_exit = delivery_rate_at_exit_on_acked_bytes_list[i]
        else:
            delivery_rate_exit = None

        if delivery_rate_exit is not None:
            diff_ookla_ndt7_libre_delivery_rate_exit.append(
                (ookla_speed - delivery_rate_exit, ndt7_speed - delivery_rate_exit, libre_speed - delivery_rate_exit)
            )
        else:
            diff_ookla_ndt7_libre_delivery_rate_exit.append((None, None, None))
    
    # remove None values from diff_ookla_ndt7_libre_delivery_rate_exit
    diff_ookla_ndt7_libre_delivery_rate_exit_list = [d for d in diff_ookla_ndt7_libre_delivery_rate_exit if all(x is not None for x in d)]

    cdf_diff_ookla, cdf_values_diff_ookla = calculate_cdf([d[0] for d in diff_ookla_ndt7_libre_delivery_rate_exit_list])
    cdf_diff_ndt7, cdf_values_diff_ndt7 = calculate_cdf([d[1] for d in diff_ookla_ndt7_libre_delivery_rate_exit_list])
    cdf_diff_libre, cdf_values_diff_libre = calculate_cdf([d[2] for d in diff_ookla_ndt7_libre_delivery_rate_exit_list])

    plt.figure(figsize=(8, 5))
    # plot vertical dashed line on zero
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.plot(cdf_diff_ookla, cdf_values_diff_ookla, marker='o', color='blue', label='(Ookla - DR at Exit)')
    plt.plot(cdf_diff_ndt7, cdf_values_diff_ndt7, marker='o', color='orange', label='(NDT7 - DR at Exit)')
    plt.plot(cdf_diff_libre, cdf_values_diff_libre, marker='o', color='green', label='(Libre - DR at Exit)')
    plt.xlabel("Difference (Mb/s)", fontsize=19)
    plt.ylabel("Cumulative Distribution", fontsize=19)
    # plt.title("CDF of Difference between Download Speeds and Delivery Rate at Exit")
    plt.legend(fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.xlim(left=-0.5)  # Set x-axis to start from 0
    plt.tight_layout()  
    plt.savefig(os.path.join(output_path, "cdf_diff_download_speeds_and_delivery_rate_at_exit.png"))
    plt.close()

    # plot cdf norm diff download speed for all tools with delivery rate at exit
    norm_diff_ookla = [(d[0] / ookla_data[i]["Download"]) if d[0] is not None and ookla_data[i]["Download"] else None for i, d in enumerate(diff_ookla_ndt7_libre_delivery_rate_exit)]
    norm_diff_ndt7 = [(d[1] / ndt7_data[i]["Download"]) if d[1] is not None and ndt7_data[i]["Download"] else None for i, d in enumerate(diff_ookla_ndt7_libre_delivery_rate_exit)]
    norm_diff_libre = [(d[2] / libre_data[i]["Download"]) if d[2] is not None and libre_data[i]["Download"] else None for i, d in enumerate(diff_ookla_ndt7_libre_delivery_rate_exit)]   

    # remove None values from norm_diff_ookla, norm_diff_ndt7, norm_diff_libre
    norm_diff_ookla = [d for d in norm_diff_ookla if d is not None]
    norm_diff_ndt7 = [d for d in norm_diff_ndt7 if d is not None]
    norm_diff_libre = [d for d in norm_diff_libre if d is not None]

    cdf_norm_diff_ookla, cdf_values_norm_diff_ookla = calculate_cdf(norm_diff_ookla)
    cdf_norm_diff_ndt7, cdf_values_norm_diff_ndt7 = calculate_cdf(norm_diff_ndt7)
    cdf_norm_diff_libre, cdf_values_norm_diff_libre = calculate_cdf(norm_diff_libre)        

    plt.figure(figsize=(8, 5))
    # plot vertical dashed line on zero
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.plot(cdf_norm_diff_ookla, cdf_values_norm_diff_ookla, marker='o', color='blue', label='norm diff(Ookla, DR at Exit)')
    plt.plot(cdf_norm_diff_ndt7, cdf_values_norm_diff_ndt7, marker='o', color='orange', label='norm diff(NDT7, DR at Exit)')
    plt.plot(cdf_norm_diff_libre, cdf_values_norm_diff_libre, marker='o', color='green', label='norm diff(Libre,  DR at Exit)')
    plt.xlabel("Normalized Difference", fontsize=19)
    plt.ylabel("Cumulative Distribution", fontsize=19)
    # plt.title("CDF of Normalized Difference between Download Speeds and Delivery Rate at Exit")
    plt.legend(fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.xlim(left=-0.5)  # Set x-axis to start from
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_norm_diff_download_speeds_and_delivery_rate_at_exit_with_ookla.png"))
    plt.close()

# plot cdf norm diff median_thropughput and tools speedtest on one graph
if median_throughputs and ookla_data and ndt7_data and libre_data:
    ookla_speed = [d["Download"] for d in ookla_data]
    ndt7_speed = [d["Download"] for d in ndt7_data]
    libre_speed = [d["Download"] for d in libre_data]

    norm_diff_ookla = [(median - speed) / median if median is not None and speed is not None else None for median, speed in zip(median_throughputs, ookla_speed)]
    norm_diff_ndt7 = [(median - speed) / median if median is not None and speed is not None else None for median, speed in zip(median_throughputs, ndt7_speed)]
    norm_diff_libre = [(median - speed) / median if median is not None and speed is not None else None for median, speed in zip(median_throughputs, libre_speed)]

    # remove None values from norm_diff_ookla, norm_diff_ndt7, norm_diff_libre
    norm_diff_ookla = [d for d in norm_diff_ookla if d is not None]
    norm_diff_ndt7 = [d for d in norm_diff_ndt7 if d is not None]
    norm_diff_libre = [d for d in norm_diff_libre if d is not None]

    cdf_norm_diff_ookla, cdf_values_norm_diff_ookla = calculate_cdf(norm_diff_ookla)
    cdf_norm_diff_ndt7, cdf_values_norm_diff_ndt7 = calculate_cdf(norm_diff_ndt7)
    cdf_norm_diff_libre, cdf_values_norm_diff_libre = calculate_cdf(norm_diff_libre)

    norm_diff_median_throughput_dr_exit_bytes = [
        (median - delivery_rate) / median if median is not None and delivery_rate is not None else None
        for median, delivery_rate in zip(median_throughputs, delivery_rate_at_exit_on_acked_bytes_list)
    ]

    norm_diff_median_throughput_dr_exit_bytes = [diff for diff in norm_diff_median_throughput_dr_exit_bytes if diff is not None]

    # remove norm_diff_median_throughput_dr_exit_bytes less than -2
    norm_diff_median_throughput_dr_exit_bytes = [diff for diff in norm_diff_median_throughput_dr_exit_bytes if diff > -2]
    cdf_norm_diff_median_throughput_dr_exit_bytes, cdf_values_norm_diff_median_throughput_dr_exit_bytes = calculate_cdf(norm_diff_median_throughput_dr_exit_bytes)  

    

    plt.figure(figsize=(8, 5))
    # plot vertical dashed line on zero
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.plot(cdf_norm_diff_median_throughput_dr_exit_bytes, cdf_values_norm_diff_median_throughput_dr_exit_bytes, marker='o', color='purple', label='SEARCH')
    plt.plot(cdf_norm_diff_ookla, cdf_values_norm_diff_ookla, marker='*', color='blue', label='Ookla')
    plt.plot(cdf_norm_diff_ndt7, cdf_values_norm_diff_ndt7, marker='s', color='darkorange', label='NDT7')
    plt.plot(cdf_norm_diff_libre, cdf_values_norm_diff_libre, marker='^', color='green', label='Libre')
    plt.xlabel("Normalized Difference", fontsize=19)
    plt.ylabel("Cumulative Distribution", fontsize=19)
    # plt.title("CDF of Normalized Difference between Median Throughput and Speedtest Tools")
    plt.legend(fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.xlim(left=-0.5)  # Set x-axis to start from
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_norm_diff_median_throughput_and_speedtest_tools.png"))
    plt.close()

# make the above figure, but with absolute values
absolute_norm_diff_ookla = [abs(d) for d in norm_diff_ookla]
absolute_norm_diff_ndt7 = [abs(d) for d in norm_diff_ndt7]
absolute_norm_diff_libre = [abs(d) for d in norm_diff_libre]
absolute_norm_diff_median_throughput_dr_exit_bytes = [abs(d) for d in norm_diff_median_throughput_dr_exit_bytes]

cdf_abs_norm_diff_ookla, cdf_values_abs_norm_diff_ookla = calculate_cdf(absolute_norm_diff_ookla)
cdf_abs_norm_diff_ndt7, cdf_values_abs_norm_diff_ndt7 = calculate_cdf(absolute_norm_diff_ndt7)
cdf_abs_norm_diff_libre, cdf_values_abs_norm_diff_libre = calculate_cdf(absolute_norm_diff_libre)
cdf_abs_norm_diff_search, cdf_values_abs_norm_diff_search = calculate_cdf(absolute_norm_diff_median_throughput_dr_exit_bytes)

plt.figure(figsize=(8, 5))
# plot vertical dashed line on zero
plt.axvline(x=0, color='gray', linestyle='--')
plt.plot(cdf_abs_norm_diff_search, cdf_values_abs_norm_diff_search, marker='o', color='purple', label='SEARCH')
plt.plot(cdf_abs_norm_diff_ookla, cdf_values_abs_norm_diff_ookla, marker='*', color='blue', label='Ookla')
plt.plot(cdf_abs_norm_diff_ndt7, cdf_values_abs_norm_diff_ndt7, marker='s', color='darkorange', label='NDT7')   
plt.plot(cdf_abs_norm_diff_libre, cdf_values_abs_norm_diff_libre, marker='^', color='green', label='Libre')
plt.xlabel("Absolute Normalized Difference", fontsize=19)
plt.ylabel("Cumulative Distribution", fontsize=19)
# plt.title("CDF of Absolute Normalized Difference between Median Throughput and Speedtest Tools")
plt.legend(fontsize=16)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(output_path, "cdf_abs_norm_diff_median_throughput_and_speedtest_tools.png"))
plt.close()


# plot cdf all tools download durations and cdf all ss exit 
if ookla_metrics and ndt7_metrics and libre_metrics and all_ss_exit_time_list:

    ookla_dwnd_duration = [v["download_duration"] for v in ookla_metrics.values() if v["download_duration"] is not None]
    ndt7_dwnd_duration = [v["download_duration"] for v in ndt7_metrics.values() if v["download_duration"] is not None]
    libre_dwnd_duration = [v["download_duration"] for v in libre_metrics.values() if v["download_duration"] is not None]

    cdf_ookla_dwnd_duration, cdf_values_ookla_dwnd_duration = calculate_cdf(ookla_dwnd_duration)
    cdf_ndt7_dwnd_duration, cdf_values_ndt7_dwnd_duration = calculate_cdf(ndt7_dwnd_duration)
    cdf_libre_dwnd_duration, cdf_values_libre_dwnd_duration = calculate_cdf(libre_dwnd_duration)
    cdf_all_ss_exit_time, cdf_values_all_ss_exit_time = calculate_cdf(all_ss_exit_time_list)

    plt.figure(figsize=(8, 5))
    plt.plot(cdf_all_ss_exit_time, cdf_values_all_ss_exit_time, marker='o', color='purple', label='SEARCH')
    plt.plot(cdf_ookla_dwnd_duration, cdf_values_ookla_dwnd_duration, marker='*', color='blue', label='Ookla')
    plt.plot(cdf_ndt7_dwnd_duration, cdf_values_ndt7_dwnd_duration, marker='s', color='darkorange', label='NDT7')
    plt.plot(cdf_libre_dwnd_duration, cdf_values_libre_dwnd_duration, marker='^', color='green', label='Libre')

    plt.xlabel("Duration (seconds)", fontsize=19)
    plt.ylabel("Cumulative Distribution", fontsize=19)
    # plt.title("CDF of Download Durations and All SS Exit Time")   
    plt.legend(fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(left=-0.5)  # Set x-axis to start from
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_download_durations_and_all_ss_exit_time.png"))
    plt.close()


# cdf download bits sent and total bits acked at exit on one graph

ookla_byte_sent = np.asarray([v["bytes_sent"] for v in ookla_metrics.values() if v["bytes_sent"] is not None]) * 1e-6  # Convert to MB
ndt7_byte_sent = np.asarray([v["bytes_sent"] for v in ndt7_metrics.values() if v["bytes_sent"] is not None]) * 1e-6  # Convert to MB
libre_byte_sent = np.asarray([v["bytes_sent"] for v in libre_metrics.values() if v["bytes_sent"] is not None]) * 1e-6  # Convert to MB
total_bytes_acked_at_exit_list = np.asarray(total_bytes_acked_at_exit_list) * 1e-6  # Convert to MB

cdf_ookla_byte_sent, cdf_values_ookla_byte_sent = calculate_cdf(ookla_byte_sent)
cdf_ndt7_byte_sent, cdf_values_ndt7_byte_sent = calculate_cdf(ndt7_byte_sent)
cdf_libre_byte_sent, cdf_values_libre_byte_sent = calculate_cdf(libre_byte_sent)
cdf_all_ss_exit_byte_acked, cdf_values_all_ss_exit_byte_acked = calculate_cdf(total_bytes_acked_at_exit_list)

plt.figure(figsize=(8, 5))
plt.plot(cdf_all_ss_exit_byte_acked, cdf_values_all_ss_exit_byte_acked, marker='o', color='purple', label='SEARCH')
plt.plot(cdf_ookla_byte_sent, cdf_values_ookla_byte_sent, marker='*', color='blue', label='Ookla')
plt.plot(cdf_ndt7_byte_sent, cdf_values_ndt7_byte_sent, marker='s', color='orange', label='NDT7')
plt.plot(cdf_libre_byte_sent, cdf_values_libre_byte_sent, marker='^', color='green', label='Libre')
plt.xlabel("Bytes Sent (MB)", fontsize=19) 
plt.ylabel("Cumulative Distribution", fontsize=19)
# plt.title("CDF of Download Bits Sent and Total Bits Acked at Exit")
plt.legend(fontsize=16)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.xlim(left=-0.5)  # Set x-axis to start from
plt.tight_layout()
plt.savefig(os.path.join(output_path, "cdf_download_byte_sent_and_all_ss_exit_byte_acked_new.png"))
plt.close()

# plot avg duration for each tools on x_axis and avg error (median - speed) for that tool on y-axis
if ookla_data and ndt7_data and libre_data and median_throughputs:
    ookla_avg_duration = np.mean([v["download_duration"] for v in ookla_metrics.values() if v["download_duration"] is not None])
    ndt7_avg_duration = np.mean([v["download_duration"] for v in ndt7_metrics.values() if v["download_duration"] is not None])
    libre_avg_duration = np.mean([v["download_duration"] for v in libre_metrics.values() if v["download_duration"] is not None])
    #also for SEARCH (all ss exit time)
    search_avg_duration = np.mean(all_ss_exit_time_list) if all_ss_exit_time_list else None

    # for error: avg norm diff between median throughput and download speed of each tool
    error_ookla = np.mean([abs(median - speed) for median, speed in zip(median_throughputs, [d["Download"] for d in ookla_data]) if speed is not None])
    error_ndt7 = np.mean([abs(median - speed) for median, speed in zip(median_throughputs, [d["Download"] for d in ndt7_data]) if speed is not None])   
    error_libre = np.mean([abs(median - speed) for median, speed in zip(median_throughputs, [d["Download"] for d in libre_data]) if speed is not None])
    error_search = np.mean([abs(median - speed) for median, speed in zip(median_throughputs, delivery_rate_at_exit_on_acked_bytes_list) if speed is not None]) if delivery_rate_at_exit_on_acked_bytes_list else None

    # create a scatter plot
    plt.figure(figsize=(8, 5))
    plt.scatter(ookla_avg_duration, error_ookla, color='blue', marker='*', s=100, label='Ookla')
    plt.scatter(ndt7_avg_duration, error_ndt7, color='orange', marker='s', s=100, label='NDT7')
    plt.scatter(libre_avg_duration, error_libre, color='green', marker='^', s=100, label='LibreSpeed')
    plt.scatter(search_avg_duration, error_search, color='purple', marker='o', s=100, label='Search')

    plt.xlabel("Avg Duration (seconds)", fontsize=19)
    plt.ylabel("Avg Error (Mb/s)", fontsize=19)
    # plt.title("Average Duration vs Average Error for Speedtest Tools")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16)
    plt.xlim(left=0)  # Set x-axis to start from 0
    plt.ylim(bottom=0)  # Set y-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "avg_duration_vs_avg_error_speedtest_tools.png"))
    plt.close()


    # plot avg bit sent for each tools on x_axis and avg error (median - speed) for that tool on y-axis 
    ookla_avg_byte_sent = np.asarray(np.mean([v["bytes_sent"] for v in ookla_metrics.values() if v["bytes_sent"] is not None])) * 1e-6 #MB
    ndt7_avg_byte_sent = np.asarray(np.mean([v["bytes_sent"] for v in ndt7_metrics.values() if v["bytes_sent"] is not None])) * 1e-6 #MB
    libre_avg_byte_sent = np.asarray(np.mean([v["bytes_sent"] for v in libre_metrics.values() if v["bytes_sent"] is not None])) * 1e-6 #MB
    search_avg_byte_sent = np.asarray(np.mean(total_bytes_acked_at_exit_list)) * 1e-6 #MB


    plt.figure(figsize=(8, 5))
    plt.scatter(ookla_avg_byte_sent, error_ookla, color='blue', marker='*', s=100, label='Ookla')
    plt.scatter(ndt7_avg_byte_sent, error_ndt7, color='orange', marker='s', s=100, label='NDT7')
    plt.scatter(libre_avg_byte_sent, error_libre, color='green', marker='^', s=100, label='LibreSpeed')
    plt.scatter(search_avg_byte_sent, error_search, color='purple', marker='o', s=100, label='Search')
    plt.xlabel("Avg_Bits Sent (MB)", fontsize=19)
    plt.ylabel("Avg_Error (Mb/s)", fontsize=19)
    # plt.title("Average Bits Sent vs Average Error for Speedtest Tools")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16)
    plt.xlim(left=-1)  # Set x-axis to start from 0
    plt.ylim(bottom=0)  # Set y-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "avg_byte_sent_vs_avg_error_speedtest_tools.png"))
    plt.close()


    # I want to save this info (AVG duration, avg sent bytes, avg error for all these tools in one csv file)
    tools_data = {
        "Tool": ["Ookla", "NDT7", "LibreSpeed", "Search"],
        "Avg Duration (seconds)": [ookla_avg_duration, ndt7_avg_duration, libre_avg_duration, search_avg_duration],
        "Avg Bytes Sent (MB)": [ookla_avg_byte_sent, ndt7_avg_byte_sent, libre_avg_byte_sent, search_avg_byte_sent],
        "Avg Error (Mb/s)": [error_ookla, error_ndt7, error_libre, error_search]
    }

    tools_df = pd.DataFrame(tools_data)
    tools_df.to_csv(os.path.join(output_path, "avg_duration_avg_byte_sent_avg_error_speedtest_tools.csv"), index=False) 

if median_throughputs_diff_windows and delivery_rate_at_exit_on_acked_bytes_list:
    plt.figure(figsize=(10, 6))

    for wnd_bin, run_dict in median_throughputs_diff_windows.items():
        # Extract non-None median throughputs
        values = [v for v in run_dict.values() if v is not None]
        if not values:
            continue   
        sorted_vals = np.sort(values)
        cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)

        plt.plot(sorted_vals, cdf, marker='o', label=f'Window {wnd_bin}s')
    plt.plot(delivery_rate_at_exit_on_acked_bytes_list, cdf_values_delivery_rate_at_exit_on_acked_bytes, marker='x', color='m', label='Delivery Rate at Exit')
    plt.xlabel('Median Throughput (Mb/s)', fontsize=16)
    plt.ylabel('Cumulative Distribution', fontsize=16)
    plt.title('CDF of Median Throughput by Window Size', fontsize=16)
    plt.legend()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14) 
    plt.xlim(left=-0.05)  # Set x-axis to start from 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cdf_median_throughput_by_window_size.png"))
    plt.close()


# # plot cdf diff median throughput and download speed of all tools and diff median thropughput and delivery rate at exit on one graph
# if ookla_data and ndt7_data and libre_data and median_throughputs and delivery_rate_at_exit_on_acked_bytes_list:
#     diff_ookla = [median_throughputs[i] - ookla_data[i]["Download"] if i < len(median_throughputs) else None for i in range(len(ookla_data))]
#     diff_ndt7 = [ median_throughputs[i] - ndt7_data[i]["Download"] if i < len(median_throughputs) else None for i in range(len(ndt7_data))]
#     diff_libre = [median_throughputs[i] - libre_data[i]["Download"] if i < len(median_throughputs) else None for i in range(len(libre_data))]
#     diff_delivery_rate_exit = [median_throughputs[i] - delivery_rate_at_exit_on_acked_bytes_list[i] if i < len(median_throughputs) else None for i in range(len(median_throughputs))]

#     cdf_diff_ookla, cdf_values_diff_ookla = calculate_cdf(diff_ookla)
#     cdf_diff_ndt7, cdf_values_diff_ndt7 = calculate_cdf(diff_ndt7)
#     cdf_diff_libre, cdf_values_diff_libre = calculate_cdf(diff_libre)
#     cdf_diff_delivery_rate_exit, cdf_values_diff_delivery_rate_exit = calculate_cdf(diff_delivery_rate_exit)

#     plt.figure(figsize=(8, 5))
#     # plot vertical dashed line on zero
#     plt.axvline(x=0, color='gray', linestyle='--')
#     plt.plot(cdf_diff_ookla, cdf_values_diff_ookla, marker='o', color='blue', label='Median Throughput - Ookla')
#     plt.plot(cdf_diff_ndt7, cdf_values_diff_ndt7, marker='o', color='orange', label='Median Throughput - NDT7')
#     plt.plot(cdf_diff_libre, cdf_values_diff_libre, marker='o', color='green', label='Median Throughput - Libre')
#     plt.plot(cdf_diff_delivery_rate_exit, cdf_values_diff_delivery_rate_exit, marker='o', color='purple', label='Median Throughput - DR at Exit')
#     plt.xlabel("Difference (Mb/s)", fontsize=19)
#     plt.ylabel("Cumulative Distribution", fontsize=19)
#     # plt.title("CDF of Difference between Median Throughput and Download Speeds of Tools and Delivery Rate at Exit")
#     plt.legend(fontsize=16)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     # plt.xlim(left=-0.5)  # Set x-axis to start from
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_diff_median_throughput_and_download_speeds_and_delivery_rate_at_exit.png"))  
#     plt.close()

#     # plot norm diff
#     norm_diff_ookla = [(median - ookla_data[i]["Download"]) / median if median is not None and ookla_data[i]["Download"] else None for i, median in enumerate(median_throughputs)]
#     norm_diff_ndt7 = [(median - ndt7_data[i]["Download"]) / median if median is not None and ndt7_data[i]["Download"] else None for i, median in enumerate(median_throughputs)] 
#     norm_diff_libre = [(median - libre_data[i]["Download"]) / median if median is not None and libre_data[i]["Download"] else None for i, median in enumerate(median_throughputs)]
#     norm_diff_delivery_rate_exit = [(median - delivery_rate) / median if median is not None and delivery_rate is not None else None
#                                      for median, delivery_rate in zip(median_throughputs, delivery_rate_at_exit_on_acked_bytes_list)]
    
#     cdf_norm_diff_ookla, cdf_values_norm_diff_ookla = calculate_cdf(norm_diff_ookla)
#     cdf_norm_diff_ndt7, cdf_values_norm_diff_ndt7 = calculate_cdf(norm_diff_ndt7)
#     cdf_norm_diff_libre, cdf_values_norm_diff_libre = calculate_cdf(norm_diff_libre)
#     cdf_norm_diff_delivery_rate_exit, cdf_values_norm_diff_delivery_rate_exit = calculate_cdf(norm_diff_delivery_rate_exit)     

#     plt.figure(figsize=(8, 5))
#     # plot vertical dashed line on zero
#     plt.axvline(x=0, color='gray', linestyle='--')
#     plt.plot(cdf_norm_diff_ookla, cdf_values_norm_diff_ookla, marker='o', color='blue', label='Norm Diff (Median Throughput - Ookla)')
#     plt.plot(cdf_norm_diff_ndt7, cdf_values_norm_diff_ndt7, marker='o', color='orange', label='Norm Diff (Median Throughput - NDT7)')
#     plt.plot(cdf_norm_diff_libre, cdf_values_norm_diff_libre, marker='o', color='green', label='Norm Diff (Median Throughput - Libre)') 
#     plt.plot(cdf_norm_diff_delivery_rate_exit, cdf_values_norm_diff_delivery_rate_exit, marker='o', color='purple', label='Norm Diff (Median Throughput - DR at Exit)')
#     plt.xlabel("Normalized Difference", fontsize=19)
#     plt.ylabel("Cumulative Distribution", fontsize=19)
#     # plt.title("CDF of Normalized Difference between Median Throughput and Download Speeds of Tools and Delivery Rate at Exit")
#     plt.legend(fontsize=16)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     # plt.xlim(left=-0.5)  # Set x-axis to start from
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_norm_diff_median_throughput_and_download_speeds_and_delivery_rate_at_exit.png"))  
#     plt.close()

# # plot cdf diff between median throughput and delivery rate at exit and cdf norm diff
# if median_throughputs and delivery_rate_at_exit_on_acked_bytes_list:
#     diff_median_throughput_delivery_rate_exit = [
#         (median - delivery_rate) if delivery_rate is not None else None
#         for median, delivery_rate in zip(median_throughputs, delivery_rate_at_exit_on_acked_bytes_list)
#     ]

#     cdf_diff_median_throughput_delivery_rate_exit, cdf_values_diff_median_throughput_delivery_rate_exit = calculate_cdf(diff_median_throughput_delivery_rate_exit)

#     plt.figure(figsize=(8, 5))
#     # plot vertical dashed line on zero
#     plt.axvline(x=0, color='gray', linestyle='--', label='Zero Line')
#     plt.plot(cdf_diff_median_throughput_delivery_rate_exit, cdf_values_diff_median_throughput_delivery_rate_exit, marker='o', color='purple', label='CDF of Diff (Median Throughput - DR at Exit)')
#     plt.xlabel("Difference (Mb/s)", fontsize=19)
#     plt.ylabel("Cumulative Distribution", fontsize=19)
#     # plt.title("CDF of Difference between Median Throughput and Delivery Rate at Exit")
#     # plt.legend(fontsize=16)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     # plt.xlim(left=-0.5)  # Set x-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_diff_median_throughput_and_delivery_rate_at_exit.png"))
#     plt.close()

#     # normalized diff
#     norm_diff_median_throughput_dr_exit = [
#         (median - delivery_rate) / median if median is not None and delivery_rate is not None else None
#         for median, delivery_rate in zip(median_throughputs, delivery_rate_at_exit_on_acked_bytes_list)
#     ]

#     cdf_norm_diff_median_throughput_dr_exit, cdf_values_norm_diff_median_throughput_dr_exit = calculate_cdf(norm_diff_median_throughput_dr_exit)
#     plt.figure(figsize=(8, 5))
#     # plot vertical dashed line on zero
#     plt.axvline(x=0, color='gray', linestyle='--', label='Zero Line')
#     plt.plot(cdf_norm_diff_median_throughput_dr_exit, cdf_values_norm_diff_median_throughput_dr_exit, marker='o', color='purple', label='CDF of Norm Diff (Median Throughput - DR at Exit)')
#     plt.xlabel("Normalized Difference", fontsize=19)
#     plt.ylabel("Cumulative Distribution", fontsize=19)
#     # plt.title("CDF of Normalized Difference between Median Throughput and Delivery Rate at Exit   ")
#     # plt.legend(fontsize=16)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     # plt.xlim(left=-0.5)  # Set x-axis to start from
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_norm_diff_median_throughput_and_delivery_rate_at_exit.png"))
#     plt.close()

############################################################################################################################
# # plot average and median throughput 
# if avg_throughputs and median_throughputs:
#     plt.figure()
#     plt.plot(range(1, num + 1), avg_throughputs, marker='o', label='Average Throughput')
#     plt.plot(range(1, num + 1), median_throughputs, marker='o', label='Median Throughput')
#     plt.xlabel("Run", fontsize=16)
#     plt.ylabel("Rate (Mb/s)", fontsize=16)
#     # plt.title("Average and Median Throughput Over Runs")
#     plt.legend()
#     # plt.ylim([0, 100]) 
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "avg_median_throughput_over_runs.png"))
#     plt.close()

#     # plot cdf of average throughput
#     cdf_avg_throughputs, cdf_values_avg_throughputs = calculate_cdf(avg_throughputs)

#     plt.figure()
#     plt.plot(cdf_avg_throughputs, cdf_values_avg_throughputs, marker='o', label='CDF of Average Throughput')
#     plt.xlabel("Avg throughput(Mb/s)", fontsize=16)
#     plt.ylabel("Cumulative Distribution", fontsize=16)
#     # plt.title("CDF of Average Throughput")
#     # plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlim(left=0)  # Set x-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_avg_throughput.png"))
#     plt.close()

#     # plot cdf of median throughput
#     cdf_median_throughputs, cdf_values_median_throughputs = calculate_cdf(median_throughputs)
#     plt.figure()
#     plt.plot(cdf_median_throughputs, cdf_values_median_throughputs, marker='o', color='orange', label='CDF of Median Throughput')
#     plt.xlabel("Median throughput (Mb/s)", fontsize=16)
#     plt.ylabel("Cumulative Distribution", fontsize=16)
#     # plt.title("CDF of Median Throughput")
#     #plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlim(left=0)  # Set x-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_median_throughput.png"))
#     plt.close()


# # plot average delivery rate
# if avg_delivery_rates and median_delivery_rates:
#     plt.figure()
#     plt.plot(range(1, num_log_files + 1), avg_delivery_rates, marker='o', label='Average Delivery Rate')
#     plt.plot(range(1, num_log_files + 1), median_delivery_rates, marker='o', label='Median Delivery Rate')
#     plt.xlabel("Run", fontsize=16)
#     plt.ylabel("Rate (Mb/s)", fontsize=16)
#     # plt.title("Average and Median Delivery Rate Over Runs")
#     plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.ylim(bottom=0)  # Set y-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "avg_median_delivery_rate_over_runs.png"))
#     plt.close()

# # plot delivery rate at exit
# if delivery_rate_at_exit:
#     plt.figure()
#     plt.plot(range(1, num_log_files + 1), delivery_rate_at_exit, marker='o', label='Delivery Rate at Exit')
#     plt.xlabel("Run", fontsize=16)
#     plt.ylabel("Delivery Rate at Exit(Mb/s)", fontsize=16)
#     # plt.title("Delivery Rate at Exit Over Runs")
#     # plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.ylim(bottom=-0.5)  # Set y-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "delivery_rate_at_exit_over_runs.png"))
#     plt.close()

# # plot delivery rate at 2 RTTs before exit
# if delivery_rate_two_pre_rtt_of_exit:
#     plt.figure()
#     plt.plot(range(1, num_log_files + 1), delivery_rate_two_pre_rtt_of_exit, marker='o', label='Delivery Rate Two RTTs Before Exit')
#     plt.xlabel("Run", fontsize=16)
#     plt.ylabel("Delivery Rate Two RTTs Before Exit(Mb/s)", fontsize=16)
#     # plt.title("Delivery Rate Two RTTs Before Exit Over Runs")
#     # plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.ylim(bottom=-0.5)  # Set y-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "delivery_rate_two_rtt_before_exit_over_runs.png"))
#     plt.close()    

# # plot delivery rate at search exit time based on cur and prev window
# if delivery_rate_at_exit_based_on_window:
#     plt.figure()
#     plt.plot( delivery_rate_at_exit_based_on_window, marker='o', label='Delivery Rate at Exit Based on Window')
#     plt.xlabel("Run", fontsize=16)
#     plt.ylabel("Delivery Rate at Exit Based on Window(Mb/s)", fontsize=16)
#     # plt.title("Delivery Rate at Exit Based on Window Over Runs")
#     # plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.ylim(bottom=-0.5)  # Set y-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "delivery_rate_at_exit_based_on_window_over_runs.png"))
#     plt.close()

# # plot exit time
# if search_exit_time_list:
#     plt.figure()
#     plt.plot(range(1, len(search_exit_time_list)+1), search_exit_time_list, marker='o', label='Exit Time')    
#     plt.xlabel("Sample", fontsize=16)
#     plt.ylabel("Time (s)", fontsize=16)
#     # plt.title("Exit Time Over Runs")
#     # plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.ylim(bottom=0)  # Set y-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "exit_time_over_runs.png"))
#     plt.close()

#     # plot cdf of exit time
#     search_exit_time_list = [time for time in search_exit_time_list if time is not None]  # Remove None value
#     cdf_exit_time, cdf_values_exit_time = calculate_cdf(search_exit_time_list)
#     plt.figure()
#     plt.plot(cdf_exit_time, cdf_values_exit_time, marker='o', label='CDF of Exit Time')
#     plt.xlabel("Time (s)", fontsize=16)
#     plt.ylabel("Cumulative Distribution", fontsize=16)
#     # plt.title("CDF of Exit Time")
#     # plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlim(left=0)  # Set x-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_search_exit_time.png"))
#     plt.close()

# # plot non search exit time
# if not_search_exit_time_list:
#     plt.figure()
#     plt.plot(range(1, len(not_search_exit_time_list)+1), not_search_exit_time_list, marker='o', label='Non-Search Exit Time')
#     plt.xlabel("Run", fontsize=16)
#     plt.ylabel("Time (s)", fontsize=16)
#     # plt.title("Non-Search Exit Time Over Runs")
#     # plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.ylim(bottom=0)  # Set y-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "non_search_exit_time_over_runs.png"))
#     plt.close()
    
#     # plot cdf of non search exit time
#     not_search_exit_time_list = [time for time in not_search_exit_time_list if time is not None]  # Remove None value
#     cdf_not_search_exit_time, cdf_values_not_search_exit_time = calculate_cdf(not_search_exit_time_list)
#     plt.figure()
#     plt.plot(cdf_not_search_exit_time, cdf_values_not_search_exit_time, marker='o', label='CDF of Non-Search Exit Time')
#     plt.xlabel("Time (s)", fontsize=16)
#     plt.ylabel("Cumulative Distribution", fontsize=16)
#     # plt.title("CDF of Non-Search Exit Time")
#     # plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlim(left=0)  # Set x-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_not_search_exit_time.png"))
#     plt.close()

# # combine seaerch and not search exit times (but for each one hase seperate color) and then also make cdf plot
# if search_exit_time_list or not_search_exit_time_list:
#     plt.figure()
#     if search_exit_time_list:
#         plt.plot(range(1, len(search_exit_time_list)+1), search_exit_time_list, marker='*', label='Search Exit Time', color='b')
#     if not_search_exit_time_list:
#         plt.plot(range(1, len(not_search_exit_time_list)+1), not_search_exit_time_list, marker='o', label='Loss Exit Time', color='r')
#     plt.xlabel("Run", fontsize=16)
#     plt.ylabel("Time (s)", fontsize=16)
#     # plt.title("Search and Non-Search Exit Time Over Runs")
#     plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.ylim(bottom=0)  # Set y-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "search_not_search_exit_time_over_runs.png"))
#     plt.close()

# if search_exit_time_list or not_search_exit_time_list:
#     # Combine all with labels
#     combined = [(x, 'SEARCH') for x in search_exit_time_list] + \
#             [(x, 'NON_SEARCH') for x in not_search_exit_time_list]

#     # Sort by exit time
#     combined.sort(key=lambda x: x[0])

#     # Separate sorted values and tags
#     values = [x[0] for x in combined]
#     tags = [x[1] for x in combined]

#     # Compute CDF
#     cdf = np.arange(1, len(values) + 1) / len(values)

#     # Plot the combined CDF as dots with color based on tag
#     for x, y, label in zip(values, cdf, tags):
#         color = 'blue' if label == 'SEARCH' else 'red'
#         marker = '*' if label == 'SEARCH' else 'o'
#         plt.plot(x, y, marker=marker, linestyle='None', color=color, label=label)

#     # Fix legend (avoid duplicates)
#     handles = [
#         plt.Line2D([0], [0], marker='*', markersize=10, color='w', label='SEARCH Exit', markerfacecolor='blue'),
#         plt.Line2D([0], [0], marker='o', color='w', label='LOSS Exit', markerfacecolor='red')
#     ]
#     plt.legend(handles=handles)
#     plt.xlabel('Exit Time (s)')
#     plt.ylabel('CDF')
#     # plt.title('Combined CDF with Category-colored Points')
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "combined_cdf_search_not_search_exit_time.png"))
#     plt.close()

# # plot cdf all ss exit times
# if all_ss_exit_time_list:
#     all_ss_exit_time_list = [time for time in all_ss_exit_time_list if time is not None]  # Remove None value
#     cdf_all_ss_exit_time, cdf_values_all_ss_exit_time = calculate_cdf(all_ss_exit_time_list)
#     plt.figure()
#     plt.plot(cdf_all_ss_exit_time, cdf_values_all_ss_exit_time, marker='o', label='CDF of All SS Exit Time')
#     plt.xlabel("Time (s)", fontsize=16)
#     plt.ylabel("Cumulative Distribution", fontsize=16)
#     # plt.title("CDF of All SS Exit Time")
#     # plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlim(left=0)  # Set x-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_all_ss_exit_time.png"))
#     plt.close()

# # calculate cdf of AVG delivery rates and then plot it
# if avg_delivery_rates:
#     cdf_avg_delivery_rates, cdf_values_avg = calculate_cdf(avg_delivery_rates)

#     plt.figure()
#     plt.plot(cdf_avg_delivery_rates, cdf_values_avg, marker='o', label='CDF of Average Delivery Rates')
#     plt.xlabel("Mb/s", fontsize=16)
#     plt.ylabel("Cumulative Distribution", fontsize=16)
#     # plt.title("CDF of Average Delivery Rates")
#     # plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlim(left=0)  # Set x-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_avg_delivery_rates.png"))
#     plt.close()

# # calculate cdf of delivery rate at exit and then plot it
# if delivery_rate_at_exit:
#     # Filter out None values from delivery_rate_at_exit
#     delivery_rate_at_exit = [rate for rate in delivery_rate_at_exit if rate is not None]
#     cdf_delivery_rate_exit, cdf_values_exit = calculate_cdf(delivery_rate_at_exit)

#     plt.figure()
#     plt.plot(cdf_delivery_rate_exit, cdf_values_exit, marker='o', color='m', label='CDF of Delivery Rate at Exit')
#     plt.xlabel("Mb/s", fontsize=16)
#     plt.ylabel("Cumulative Distribution", fontsize=16)
#     # plt.title("CDF of Delivery Rate at Exit")
#     # plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlim(left=0)  # Set x-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_delivery_rate_at_exit.png"))
#     plt.close()

# # Plot the cdf of delivery rate and delivery rate at exit in one figure
# if avg_delivery_rates and delivery_rate_at_exit:
#     plt.figure()
#     plt.plot(cdf_avg_delivery_rates, cdf_values_avg, marker='o', color='b', label='CDF of Average Delivery Rates')
#     plt.plot(cdf_delivery_rate_exit, cdf_values_exit, marker='o', color='m', label='CDF of Delivery Rate at Exit')
#     plt.xlabel("Mb/s", fontsize=16)  
#     plt.ylabel("Cumulative Distribution", fontsize=16)
#     # plt.title("CDF of Delivery Rates")
#     plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlim(left=0)  # Set x-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_delivery_rates_combined.png"))
#     plt.close()

# # plot the cdf of median delivery rates and delivery rate at exit in one figure
# if median_delivery_rates and delivery_rate_at_exit:
#     cdf_median_delivery_rates, cdf_values_median = calculate_cdf(median_delivery_rates)

#     plt.figure()
#     plt.plot(cdf_median_delivery_rates, cdf_values_median, marker='o', color='c', label='CDF of Median Delivery Rates')
#     plt.plot(cdf_delivery_rate_exit, cdf_values_exit, marker='o', color='m', label='CDF of Delivery Rate at Exit')
#     plt.xlabel("Mb/s", fontsize=16)  
#     plt.ylabel("Cumulative Distribution", fontsize=16)
#     # plt.title("CDF of Median Delivery Rates and Delivery Rate at Exit")
#     plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlim(left=-0.05)  # Set x-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_median_delivery_rates_combined.png"))
#     plt.close()

# # plot the cdf of delivery rate at search exit time and delivery rate at non-search exit time in one figure
# if delivery_are_at_search_exit_time or delivery_rate_at_non_search_exit_time:
#     # remove none values from delivery_are_at_search_exit_time and delivery_rate_at_non_search_exit_time
#     delivery_are_at_search_exit_time_no_none= [x for x in delivery_are_at_search_exit_time if x is not None]
#     delivery_rate_at_non_search_exit_time_no_none = [x for x in delivery_rate_at_non_search_exit_time if x is not None]
#     # Combine all with labels
#     combined = [(x, 'SEARCH') for x in delivery_are_at_search_exit_time_no_none] + \
#             [(x, 'NON_SEARCH') for x in delivery_rate_at_non_search_exit_time_no_none]

#     # Sort by exit time
#     combined.sort(key=lambda x: x[0])

#     # Separate sorted values and tags
#     values = [x[0] for x in combined]
#     tags = [x[1] for x in combined]

#     # Compute CDF
#     cdf = np.arange(1, len(values) + 1) / len(values)

#     # Plot the combined CDF as dots with color based on tag
#     for x, y, label in zip(values, cdf, tags):
#         color = 'blue' if label == 'SEARCH' else 'red'
#         marker = '*' if label == 'SEARCH' else 'o'
#         plt.plot(x, y, marker=marker, linestyle='None', color=color, label=label)

#     # Fix legend (avoid duplicates)
#     handles = [
#         plt.Line2D([0], [0], marker='*', markersize=10, color='w', label='Delivery Rate (SEARCH Exit)', markerfacecolor='blue'),
#         plt.Line2D([0], [0], marker='o', color='w', label='Delivery Rate (LOSS Exit)', markerfacecolor='red')
#     ]
#     plt.legend(handles=handles)
#     plt.xlabel('Delivery Rate  at Exit(Mb/s)')
#     plt.ylabel('CDF')
#     # plt.title('Combined CDF with Category-colored Points')
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "combined_cdf_search_not_search_delivery_rate.png"))
#     plt.close()

# # plot the cdf of median throughput and delivery rate at exit in one figure
# if median_throughputs and delivery_rate_at_exit:
#     plt.figure()
#     plt.plot(cdf_median_throughputs, cdf_values_median_throughputs, marker='o', color='c', label='CDF of Median Throughput')
#     plt.plot(cdf_delivery_rate_exit, cdf_values_exit, marker='o', color='m', label='CDF of Delivery Rate at Exit')
#     plt.xlabel("Mb/s", fontsize=16)  
#     plt.ylabel("Cumulative Distribution", fontsize=16)
#     # plt.title("CDF of Median Throughput and Delivery Rate at Exit")    
#     plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlim(left=-0.05)  # Set x-axis to start from
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_median_throughput_delivery_rate_exit_combined.png"))
#     plt.close()
     
#     if delivery_rate_at_exit_on_acked_bytes_list:
#         # plot cdf of delivery rate at exit on acked bytes
#         delivery_rate_at_exit_on_acked_bytes = [rate for rate in delivery_rate_at_exit_on_acked_bytes_list if rate is not None]
#         cdf_delivery_rate_exit_on_acked_bytes, cdf_values_exit_on_acked_bytes = calculate_cdf(delivery_rate_at_exit_on_acked_bytes)

#         plt.figure()
#         plt.plot(cdf_delivery_rate_exit_on_acked_bytes, cdf_values_exit_on_acked_bytes, marker='o', color='g', label='CDF of Delivery Rate at Exit on Acked Bytes')
#         plt.xlabel("Mb/s", fontsize=16)  
#         plt.ylabel("Cumulative Distribution", fontsize=16)
#         # plt.title("CDF of Delivery Rate at Exit on Acked Bytes")    
#         plt.legend()
#         plt.xticks(fontsize=14)
#         plt.yticks(fontsize=14)
#         plt.xlim(left=-0.05)  # Set x-axis to start from 0
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_path, "cdf_delivery_rate_exit_on_acked_bytes.png"))
#         plt.close()

#         # plot cdf of delivery rate at exit on bytes acked and median throughput in one figure
#         plt.figure()
#         plt.plot(cdf_delivery_rate_exit_on_acked_bytes, cdf_values_exit_on_acked_bytes, marker='o', color='g', label='CDF of Delivery Rate at Exit on Acked Bytes')
#         plt.plot(cdf_median_throughputs, cdf_values_median_throughputs, marker='o', color='c', label='CDF of Median Throughput')
#         plt.xlabel("Mb/s", fontsize=16)  
#         plt.ylabel("Cumulative Distribution", fontsize=16)
#         # plt.title("CDF of Delivery Rate at Exit on Acked Bytes and Median Throughput")    
#         plt.legend()
#         plt.xticks(fontsize=14) 
#         plt.yticks(fontsize=14)
#         plt.xlim(left=-0.05)  # Set x-axis to start from 0
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_path, "cdf_delivery_rate_exit_on_acked_bytes_median_throughput_combined.png"))
#         plt.close()

#         # diff between median throughput and delivery rate at exit on acked bytes
#         # Only compute when both values are not None (i.e., not NaN in pandas   )
#         median_throughput_series = pd.Series(median_throughputs, index=range(1, len(median_throughputs) + 1))
#         delivery_rate_exit_on_acked_bytes_series = pd.Series(delivery_rate_at_exit_on_acked_bytes, index=range(1, len(delivery_rate_at_exit_on_acked_bytes) + 1)) 

#         valid_mask = median_throughput_series.notna() & delivery_rate_exit_on_acked_bytes_series.notna()    
#         normalized_dif_median_throughput_exit_on_acked_bytes_percent = pd.Series(index=median_throughput_series.index, dtype="float")
#         normalized_dif_median_throughput_exit_on_acked_bytes_percent[valid_mask] = (
#             (median_throughput_series[valid_mask] - delivery_rate_exit_on_acked_bytes_series[valid_mask]) /
#             median_throughput_series[valid_mask]) * 100

#         # plot normalized difference
#         plt.figure()
#         plt.plot(range(1, len(normalized_dif_median_throughput_exit_on_acked_bytes_percent) + 1), normalized_dif_median_throughput_exit_on_acked_bytes_percent, marker='o', color='brown', label='Normalized Difference')
#         plt.xlabel("Run", fontsize=16)
#         plt.ylabel("Normalized Difference (%)", fontsize=16)
#         # plt.title("Normalized Difference Between Median Throughput and Delivery Rate at Exit on Acked Bytes")    
#         # plt.legend()
#         plt.xticks(fontsize=14)
#         plt.yticks(fontsize=14)
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_path, "normalized_difference_median_throughput_vs_delivery_rate_exit_on_acked_bytes.png"))
#         plt.close()

# # plot diff between median throughput and delivery rate at exit
# if median_throughputs and delivery_rate_at_exit:
#     # Filter out None values from delivery_rate_at_exit
#     delivery_rate_at_exit = [rate for rate in delivery_rate_at_exit if rate is not None]

#     # Create a Series for median throughput
#     median_throughput_series = pd.Series(median_throughputs, index=range(1, len(median_throughputs) + 1))

#     # Create a Series for delivery rate at exit
#     delivery_rate_exit_series = pd.Series(delivery_rate_at_exit, index=range(1, len(delivery_rate_at_exit) + 1))

#     # Only compute when both values are not None (i.e., not NaN in pandas)
#     valid_mask = median_throughput_series.notna() & delivery_rate_exit_series.notna()

#     normalized_dif_median_throughput_exit_percent = pd.Series(index=median_throughput_series.index, dtype="float")
#     normalized_dif_median_throughput_exit_percent[valid_mask] = (
#         (median_throughput_series[valid_mask] - delivery_rate_exit_series[valid_mask]) /
#         median_throughput_series[valid_mask]
#     ) * 100

#     # plot normalized difference
#     plt.figure()
#     plt.plot(range(1, len(normalized_dif_median_throughput_exit_percent) + 1), normalized_dif_median_throughput_exit_percent, marker='o', color='brown', label='Normalized Difference')
#     plt.xlabel("Run", fontsize=16)
#     plt.ylabel("Normalized Difference (%)", fontsize=16)
#     # plt.title("Normalized Difference Between Median Throughput and Delivery Rate at Exit")    
#     # plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "normalized_difference_median_throughput_vs_delivery_rate_exit.png"))
#     plt.close()

# # plot the cdf of delivery rate at search exit time and delivery rate at non-search exit time and cdf of median throughput in one figure
# if delivery_are_at_search_exit_time or delivery_rate_at_non_search_exit_time:
#     # remove none values from delivery_are_at_search_exit_time and delivery_rate_at_non_search_exit_time
#     delivery_are_at_search_exit_time_no_none= [x for x in delivery_are_at_search_exit_time if x is not None]
#     delivery_rate_at_non_search_exit_time_no_none = [x for x in delivery_rate_at_non_search_exit_time if x is not None]
#     # Combine all with labels
#     combined = [(x, 'SEARCH') for x in delivery_are_at_search_exit_time_no_none] + \
#             [(x, 'NON_SEARCH') for x in delivery_rate_at_non_search_exit_time_no_none]

#     # Sort by exit time
#     combined.sort(key=lambda x: x[0])

#     # Separate sorted values and tags
#     values = [x[0] for x in combined]
#     tags = [x[1] for x in combined]

#     # Compute CDF
#     cdf = np.arange(1, len(values) + 1) / len(values)

#     # Plot the combined CDF as dots with color based on tag
#     for x, y, label in zip(values, cdf, tags):
#         color = 'blue' if label == 'SEARCH' else 'red'
#         marker = '*' if label == 'SEARCH' else 'o'
#         plt.plot(x, y, marker=marker, linestyle='None', color=color, label=label)

#     # Fix legend (avoid duplicates)
#     handles = [
#         plt.Line2D([0], [0], marker='*', markersize=10, color='w', label='Delivery Rate (SEARCH Exit)', markerfacecolor='blue'),
#         plt.Line2D([0], [0], marker='o', color='w', label='Delivery Rate (LOSS Exit)', markerfacecolor='red'),
#         plt.Line2D([0], [0], marker='^', color='c', label='Median Throughput', markerfacecolor='c'),
#     ]
#     plt.plot(cdf_median_throughputs, cdf_values_median_throughputs, marker='^', color='c', label='CDF of Median Throughput')
#     plt.legend(handles=handles)
#     plt.xlabel('Delivery Rate  at Exit(Mb/s)')
#     plt.ylabel('Cumulative Distribution')
#     # plt.title('Combined CDF with Category-colored Points')
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "combined_cdf_search_not_search_delivery_rate_with_median_th.png"))
#     plt.close()

#     # plot cdf of normalized difference
#     cdf_normalized_dif_median_throughput_exit, cdf_values_normalized_median_throughput_exit = calculate_cdf(normalized_dif_median_throughput_exit_percent)

#     plt.figure()
#     plt.plot(cdf_normalized_dif_median_throughput_exit, cdf_values_normalized_median_throughput_exit, marker='o', color='brown', linestyle="", label='CDF of Normalized Difference')
#     # plot v line on 0
#     plt.axvline(0, color='black', linestyle='--', linewidth=1)
#     plt.xlabel("Normalized Difference (%)", fontsize=16)
#     plt.ylabel("Cumulative Distribution", fontsize=16)
#     # plt.title("CDF of Normalized Difference Between Median Throughput and Delivery Rate at Exit")
#     plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     # plt.xlim(left=-10, right=20)  # Set x-axis to

#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_normalized_difference_median_throughput_vs_delivery_rate_exit.png"))
#     plt.close()


#     # plot just difference between median throughput and delivery rate at exit (median throughput - delivery rate at exit)
#     plt.figure()
#     plt.plot(range(1, len(median_throughputs) + 1), np.array(median_throughputs) - np.array(delivery_rate_at_exit), marker='o', color='brown', label='Difference')
#     plt.xlabel("Run", fontsize=16)
#     plt.ylabel("Difference (Mb/s)", fontsize=16)
#     # plt.title("Difference Between Median Throughput and Delivery Rate at Exit")
#     plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "difference_median_throughput_vs_delivery_rate_exit.png"))
#     plt.close()

#     # plot cdf of difference between median throughput and delivery rate at exit
#     difference_median_throughput_exit = np.array(median_throughputs) - np.array(delivery_rate_at_exit)
#     difference_median_throughput_exit = [diff for diff in difference_median_throughput_exit if diff is not None]  # Remove None values
#     cdf_difference_median_throughput_exit, cdf_values_difference_median_throughput_exit = calculate_cdf(difference_median_throughput_exit)

#     plt.figure()
#     plt.plot(cdf_difference_median_throughput_exit, cdf_values_difference_median_throughput_exit, marker='o', color='brown', linestyle="", label='CDF of Difference')
#     # plot v line on 0
#     plt.axvline(0, color='black', linestyle='--', linewidth=1)
#     plt.xlabel("Difference (Mb/s)", fontsize=16)
#     plt.ylabel("Cumulative Distribution", fontsize=16)
#     # plt.title("CDF of Difference Between Median Throughput and Delivery Rate at Exit")
#     plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     # plt.xlim(left=-10, right=20)  # Set x-axis to start from -100 to 100
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_difference_median_throughput_vs_delivery_rate_exit.png"))
#     plt.close()

# # scatter plot of ookla speed test pre download (x) and search exit delivery rate (y-axis), one dot for each run
# if not ookla_df_client.empty and delivery_rate_at_exit:

#     plt.figure()
#     plt.scatter(ookla_df_client["Download"], delivery_rate_at_exit, marker='o', color='g')
#     plt.xlabel("ookla Download Speed (Mb/s)", fontsize=16)
#     plt.ylabel("Delivery Rate at Exit (Mb/s)", fontsize=16)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlim(left=-0.05)
#     plt.ylim(bottom=-0.05)  # Set y-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "ookla_download_vs_delivery_rate_exit.png"))
#     plt.close()

# # scatter plot of ndt7 speed test pre download (x) and search exit delivery rate (y-axis), one dot for each run
# if not ndt7_df_client.empty and delivery_rate_at_exit:
    
#     plt.figure()
#     plt.scatter(ndt7_df_client["Download"], delivery_rate_at_exit, marker='o', color='b')
#     plt.xlabel("ndt7 Download Speed (Mb/s)", fontsize=16)
#     plt.ylabel("Delivery Rate at Exit (Mb/s)", fontsize=16)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlim(left=-0.05)
#     plt.ylim(bottom=-0.05)  # Set y-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "ndt7_download_vs_delivery_rate_exit.png"))
#     plt.close()

# # scatter plot of libre speed test pre download (x) and search exit delivery rate (y-axis), one dot for each run
# if not libre_df_client.empty and delivery_rate_at_exit:
#     plt.figure()
#     plt.scatter(libre_df_client["Download"], delivery_rate_at_exit, marker='o', color='r')
#     plt.xlabel("libre Download Speed (Mb/s)", fontsize=16)
#     plt.ylabel("Delivery Rate at Exit (Mb/s)", fontsize=16)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlim(left=-0.05)
#     plt.ylim(bottom=-0.05)  # Set y-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "libre_download_vs_delivery_rate_exit.png"))
#     plt.close()

# # compute normalized difference ( download - delivery rate at exit) /  download * 100
# if not ookla_df_client.empty and not ndt7_df_client.empty and not libre_df_client.empty and delivery_rate_at_exit:

#     delivery_rate_series = pd.Series(delivery_rate_at_exit, index=ookla_df_client.index)

#     # normalized difference for ookla and delivery rate at exit
#     valid_mask_ookla = ookla_df_client["Download"].notna() & delivery_rate_series.notna()
#     normalized_dif_ookla_exit_percent = pd.Series(index= ookla_df_client.index, dtype="float")
#     normalized_dif_ookla_exit_percent[valid_mask_ookla] = (    
#         (ookla_df_client["Download"][valid_mask_ookla] - delivery_rate_series[valid_mask_ookla]) /
#         ookla_df_client["Download"][valid_mask_ookla]
#     ) * 100

#     # normalized difference for ndt7 and delivery rate at exit
#     valid_mask_ndt7 = ndt7_df_client["Download"].notna() & delivery_rate_series.notna()
#     normalized_dif_ndt7_exit_percent = pd.Series(index= ndt7_df_client.index, dtype="float")
#     normalized_dif_ndt7_exit_percent[valid_mask_ndt7] = (
#         (ndt7_df_client["Download"][valid_mask_ndt7] - delivery_rate_series[valid_mask_ndt7]) /
#         ndt7_df_client["Download"][valid_mask_ndt7]
#     ) * 100 

#     # normalized difference for libre and delivery rate at exit
#     valid_mask_libre = libre_df_client["Download"].notna() & delivery_rate_series
#     normalized_dif_libre_exit_percent = pd.Series(index= libre_df_client.index, dtype="float")
#     normalized_dif_libre_exit_percent[valid_mask_libre] = (
#         (libre_df_client["Download"][valid_mask_libre] - delivery_rate_series[valid_mask_libre]) /
#         libre_df_client["Download"][valid_mask_libre]
#     ) * 100


#     # plot normalized difference
#     plt.figure()
#     plt.plot(range(1, len(normalized_dif_ookla_exit_percent) + 1), normalized_dif_ookla_exit_percent, marker='o', color='g', label='Ookla')
#     plt.plot(range(1, len(normalized_dif_ndt7_exit_percent) + 1), normalized_dif_ndt7_exit_percent, marker='o', color='b', label='NDT7')
#     plt.plot(range(1, len(normalized_dif_libre_exit_percent) + 1), normalized_dif_libre_exit_percent, marker='o', color='r', label='Libre')
#     plt.xlabel("Run", fontsize=16)
#     plt.ylabel("Normalized Difference (%)", fontsize=16) 
#     plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "normalized_difference_download_vs_delivery_rate_exit.png"))
#     plt.close()

#     # plot cdf of normalized difference
#     cdf_normalized_dif_ookla_exit, cdf_values_normalized_ookla = calculate_cdf(normalized_dif_ookla_exit_percent)
#     cdf_normalized_dif_ndt7_exit, cdf_values_normalized_ndt7 = calculate_cdf(normalized_dif_ndt7_exit_percent)
#     cdf_normalized_dif_libre_exit, cdf_values_normalized_libre = calculate_cdf(normalized_dif_libre_exit_percent)

#     plt.figure()
#     plt.plot(cdf_normalized_dif_ookla_exit, cdf_values_normalized_ookla, marker='o', color='g', linestyle="-", label='CDF of Ookla Normalized Difference')
#     plt.plot(cdf_normalized_dif_ndt7_exit, cdf_values_normalized_ndt7, marker='o', color='b', linestyle="-", label='CDF of NDT7 Normalized Difference')
#     plt.plot(cdf_normalized_dif_libre_exit, cdf_values_normalized_libre, marker='o', color='r', linestyle="-", label='CDF of Libre Normalized Difference')
#     #plot v line on 0
#     plt.axvline(0, color='black', linestyle='--', linewidth=1)
#     plt.xlabel("Normalized Difference (%)", fontsize=16)
#     plt.ylabel("Cumulative Distribution", fontsize=16)
#     plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     # plt.xlim(left=-10, right=20)  # Set x-axis to start from -100 to 100
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_normalized_difference_downloads_vs_delivery_rate_exit.png"))
#     plt.close()

# # plot cdf speed info from ookla, ndt7, and libre and and SEARCH info
# if ookla_metrics and ndt7_metrics and libre_metrics and all_ss_exit_time_list:

#     # CDF OF DOWNLOAD DURATIONS AND EXIT TIME

#     # extract download duration from ookla, ndt7, and libre metrics
#     download_duration_ookla_list = [v["download_duration"] for v in ookla_metrics.values() if v["download_duration"] is not None]

#     download_duration_ndt7_list = [v["download_duration"] for v in ndt7_metrics.values() if v["download_duration"] is not None]

#     download_duration_libre_list = [v["download_duration"] for v in libre_metrics.values() if v["download_duration"] is not None]

#     # remove None values from download duration lists
#     download_duration_ookla_list = [x for x in download_duration_ookla_list if x is not None]
#     download_duration_ndt7_list = [x for x in download_duration_ndt7_list if x is not None]
#     download_duration_libre_list = [x for x in download_duration_libre_list if x is not None]
#     all_ss_exit_time_list = [x for x in all_ss_exit_time_list if x is not None]

#     # Calculate CDF for download duration pre
#     cdf_download_duration_ookla, cdf_values_download_ookla = calculate_cdf(download_duration_ookla_list)
#     cdf_download_duration_ndt7, cdf_values_download_ndt7 = calculate_cdf(download_duration_ndt7_list)
#     cdf_download_duration_libre, cdf_values_download_libre = calculate_cdf(download_duration_libre_list)
#     cdf_exit_time, cdf_values_exit_time = calculate_cdf(all_ss_exit_time_list)

#     plt.figure()
#     plt.plot(cdf_download_duration_ookla, cdf_values_download_ookla, marker='o', color='g', label='CDF of Ookla Download Duration')
#     plt.plot(cdf_download_duration_ndt7, cdf_values_download_ndt7, marker='o', color='b', label='CDF of NDT7 Download Duration')
#     plt.plot(cdf_download_duration_libre, cdf_values_download_libre, marker='o', color='r', label='CDF of Libre Download Duration')
#     plt.plot(cdf_exit_time, cdf_values_exit_time, marker='o', color='m', label='CDF of Exit Time')
#     plt.xlabel("Time (s)", fontsize=16)
#     plt.ylabel("Cumulative Distribution", fontsize=16)
#     plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlim(left=-1)  # Set x-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_download_durations_and_exit_time.png"))
#     plt.close()

#     # CDF OF DOWNLOAD BITS SENT AND TOTAL BITS ACKED AT EXIT

#     download_byte_sent_ookla_list = [v["byte_sent"] for v in ookla_metrics.values() if v["byte_sent"] is not None]
#     download_byte_sent_ndt7_list = [v["byte_sent"] for v in ndt7_metrics.values() if v["byte_sent"] is not None]
#     download_byte_sent_libre_list = [v["byte_sent"] for v in libre_metrics.values() if v["byte_sent"] is not None]

#     # remove None values from download bits sent lists
#     download_byte_sent_ookla_list = [x for x in download_byte_sent_ookla_list if x is not None]
#     download_byte_sent_ndt7_list = [x for x in download_byte_sent_ndt7_list if x is not None]
#     download_byte_sent_libre_list = [x for x in download_byte_sent_libre_list if x is not None]
#     total_bytes_acked_at_exit_list = [x for x in total_bytes_acked_at_exit_list if x is not None]

#     # Calculate CDF for download bits sent
#     cdf_download_byte_sent_ookla, cdf_values_download_ookla = calculate_cdf(download_byte_sent_ookla_list)
#     cdf_download_byte_sent_ndt7, cdf_values_download_ndt7 = calculate_cdf(download_byte_sent_ndt7_list)
#     cdf_download_byte_sent_libre, cdf_values_download_libre = calculate_cdf(download_byte_sent_libre_list)
#     cdf_total_byte_acked_exit, cdf_values_total_byte_acked_exit = calculate_cdf(total_bytes_acked_at_exit_list)  

#     plt.figure()
#     plt.plot(cdf_download_byte_sent_ookla, cdf_values_download_ookla, marker='o', color='g', label='Ookla')
#     plt.plot(cdf_download_byte_sent_ndt7, cdf_values_download_ndt7  , marker='o', color='b', label='NDT7')
#     plt.plot(cdf_download_byte_sent_libre, cdf_values_download_libre, marker='o', color='r', label='Libre')       
#     plt.plot(cdf_total_byte_acked_exit, cdf_values_total_byte_acked_exit, marker='o', color='m', label='Total Bits Acked at Exit')
#     plt.xlabel("Bits (Mb)", fontsize=16)
#     plt.ylabel("Cumulative Distribution", fontsize=16)
#     plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlim(left=-1)  # Set x-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_download_byte_sent_and_total_byte_acked_at_exit.png"))
#     plt.close()

# # plot total bits sent over exit time
# if total_bytes_acked_over_exit_time and all_ss_exit_time_list:
#     # remove None values from total bits sent over exit time list and all ss exit time list
#     total_byte_sent_over_exit_time_list = [x for x in total_bytes_acked_over_exit_time if x is not None]
#     all_ss_exit_time_list = [x for x in all_ss_exit_time_list if x is not None]

#     plt.figure()
#     plt.plot(total_byte_sent_over_exit_time_list, marker='o', color='c', label='Total Bits Sent Over Exit Time')
#     plt.xlabel("Sample", fontsize=16)
#     plt.ylabel("Total Bits Sent (Mb)", fontsize=16)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlim(left=-1)  # Set x-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "total_byte_sent_over_exit_time.png"))
#     plt.close()

#     # plot cdf of total bits sent over exit time
#     cdf_total_byte_sent_over_exit_time, cdf_values_total_byte_sent_over_exit_time = calculate_cdf(total_byte_sent_over_exit_time_list)

#     plt.figure()
#     plt.plot(cdf_total_byte_sent_over_exit_time, cdf_values_total_byte_sent_over_exit_time, marker='o', color='c', label='CDF of Total Bits Sent Over Exit Time')
#     plt.xlabel("Total Bits Sent (Mb)", fontsize=16)
#     plt.ylabel("Cumulative Distribution", fontsize=16)
#     # plt.title("CDF of Total Bits Sent Over Exit Time")
#     # plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlim(left=-1)  # Set x-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_total_byte_sent_over_exit_time.png"))
#     plt.close() 

#     # plot cdf total bits sent over exit time and median throughput
#     if median_throughputs:  
#         cdf_median_throughputs, cdf_values_median_throughputs = calculate_cdf(median_throughputs)

#         plt.figure()
#         plt.plot(cdf_total_byte_sent_over_exit_time, cdf_values_total_byte_sent_over_exit_time, marker='o', color='c', label='CDF of Total Bits Sent Over Exit Time')
#         plt.plot(cdf_median_throughputs, cdf_values_median_throughputs, marker='o', color='m', label='CDF of Median Throughput')
#         plt.xlabel("Mb", fontsize=16)
#         plt.ylabel("Cumulative Distribution", fontsize=16)
#         # plt.title("CDF of Total Bits Sent Over Exit Time and Median Throughput")
#         plt.legend()
#         plt.xticks(fontsize=14)
#         plt.yticks(fontsize=14)
#         plt.xlim(left=-1, right=50)  # Set x-axis to start from 0
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_path, "cdf_total_byte_sent_over_exit_time_and_median_throughput.png"))
#         plt.close()

# if median_throughputs_diff_windows and cdf_delivery_rate_exit:
#     plt.figure(figsize=(10, 6))

#     for wnd_bin, run_dict in median_throughputs_diff_windows.items():
#         # Extract non-None median throughputs
#         values = [v for v in run_dict.values() if v is not None]
#         if not values:
#             continue   
#         sorted_vals = np.sort(values)
#         cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)

#         plt.plot(sorted_vals, cdf, marker='o', label=f'Window {wnd_bin}s')
#     plt.plot(cdf_delivery_rate_exit, cdf_values_exit, marker='x', color='m', label='Delivery Rate at Exit')
#     plt.xlabel('Median Throughput (Mb/s)', fontsize=16)
#     plt.ylabel('Cumulative Distribution', fontsize=16)
#     plt.title('CDF of Median Throughput by Window Size', fontsize=16)
#     plt.legend()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14) 
#     plt.xlim(left=-0.05)  # Set x-axis to start from 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, "cdf_median_throughput_by_window_size.png"))
#     plt.close()


# plot throughput of search and throughput of pre download on one graph
    # if throughput_df_pre_list and throughput_all:
    #     for i in range(len(throughput_df_pre_list)):
    #         if i+1 in time_throughput_all:
                
    #             plt.figure(figsize=(10, 5))
    #             plt.plot(throughput_time_pre_list[i], throughput_df_pre_list[i]["throughput_Mbps"], marker='o', label=f'Pre Download Run {i+1}')
    #             plt.plot(time_throughput_all[i+1], throughput_all[i+1], marker='x', label=f'Search Throughput Run {i+1}')
    #             plt.xlabel('Time (s)', fontsize=16)
    #             plt.ylabel('Throughput (Mb/s)', fontsize=16)
    #             plt.title('Throughput of Pre Download and Search Over Time')
    #             plt.legend()
    #             plt.xticks(fontsize=14)
    #             plt.yticks(fontsize=14)
    #             plt.xlim([-1, 11])
    #             plt.ylim(bottom=0)  # Set y-axis to start from 0
    #             plt.tight_layout()
    #             plt.savefig(os.path.join(output_path, f"throughput_pre_download_and_search_{i+1}.png"))
    #             plt.close()

    # # plot throughput of search and throughput of post download on one graph
    # if throughput_df_post_list and throughput_all and throughput_df_pre_list:
    #     for i in range(len(throughput_all)):
    #         # if throughput_time_post_list[i] or throughput_time_pre_list[i] does not exist, skip this iteration
    #         if i+1 in time_throughput_all:
    #             plt.figure(figsize=(10, 5))
    #             plt.plot(throughput_time_post_list[i], throughput_df_post_list[i]["throughput_Mbps"], marker='o', label=f'Post Download Run {i+1}')
    #             plt.plot(time_throughput_all[i+1], throughput_all[i+1], marker='x', label=f'Search Throughput Run {i+1}')
    #             plt.plot(throughput_time_pre_list[i], throughput_df_pre_list[i]["throughput_Mbps"], marker='o', label=f'Pre Download Run {i+1}')
    #             plt.xlabel('Time (s)', fontsize=16)
    #             plt.ylabel('Throughput (Mb/s)', fontsize=16)
    #             plt.title('Throughput of Post and pre Download and Search Over Time')
    #             plt.legend()
    #             plt.xticks(fontsize=14)
    #             plt.yticks(fontsize=14)
    #             plt.xlim([-1, 11])
    #             plt.ylim(bottom=0)  # Set y-axis to start from 0
    #             plt.tight_layout()
    #             plt.savefig(os.path.join(output_path, f"throughput_post_pre_download_and_search_{i+1}.png"))
    #             plt.close()

    # # plot throughput of search and throughput of post download on one graph
    # if throughput_df_post_list and throughput_all:
    #     for i in range(len(throughput_df_post_list)):
    #         if i+1 in time_throughput_all:
    #             plt.figure(figsize=(10, 5))
    #             plt.plot(throughput_time_post_list[i], throughput_df_post_list[i]["throughput_Mbps"], marker='o', label=f'Post Download Run {i+1}')
    #             plt.plot(time_throughput_all[i+1], throughput_all[i+1], marker='x', label=f'Search Throughput Run {i+1}')
    #             plt.xlabel('Time (s)', fontsize=16)
    #             plt.ylabel('Throughput (Mb/s)', fontsize=16)
    #             plt.title('Throughput of Post Download and Search Over Time')
    #             plt.legend()
    #             plt.xticks(fontsize=14)
    #             plt.yticks(fontsize=14)
    #             plt.xlim([-1, 11])
    #             plt.ylim(bottom=0)  # Set y-axis to start from 0
    #             plt.tight_layout()
    #             plt.savefig(os.path.join(output_path, f"throughput_post_download_and_search_{i+1}.png"))
    #             plt.close()

# Save all data to a CSV file
# if avg_throughputs and median_throughputs and avg_delivery_rates and median_delivery_rates and delivery_rate_at_exit and all_ss_exit_time_list:
#     df_results = pd.DataFrame({
#         "Run": range(1, num + 2),
#         "Average Throughput (Mb/s)": avg_throughputs,
#         "Median Throughput (Mb/s)": median_throughputs,
#         "Average Delivery Rate (Mb/s)": avg_delivery_rates,
#         "Median Delivery Rate (Mb/s)": median_delivery_rates,
#         "Delivery Rate at Exit (Mb/s)": delivery_rate_at_exit,
#         "Exit Time (s)": all_ss_exit_time_list
#     })
#     results_csv_path = os.path.join(output_path, "speedtest_analysis_results.csv")
#     df_results.to_csv(results_csv_path, index=False)
