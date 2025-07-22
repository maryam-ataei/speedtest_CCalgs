import paramiko
import time
import os
from datetime import datetime
import traceback

# Configuration
SERVER_IP = "130.215.28.249"
CLIENT_IP = "localhost"  # because you're connecting through the tunnel on cs.wpi.edu
CLIENT_PORT = 2222       # tunnel port
SERVER_USER = "mataeikachooei"
CLIENT_USER = "maryam"
SERVER_PASS = "Iric@@15975346"
CLIENT_PASS = "Wpi@@414"
# SERVER_IP = "130.215.28.249"
# CLIENT_IP = "glomma.cs.wpi.edu" # "localhost" (because you're connecting through the tunnel on cs.wpi.edu)
# CLIENT_PORT = 22    #2222 (tunnel port)
# SERVER_USER = "mataeikachooei"
# CLIENT_USER = "mataeikachooei" #maryam
# SERVER_PASS = "Iric@@15975346"
# CLIENT_PASS = "Iric@@15975346" #Wpi@@414
ITERATIONS = 52
LOG_DIR = os.path.expanduser("~/speedtest_results_fios_wireless")
os.makedirs(LOG_DIR, exist_ok=True)

# def setup_ssh(ip, username, password):
#     ssh = paramiko.SSHClient()
#     ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#     ssh.connect(ip, username=username, password=password)
#     return ssh
def setup_ssh(ip, username, password, port=22):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, port=port, username=username, password=password)
    return ssh


def ssh_run_cmd(ssh, cmd, sudo=False, ssh_password=None):
    shell = ssh.invoke_shell()
    time.sleep(0.1)
    if sudo:
        shell.send(f"echo {ssh_password} | sudo -S {cmd}\n")
    else:
        shell.send(cmd + "\n")
    time.sleep(0.1)
    output = ""
    while not shell.recv_ready():
        time.sleep(0.1)
    while shell.recv_ready():
        output += shell.recv(4096).decode('utf-8')
    return output

def ssh_exec_command(ssh, cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    output = stdout.read().decode()
    error = stderr.read().decode()
    return output + error

def run_test_case(i, server_ssh, client_ssh):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Running test case #{i} at {timestamp}")

    iter_dir = os.path.join(LOG_DIR, f"test_{i}_{timestamp}")
    os.makedirs(iter_dir, exist_ok=True)

    # === OOKLA Test Phase ===
    ssh_run_cmd(server_ssh, f"tcpdump -i eno1 -s 200 -w tcpdump_ookla_server_{i}.pcap &", sudo=True, ssh_password=SERVER_PASS)
    pcap_path = f"/home/maryam/tcpdump_ookla_client_{i}.pcap"
    ssh_run_cmd(
        client_ssh,
        f"nohup ~/run_tcpdump.sh '{CLIENT_PASS}' '{pcap_path}' > /dev/null 2>&1 &"
    )
    # time.sleep(0.2)

    # # Confirm tcpdump is running
    # tcpdump_check = ssh_exec_command(client_ssh, "pgrep -a tcpdump")
    # print(f"[DEBUG] Client tcpdump processes:\n{tcpdump_check}")

    # # Confirm output file is being written
    # check_cmd = f"ls -l {pcap_path}"
    # check_result = ssh_exec_command(client_ssh, check_cmd)
    # print(f"[DEBUG] Client pcap file check:\n{check_result}")
        
    ookla_cmd = (
        f"speedtest --secure --no-upload"
    )
    ookla_output = ssh_exec_command(client_ssh, ookla_cmd)
    with open(os.path.join(iter_dir, "ookla_client_output.txt"), "w") as f:
        f.write(ookla_output)

    # Stop tcpdump
    ssh_run_cmd(server_ssh, "pkill tcpdump", sudo=True, ssh_password=SERVER_PASS)
    ssh_run_cmd(client_ssh, "pkill tcpdump", sudo=True, ssh_password=CLIENT_PASS)
    time.sleep(0.2)

    # === NDT7 Test Phase ===
    ssh_run_cmd(server_ssh, f"tcpdump port 3001 -s 200 -w tcpdump_ndt7_server_{i}.pcap &", sudo=True, ssh_password=SERVER_PASS)
    pcap_path = f"/home/maryam/tcpdump_ndt7_client_{i}.pcap"
    ssh_run_cmd(
        client_ssh,
        f"nohup ~/run_tcpdump.sh '{CLIENT_PASS}' '{pcap_path}' > /dev/null 2>&1 &"
    )
    # time.sleep(0.2)

    # # Confirm tcpdump is running
    # tcpdump_check = ssh_exec_command(client_ssh, "pgrep -a tcpdump")
    # print(f"[DEBUG] Client tcpdump processes:\n{tcpdump_check}")

    # # Confirm output file is being written
    # check_cmd = f"ls -l {pcap_path}"
    # check_result = ssh_exec_command(client_ssh, check_cmd)
    # print(f"[DEBUG] Client pcap file check:\n{check_result}")
        

    ndt_cmd = (
        f"cd ~/ndt7-client-go/cmd/ndt7-client && "
        f"./ndt7-client -server {SERVER_IP}:3001 -scheme ws -no-verify -upload=false"
    )
    ndt_output = ssh_exec_command(client_ssh, ndt_cmd)
    with open(os.path.join(iter_dir, "ndt7_client_output.txt"), "w") as f:
        f.write(ndt_output)

    # Stop tcpdump
    ssh_run_cmd(server_ssh, "pkill tcpdump", sudo=True, ssh_password=SERVER_PASS)
    ssh_run_cmd(client_ssh, "pkill tcpdump", sudo=True, ssh_password=CLIENT_PASS)
    time.sleep(0.5)

    # === LIBRE Test Phase ===
    ssh_run_cmd(server_ssh, f"tcpdump -i eno1 -s 200 -w tcpdump_libre_server_{i}.pcap &", sudo=True, ssh_password=SERVER_PASS)
    pcap_path = f"/home/maryam/tcpdump_libre_client_{i}.pcap"
    ssh_run_cmd(
        client_ssh,
        f"nohup ~/run_tcpdump.sh '{CLIENT_PASS}' '{pcap_path}' > /dev/null 2>&1 &"
    )

    # time.sleep(0.5)

    # # Confirm tcpdump is running
    # tcpdump_check = ssh_exec_command(client_ssh, "pgrep -a tcpdump")
    # print(f"[DEBUG] Client tcpdump processes:\n{tcpdump_check}")

    # # Confirm output file is being written
    # check_cmd = f"ls -l {pcap_path}"
    # check_result = ssh_exec_command(client_ssh, check_cmd)
    # print(f"[DEBUG] Client pcap file check:\n{check_result}")
        

    libre_cmd = (
    "cd ~/libre/speedtest-cli && "
    "./librespeed-cli --local-json my_servers.json --server 1 --no-upload --json"
    )   

    libre_output = ssh_exec_command(client_ssh, libre_cmd)
    with open(os.path.join(iter_dir, "libre_client_output.txt"), "w") as f:
        f.write(libre_output)

    # Stop tcpdump
    ssh_run_cmd(server_ssh, "pkill tcpdump", sudo=True, ssh_password=SERVER_PASS)
    ssh_run_cmd(client_ssh, "pkill tcpdump", sudo=True, ssh_password=CLIENT_PASS)

    # Retrieve pcap files
    sftp_server = server_ssh.open_sftp()
    sftp_client = client_ssh.open_sftp()
    sftp_server.get(f"tcpdump_ookla_server_{i}.pcap", os.path.join(iter_dir, "tcpdump_ookla_server.pcap"))
    sftp_client.get(f"tcpdump_ookla_client_{i}.pcap", os.path.join(iter_dir, "tcpdump_ookla_client.pcap"))
    sftp_server.get(f"tcpdump_ndt7_server_{i}.pcap", os.path.join(iter_dir, "tcpdump_ndt7_server.pcap"))
    sftp_client.get(f"tcpdump_ndt7_client_{i}.pcap", os.path.join(iter_dir, "tcpdump_ndt7_client.pcap"))
    sftp_server.get(f"tcpdump_libre_server_{i}.pcap", os.path.join(iter_dir, "tcpdump_libre_server.pcap"))
    sftp_client.get(f"tcpdump_libre_client_{i}.pcap", os.path.join(iter_dir, "tcpdump_libre_client.pcap"))
    sftp_server.remove(f"tcpdump_ookla_server_{i}.pcap")
    sftp_client.remove(f"tcpdump_ookla_client_{i}.pcap")
    sftp_server.remove(f"tcpdump_ndt7_server_{i}.pcap")
    sftp_client.remove(f"tcpdump_ndt7_client_{i}.pcap")
    sftp_server.remove(f"tcpdump_libre_server_{i}.pcap")
    sftp_client.remove(f"tcpdump_libre_client_{i}.pcap")
    sftp_server.close()
    sftp_client.close()

    # === iPerf3 Test Phase ===
    ssh_run_cmd(server_ssh, "sudo sysctl -w net.ipv4.tcp_congestion_control=cubic_search", sudo=True, ssh_password=SERVER_PASS)

    # Save kernel/module config before iperf3
    config_output = ssh_exec_command(server_ssh, """
        echo "tcp_congestion_control: $(cat /proc/sys/net/ipv4/tcp_congestion_control)"
        echo "tcp_rmem: $(cat /proc/sys/net/ipv4/tcp_rmem)"
        echo "tcp_wmem: $(cat /proc/sys/net/ipv4/tcp_wmem)"
        echo "slow_start_mode: $(cat /sys/module/tcp_cubic_search/parameters/slow_start_mode)"
        echo "cwnd_set: $(cat /sys/module/tcp_cubic_search/parameters/cwnd_rollback)"
    """)
    with open(os.path.join(iter_dir, "server_tcp_config.txt"), "w") as f:
        f.write(config_output)

    # Start tcpdump
    ssh_run_cmd(server_ssh, f"tcpdump -i eno1 -s 200 -w tcpdump_iperf3_server_{i}.pcap &", sudo=True, ssh_password=SERVER_PASS)
    pcap_path = f"/home/maryam/tcpdump_iperf3_client_{i}.pcap"
    ssh_run_cmd(
        client_ssh,
        f"nohup ~/run_tcpdump.sh '{CLIENT_PASS}' '{pcap_path}' > /dev/null 2>&1 &"
    )

    # Extract log
    kernel_log_path = f"/tmp/kernel_ccrg_log_{i}.txt"
    session_name = f"tail_grep_session_{i}"

    ssh_run_cmd(
        server_ssh,
        f"screen -dmS {session_name} bash -c 'sudo tail -n 0 -f /var/log/kern.log | grep CCRG > {kernel_log_path}'",
        sudo=True,
        ssh_password=SERVER_PASS
    )


    # Run iperf3 server
    ssh_run_cmd(server_ssh, f"nohup iperf3 -s > iperf3_server_{i}.log 2>&1 &")
    time.sleep(0.5)

    # Run iperf3 client
    iperf_output = ssh_exec_command(client_ssh, f"iperf3 -c {SERVER_IP} -R -t 10")
    with open(os.path.join(iter_dir, "iperf3_client_output.txt"), "w") as f:
        f.write(iperf_output)

    # Wait briefly before cleanup
    time.sleep(2)

    # Stop iperf3, tcpdumps
    ssh_run_cmd(server_ssh, "pkill tcpdump", sudo=True, ssh_password=SERVER_PASS)
    ssh_run_cmd(client_ssh, "pkill tcpdump", sudo=True, ssh_password=CLIENT_PASS)
    ssh_run_cmd(server_ssh, "pkill -f 'iperf3 -s'")
    ssh_run_cmd(server_ssh, "pkill -f tail")

    # Stop the background screen session
    ssh_run_cmd(server_ssh, f"screen -S {session_name} -X quit")
    ssh_run_cmd(server_ssh, "pkill tail", sudo=True, ssh_password=SERVER_PASS)

    # Retrieve server log and pcap files
    sftp_server = server_ssh.open_sftp()
    sftp_client = client_ssh.open_sftp()
    sftp_server.get(f"iperf3_server_{i}.log", os.path.join(iter_dir, "iperf3_server_output.txt"))
    sftp_server.remove(f"iperf3_server_{i}.log")
    sftp_server.get(f"tcpdump_iperf3_server_{i}.pcap", os.path.join(iter_dir, "tcpdump_iperf3_server.pcap"))
    sftp_client.get(f"tcpdump_iperf3_client_{i}.pcap", os.path.join(iter_dir, "tcpdump_iperf3_client.pcap"))
    sftp_server.remove(f"tcpdump_iperf3_server_{i}.pcap")
    sftp_client.remove(f"tcpdump_iperf3_client_{i}.pcap")
    sftp_server.get(kernel_log_path, os.path.join(iter_dir, f"kernel_ccrg_log_{i}.txt"))
    #sftp_server.remove(kernel_log_path)
    sftp_server.close()
    sftp_client.close()

    time.sleep(5)

if __name__ == "__main__":
    server_ssh = setup_ssh(SERVER_IP, SERVER_USER, SERVER_PASS)
    # client_ssh = setup_ssh(CLIENT_IP, CLIENT_USER, CLIENT_PASS)
    client_ssh = setup_ssh(CLIENT_IP, CLIENT_USER, CLIENT_PASS, port=CLIENT_PORT)


    for i in range(1, ITERATIONS + 1):
        try:
            run_test_case(i, server_ssh, client_ssh)
        except Exception as e:
            traceback.print_exc()
            print(f"Error in iteration {i}: {e}")
    server_ssh.close()
    client_ssh.close()
    print("All tests completed.")
