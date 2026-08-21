# AWS WireGuard Campus Remote Access Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not dispatch subagents: this deployment crosses a user-controlled AWS console, a Mac, a phone, and a shared campus server, so each checkpoint needs the user's direct confirmation.

**Goal:** Build a reversible split-tunnel path that lets the user's phone, while on mobile data, reach the campus server terminal and control the server's Codex CLI through ChatGPT Remote without exposing the campus server to public inbound traffic.

**Architecture:** The phone and campus server are WireGuard peers of one AWS EC2 relay. The campus server initiates an outbound UDP tunnel to AWS; the phone routes only the WireGuard subnet and the single campus-server address through AWS. Normal ChatGPT and mobile traffic remain on the phone's ordinary connection.

**Tech Stack:** AWS EC2 in `ap-southeast-2`, Elastic IP, security groups, WireGuard, systemd, Amazon Linux or Ubuntu on EC2, Ubuntu 22.04 on the campus server, WireGuard mobile app, OpenSSH, Codex CLI `0.144.1` Remote Control.

---

## 1. Fixed deployment facts

| Item | Fixed value |
|---|---|
| AWS region | `ap-southeast-2` (Sydney) |
| EC2 name | `myec` |
| EC2 instance ID | `i-0e11b2699f7bded19` |
| EC2 public/EIP address | `54.206.124.213` |
| EC2 public DNS | `ec2-54-206-124-213.ap-southeast-2.compute.amazonaws.com` |
| Expected EC2 login user | `ec2-user`; verify from `/etc/os-release` after login |
| Mac SSH key | `$HOME/Downloads/pass.pem` |
| Mac direct public IPv4 | `123.112.10.42/32` at the time of planning |
| Mac local proxy symptom | ordinary `ssh` is intercepted by `127.0.0.1:7897` |
| Campus server user | `zcy` |
| Campus server IPv4 | `10.25.144.116/24` |
| Campus server OS | Ubuntu 22.04 |
| AWS WireGuard address | `10.77.0.1/24` |
| Phone WireGuard address | `10.77.0.2/32` |
| Campus WireGuard address | `10.77.0.3/32` |
| WireGuard listen port | UDP `51820` |

The EC2 public IP must be confirmed as an associated Elastic IP before deployment. If the address changes, update every occurrence of `54.206.124.213` in the runtime commands and phone profile before starting WireGuard.

## 2. Safety contract

- Never paste a WireGuard private key, `pass.pem`, AWS credential, ChatGPT token, or pairing code into this repository, a chat, a command log, or a screenshot.
- Record only the three WireGuard public keys during execution.
- Keep EC2 SSH restricted to the Mac's current direct public IP `/32`. Never leave TCP 22 open to `0.0.0.0/0`.
- UDP 51820 may use source `0.0.0.0/0` because mobile carrier addresses change; WireGuard still requires a valid peer key.
- Do not change the campus server's default route. Do not modify its existing `mihomo` service or ports `7897`/`9097`.
- Do not stop training jobs, reboot the shared campus server, terminate the EC2 instance, release the Elastic IP, or delete keys without separate user approval.
- Stop at the first failed checkpoint. Collect the prescribed evidence before changing another variable.
- Commands under **Campus server** must retain the repository rule and use `rtk`. Commands under **Mac** and **EC2** run on machines where `rtk` is not assumed to exist and therefore use the native command directly.

## 3. Files and resources changed

No project source code is modified during deployment. The executor creates or modifies only:

- EC2: `/etc/wireguard/private.key`
- EC2: `/etc/wireguard/public.key`
- EC2: `/etc/wireguard/wg0.conf`
- EC2: `/etc/sysctl.d/99-wireguard-forward.conf`
- Campus server: `/etc/wireguard/private.key`
- Campus server: `/etc/wireguard/public.key`
- Campus server: `/etc/wireguard/wg0.conf`
- Phone: one WireGuard profile named `AWS-Campus`
- AWS: one Elastic IP association and security-group rules for TCP 22 and UDP 51820

Do not store generated keys or copied configuration files under `/data1/zcy/OV-CapFlow`.

---

## Task 1: Finish and verify the AWS control plane

- [ ] **Step 1: Verify the Elastic IP association**

In AWS Console, open **EC2 → Instances → myec**. Confirm that the **Elastic IP** column contains `54.206.124.213`, not `-`.

Expected: the instance public DNS is `ec2-54-206-124-213.ap-southeast-2.compute.amazonaws.com` and the Elastic IP is `54.206.124.213`.

If the Elastic IP column is `-`, open **Network & Security → Elastic IPs**, allocate an address, and associate it with instance `i-0e11b2699f7bded19`. If AWS allocates a different address, stop and update the runtime value before continuing.

- [ ] **Step 2: Normalize the security-group inbound rules**

For the security group attached to `myec`, set:

| Protocol | Port | Source | Purpose |
|---|---:|---|---|
| TCP | 22 | `123.112.10.42/32` | Temporary Mac administration |
| UDP | 51820 | `0.0.0.0/0` | WireGuard peers |

Remove TCP 80 and TCP 443 rules if this EC2 instance does not host a website. Leave outbound rules at their AWS default.

Expected: no public TCP management port except TCP 22 restricted to one `/32` address.

- [ ] **Step 3: Refresh the Mac's direct address before SSH**

Run on the Mac:

```bash
/usr/bin/curl -4 --noproxy '*' --connect-timeout 8 https://checkip.amazonaws.com
```

Expected: `123.112.10.42`.

If it differs, update the TCP 22 security-group source to the new result plus `/32`. Do not use the AWS console's **My IP** shortcut when the browser is proxied; it may return the proxy exit address instead of the Mac's direct address.

## Task 2: Establish direct Mac-to-EC2 SSH access

- [ ] **Step 1: Fix local key permissions**

Run on the Mac:

```bash
/bin/chmod 400 "$HOME/Downloads/pass.pem"
```

Expected: no output.

- [ ] **Step 2: Bypass the Mac's SSH wrapper and local proxy**

Run on the Mac:

```bash
/usr/bin/ssh \
  -F /dev/null \
  -o ProxyCommand=none \
  -o ProxyJump=none \
  -o ConnectTimeout=10 \
  -i "$HOME/Downloads/pass.pem" \
  ec2-user@54.206.124.213
```

Expected on first connection: a host-key confirmation prompt. Confirm only if the destination shown is `54.206.124.213`, then expect an EC2 shell prompt.

Failure gates:

- `Connection timed out`: re-run the direct-IP check from Task 1 and correct the TCP 22 source. If it still times out, test once from a phone hotspot to distinguish campus egress filtering from AWS configuration.
- `Permission denied (publickey)`: verify that `pass.pem` is the key pair assigned to instance `i-0e11b2699f7bded19`. If the AMI is Ubuntu, retry once with user `ubuntu`; do not try random usernames.
- Any reference to `127.0.0.1:7897`: confirm that the command begins with the absolute binary `/usr/bin/ssh` and includes `-F /dev/null`.

- [ ] **Step 3: Identify the EC2 operating system**

Run after login:

```bash
cat /etc/os-release
uname -r
```

Expected: either Amazon Linux 2023 (`ID="amzn"`) or Ubuntu. Record only the OS family, not the full terminal transcript.

## Task 3: Install WireGuard and generate the EC2 key pair

- [ ] **Step 1: Install the OS-appropriate WireGuard package**

For Amazon Linux 2023:

```bash
sudo dnf install -y wireguard-tools
```

For Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y wireguard
```

Run exactly one branch based on `/etc/os-release`.

Verify:

```bash
wg --version
```

Expected: a WireGuard tools version, not `command not found`.

- [ ] **Step 2: Create the EC2 key pair without printing the private key**

```bash
sudo install -d -m 700 /etc/wireguard
sudo sh -c 'umask 077; wg genkey > /etc/wireguard/private.key; wg pubkey < /etc/wireguard/private.key > /etc/wireguard/public.key'
sudo chmod 600 /etc/wireguard/private.key
sudo chmod 644 /etc/wireguard/public.key
sudo cat /etc/wireguard/public.key
```

Expected: one base64 public key. Record it in the executor's private runtime state as `AWS_PUBLIC_KEY`. Do not run `cat` on `private.key`.

## Task 4: Install WireGuard and generate the campus-server key pair

- [ ] **Step 1: Check whether WireGuard is already installed**

Run on the campus server:

```bash
rtk wg --version
```

If the command is missing, run:

```bash
rtk sudo apt-get update
rtk sudo apt-get install -y wireguard
```

Expected: WireGuard tools are available without rebooting the campus server.

- [ ] **Step 2: Check for an existing key before generating one**

```bash
rtk sudo ls -l /etc/wireguard/private.key /etc/wireguard/public.key
```

If both files exist and are nonempty, preserve them and print only the public key:

```bash
rtk sudo cat /etc/wireguard/public.key
```

If they do not exist, generate them:

```bash
rtk sudo install -d -m 700 /etc/wireguard
rtk sudo sh -c 'umask 077; wg genkey > /etc/wireguard/private.key; wg pubkey < /etc/wireguard/private.key > /etc/wireguard/public.key'
rtk sudo chmod 600 /etc/wireguard/private.key
rtk sudo chmod 644 /etc/wireguard/public.key
rtk sudo cat /etc/wireguard/public.key
```

Expected: one base64 public key. Record it as `CAMPUS_PUBLIC_KEY`. Never regenerate a working key pair during a retry unless the deployment explicitly enters key-rotation recovery.

## Task 5: Create the phone peer and collect its public key

- [ ] **Step 1: Create an incomplete phone profile**

In the official WireGuard mobile app:

1. Add a tunnel from scratch.
2. Name it `AWS-Campus`.
3. Set the interface address to `10.77.0.2/32`.
4. Leave DNS empty.
5. Let the app generate its key pair.

- [ ] **Step 2: Provide only the phone public key to the executor**

Copy the value labeled **Public key** and record it as `PHONE_PUBLIC_KEY`. Do not export or share the phone private key.

This is a user-input gate. The executor must not continue until `AWS_PUBLIC_KEY`, `CAMPUS_PUBLIC_KEY`, and `PHONE_PUBLIC_KEY` are all available and each decodes as a 32-byte WireGuard public key.

## Task 6: Configure and start the EC2 relay

- [ ] **Step 1: Enable IPv4 forwarding persistently**

Run on EC2:

```bash
printf '%s\n' 'net.ipv4.ip_forward=1' | sudo tee /etc/sysctl.d/99-wireguard-forward.conf
sudo sysctl --system
sysctl net.ipv4.ip_forward
```

Expected final line:

```text
net.ipv4.ip_forward = 1
```

- [ ] **Step 2: Build `/etc/wireguard/wg0.conf` without exposing the private key**

On EC2, enter the two peer public keys when prompted:

```bash
read -r -p 'PHONE_PUBLIC_KEY: ' PHONE_PUBLIC_KEY
read -r -p 'CAMPUS_PUBLIC_KEY: ' CAMPUS_PUBLIC_KEY
{
  printf "%s\n" "[Interface]"
  printf "%s\n" "Address = 10.77.0.1/24"
  printf "%s\n" "ListenPort = 51820"
  printf "%s\n" "PostUp = wg set %i private-key /etc/wireguard/private.key; iptables -A FORWARD -i %i -o %i -j ACCEPT"
  printf "%s\n\n" "PostDown = iptables -D FORWARD -i %i -o %i -j ACCEPT"
  printf "%s\n" "[Peer]"
  printf "%s\n" "PublicKey = $PHONE_PUBLIC_KEY"
  printf "%s\n\n" "AllowedIPs = 10.77.0.2/32"
  printf "%s\n" "[Peer]"
  printf "%s\n" "PublicKey = $CAMPUS_PUBLIC_KEY"
  printf "%s\n" "AllowedIPs = 10.77.0.3/32, 10.25.144.116/32"
} | sudo tee /etc/wireguard/wg0.conf >/dev/null
sudo chmod 600 /etc/wireguard/wg0.conf
unset PHONE_PUBLIC_KEY CAMPUS_PUBLIC_KEY
sudo wg-quick strip wg0
```

Expected: `wg-quick strip` prints a syntactically valid WireGuard configuration with two peers. It will display public keys but must not be copied into the repository.

- [ ] **Step 3: Start the relay and verify local state**

```bash
sudo systemctl enable --now wg-quick@wg0
sudo systemctl --no-pager --full status wg-quick@wg0
sudo wg show
ip -brief address show wg0
```

Expected:

- service state is `active (exited)`;
- `wg0` owns `10.77.0.1/24`;
- two peers are listed;
- no handshake is expected until the campus server and phone are started.

## Task 7: Configure and start the campus peer

- [ ] **Step 1: Create the campus configuration without exposing its private key**

Run on the campus server and paste only `AWS_PUBLIC_KEY` when prompted:

```bash
rtk bash -lc '
read -r -p "AWS_PUBLIC_KEY: " AWS_PUBLIC_KEY
{
  printf "%s\n" "[Interface]"
  printf "%s\n" "Address = 10.77.0.3/32"
  printf "%s\n" "PostUp = wg set %i private-key /etc/wireguard/private.key"
  printf "%s\n\n" "[Peer]"
  printf "%s\n" "PublicKey = $AWS_PUBLIC_KEY"
  printf "%s\n" "Endpoint = 54.206.124.213:51820"
  printf "%s\n" "AllowedIPs = 10.77.0.0/24"
  printf "%s\n" "PersistentKeepalive = 25"
} | sudo tee /etc/wireguard/wg0.conf >/dev/null
sudo chmod 600 /etc/wireguard/wg0.conf
unset AWS_PUBLIC_KEY
'
rtk sudo wg-quick strip wg0
```

Expected: a valid configuration with interface address `10.77.0.3/32`, endpoint `54.206.124.213:51820`, and no default route.

- [ ] **Step 2: Start the peer and verify the first handshake**

```bash
rtk sudo systemctl enable --now wg-quick@wg0
rtk sudo systemctl --no-pager --full status wg-quick@wg0
rtk sudo wg show
rtk ip -brief address show wg0
rtk ping -c 3 10.77.0.1
```

Expected:

- `wg0` owns `10.77.0.3/32`;
- `latest handshake` is recent;
- transfer counters are nonzero;
- ping to `10.77.0.1` succeeds.

If no handshake appears, stop here and use the failure matrix. Do not configure random additional ports.

## Task 8: Complete and activate the phone profile

- [ ] **Step 1: Add the AWS peer in the phone app**

Set the peer fields as follows. For **Public key**, paste the exact `AWS_PUBLIC_KEY` collected in Task 3:

```ini
Endpoint = 54.206.124.213:51820
AllowedIPs = 10.77.0.0/24, 10.25.144.116/32
PersistentKeepalive = 25
```

Do not paste any private key into the peer fields. Do not set `0.0.0.0/0`; this plan is intentionally split-tunnel.

- [ ] **Step 2: Activate under the real external-network condition**

1. Disable campus Wi-Fi on the phone.
2. Use 4G/5G.
3. Confirm ordinary ChatGPT conversation still works before enabling WireGuard.
4. Enable `AWS-Campus`.
5. Confirm the WireGuard app reports a recent handshake and nonzero transfer counters.

Expected: normal public traffic remains on mobile data; only `10.77.0.0/24` and `10.25.144.116/32` enter the tunnel.

## Task 9: Validate terminal and Codex Remote access

- [ ] **Step 1: Validate the phone-to-server network path**

Using the phone's existing SSH client and existing campus-server SSH credentials, connect to:

```text
Host: 10.77.0.3
Port: 22
User: zcy
```

Expected: a campus-server shell opens over mobile data.

If the server uses UFW, inspect it before changing anything:

```bash
rtk sudo ufw status verbose
```

Only if UFW is active and explicitly blocks the WireGuard subnet, add:

```bash
rtk sudo ufw allow from 10.77.0.0/24 to any port 22 proto tcp
```

- [ ] **Step 2: Reconfirm Codex's managed app-server state**

Run on the campus server:

```bash
rtk codex app-server daemon version
```

Expected JSON fields:

```text
"status":"running"
"cliVersion":"0.144.1"
"appServerVersion":"0.144.1"
```

If the daemon is not running:

```bash
rtk codex app-server daemon start
rtk codex app-server daemon enable-remote-control
```

- [ ] **Step 3: Validate ChatGPT Remote from mobile data**

With the phone still on 4G/5G and `AWS-Campus` enabled:

1. Open ChatGPT.
2. Open **Remote**.
3. Select the existing campus-server host.
4. Send a read-only prompt such as `pwd` or request the current repository path.
5. Confirm that terminal output returns from `/data1/zcy/OV-CapFlow`.

If the host is no longer paired, generate a short-lived pairing code on the campus server:

```bash
rtk codex remote-control pair
```

Enter the pairing code only in the user's ChatGPT app. Do not save it.

- [ ] **Step 4: Record the deployment result without secrets**

Record only:

- deployment date and time;
- EC2 instance ID and Elastic IP;
- recent handshake observed for campus and phone peers;
- phone SSH result;
- ChatGPT Remote result;
- any failure category from the matrix below.

Do not record key material, pairing codes, SSH host keys, AWS account identifiers, or full network dumps.

## Task 10: Apply the failure matrix one boundary at a time

### A. Mac cannot SSH to EC2

Evidence command on Mac:

```bash
/usr/bin/curl -4 --noproxy '*' --connect-timeout 8 https://checkip.amazonaws.com
/usr/bin/ssh -vv -F /dev/null -o ProxyCommand=none -o ProxyJump=none -o ConnectTimeout=10 -i "$HOME/Downloads/pass.pem" ec2-user@54.206.124.213
```

- `127.0.0.1:7897`: the wrong SSH binary/wrapper was used.
- Timeout before banner: direct IP does not match the security group, campus egress blocks TCP 22, or the EIP is not associated.
- SSH banner followed by public-key denial: username/key mismatch, not a network problem.

### B. Campus peer has no WireGuard handshake

Collect:

```bash
rtk sudo wg show
rtk ip route get 54.206.124.213
rtk sudo journalctl -u wg-quick@wg0 -n 80 --no-pager
```

On EC2 collect:

```bash
sudo wg show
sudo journalctl -u wg-quick@wg0 -n 80 --no-pager
```

Check, in order: EIP, UDP 51820 security-group rule, AWS public key on campus, campus public key on AWS, endpoint port, then campus UDP egress. If all configuration values match and only UDP egress remains suspect, test a single controlled alternative using UDP 443 on both sides and in the security group. Do not switch to TCP tunneling in the same test.

### C. WireGuard handshakes but phone SSH fails

Check:

```bash
rtk ip -brief address show wg0
rtk ip route get 10.77.0.2
rtk ss -ltn
rtk sudo ufw status verbose
```

Expected: `wg0` is up, the route to `10.77.0.2` uses `wg0`, SSH listens on port 22, and host firewall permits `10.77.0.0/24`.

### D. Phone SSH works but ChatGPT Remote fails

This proves AWS, WireGuard, routing, and SSH are working. Do not open additional random AWS ports. The remaining fault is Codex Remote discovery/transport behavior.

At that point:

1. Confirm the managed app server and Remote Control are enabled.
2. Re-pair once.
3. If failure persists, capture one same-campus successful Remote connection and one WireGuard failure with a tightly scoped packet capture to identify the actual destination address/protocol.
4. Decide from evidence whether routed unicast is sufficient or whether a discovery relay is needed.

## Task 11: Roll back safely

- [ ] **Step 1: Disable the phone tunnel**

Turn off `AWS-Campus`. This immediately restores the phone's pre-deployment routing.

- [ ] **Step 2: Stop WireGuard on the campus server**

```bash
rtk sudo systemctl disable --now wg-quick@wg0
rtk ip -brief address show wg0
```

Expected: `wg0` is absent. The campus default route and `mihomo` remain unchanged.

- [ ] **Step 3: Stop WireGuard on EC2**

```bash
sudo systemctl disable --now wg-quick@wg0
ip -brief address show wg0
```

Expected: `wg0` is absent.

- [ ] **Step 4: Close AWS ingress**

Remove the UDP 51820 rule. Keep or remove the restricted TCP 22 rule according to whether the EC2 instance will still be administered.

Do not terminate the instance or release the Elastic IP automatically. Those actions affect billing and recoverability and require explicit user confirmation.

---

## Completion criteria

The deployment is complete only when all conditions hold:

- EC2 uses an associated Elastic IP and exposes only the intended security-group rules.
- Mac can administer EC2 using direct `/usr/bin/ssh` without `127.0.0.1:7897` interception.
- EC2 sees recent handshakes from both campus and phone peers.
- Campus server sees a recent handshake and can reach `10.77.0.1`.
- Phone on 4G/5G can SSH to `zcy@10.77.0.3`.
- Ordinary ChatGPT conversation still works with split tunneling.
- ChatGPT Remote can reach and control the existing Codex CLI host.
- No private key, pairing code, or credential has been persisted outside its owning device.

## References

- AWS EC2 SSH connection guide: <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect-linux-inst-ssh.html>
- AWS security-group guidance: <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/changing-security-group.html>
- AWS Elastic IP guide: <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/elastic-ip-addresses-eip.html>
- WireGuard quick start and persistent keepalive: <https://www.wireguard.com/quickstart/>
- AWS CloudShell WebSocket timeout troubleshooting: <https://docs.aws.amazon.com/cloudshell/latest/userguide/troubleshooting.html>
