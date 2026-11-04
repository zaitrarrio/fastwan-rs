# Running on Vast.ai without SSH

The session that built this repo can only make outbound HTTPS calls, so a
rental is driven through Cloudflare R2 instead of SSH:

1. `make_boot.py BIN_KEY` writes the bootstrap to `fastwan-runs/<run>/boot.sh`
   and prints the instance's `onstart` line.
2. Rent with `vast_api.py` (`PUT /asks/<offer>/`, image
   `nvidia/cuda:12.8.1-runtime-ubuntu24.04`, `runtype: ssh`, env
   `NVIDIA_DRIVER_CAPABILITIES=all`, disk 100 GB). Arm a destroy timer
   immediately: instances bill until destroyed.
3. Queue work with `rr.py cmd N file.sh`; read it back with `rr.py out N`.
   Commands run one at a time, in order, with `/w/fastwan` holding the weights.
4. Destroy (`DELETE /instances/<id>/`) and check the credit.

Watchdog: `sleep N && <destroy>`, never `sleep N; <destroy>`. With `;`,
killing the sleep to cancel the timer runs the destroy at once (it took down
a booting rental on 2026-11-04).

To use a rental's `fastwan serve` from a browser without inbound ports, run a
Cloudflare quick tunnel on the box (`cloudflared tunnel --url
http://127.0.0.1:7860`). The URL is public: prefer `--gallery` mode, and
destroy the rental when done.

Env: `VAST_API_KEY`, `CLOUDFLARE_API_KEY` (token with R2 read/write),
`CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_TOKEN_ID`, `FASTWAN_RUN`. Needs boto3.

Filters that worked (2026-11-03): one RTX 4090, `cpu_ram >= 64 GB` (the fp32
UMT5 shards are 22 GB and the CPU encoder wants them in page cache),
`inet_down_cost <= 0.01 $/GB`, `reliability2 >= 0.98`, `verified`. The
27 GB weight pull took 101 s.

A command's output only arrives when it finishes, and there is no way to
interrupt a running command: give each one a `timeout`, and run the cheap,
decisive work first. The first run lost 45 minutes to a WebGPU benchmark that
could not finish one step in 20 minutes.
