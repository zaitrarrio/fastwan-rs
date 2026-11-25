"""Write and upload the rental's bootstrap script; print the onstart line.

    FASTWAN_RUN=vast-20261103-1445 python3 make_boot.py BIN_KEY

The bootstrap installs curl / Vulkan / X11-GL libs / ffmpeg, pulls the
FastWan 2.1 1.3B tree and taew2_1 from the `fv-weights` bucket with presigned
URLs (R2 egress is free; Vast bills the download per GB; with the static
`weights` CLI uploaded to `<run>/bin/weights`, weights-rs's release binary, it
is one ranged, parallel, resumable transfer, otherwise one curl per file),
fetches the fastwan
binary from `<run>/BIN_KEY`, then runs `<run>/cmd/N.sh` for N = 1..40 as they
appear, uploading `<run>/out/N.log` after each. The X11/GL libraries matter:
without them NVIDIA's Vulkan ICD (libGLX_nvidia) fails to load and wgpu
silently falls back to llvmpipe.
"""
import os, shlex, sys
exec(open(os.path.join(os.path.dirname(__file__), "r2.py")).read())
run = os.environ["FASTWAN_RUN"]; B = os.environ.get("FASTWAN_BUCKET", "fastwan-runs"); EXP = 6 * 3600
SRC = "weights/fastwan21-1.3b/25e7ed7f41fd8ce2fdd108688c65e8caf0ce3aef/"
TAE = "weights/auxiliary/tae/eu-20261011/taew2_1.safetensors"
get = lambda b, k: s3.generate_presigned_url("get_object", Params={"Bucket": b, "Key": k}, ExpiresIn=EXP)
put = lambda k: s3.generate_presigned_url("put_object", Params={"Bucket": B, "Key": k}, ExpiresIn=EXP)
files = []
for page in s3.get_paginator("list_objects_v2").paginate(Bucket="fv-weights", Prefix=SRC):
    for o in page.get("Contents", []):
        rel = o["Key"][len(SRC):]
        if rel.split("/")[0] in ("transformer", "text_encoder", "tokenizer"):
            files.append((get("fv-weights", o["Key"]), o["Size"], "/w/fastwan/" + rel))
tae = s3.head_object(Bucket="fv-weights", Key=TAE)["ContentLength"]
files.append((get("fv-weights", TAE), tae, "/w/fastwan/taew2_1.safetensors"))
WBIN = os.environ.get("FASTWAN_WEIGHTS_BIN", "bin/weights")
try:
    s3.head_object(Bucket=B, Key=f"{run}/{WBIN}")
    wbin_url = get(B, f"{run}/{WBIN}")
except Exception:
    wbin_url = None
print("weights CLI: " + ("yes" if wbin_url else f"no ({run}/{WBIN} not uploaded; curl per file)"), file=sys.stderr)
N = 40
L = ["#!/bin/bash", "set -u", "LOG=/root/boot.log", "exec >>$LOG 2>&1",
     f"LOGPUT={shlex.quote(put(run + '/boot.log'))}",
     'up() { curl -s -X PUT --upload-file "$1" "$2" >/dev/null; }',
     '(while true; do sleep 20; up $LOG "$LOGPUT"; done) &',
     'echo "== boot $(date -u)"', "nvidia-smi", "free -g; nproc; df -h /",
     "export DEBIAN_FRONTEND=noninteractive",
     "apt-get update -qq && apt-get install -y -qq curl ca-certificates libvulkan1 vulkan-tools ffmpeg "
     "libx11-6 libxext6 libegl1 libgl1 libglvnd0 >/dev/null && echo 'apt ok'",
     "mkdir -p /w/fastwan/transformer /w/fastwan/text_encoder /w/fastwan/tokenizer",
     "T0=$(date +%s)", "cat > /root/files.txt <<'EOF2'"]
L += [f"{u} {z} {p}" for u, z, p in files]
L += ["EOF2",
      (f"curl -sf -o /w/weights {shlex.quote(wbin_url)} && chmod +x /w/weights && "
       "{ /w/weights plan /root/files.txt -j 32 --progress 30 || echo 'FAILED weights plan'; } || " if wbin_url else "")
      + "awk '{print $1, $3}' /root/files.txt | xargs -P 8 -n 2 sh -c 'curl -sf --retry 5 -o \"$1\" \"$0\" || echo \"FAILED $1\"'",
      'echo "weights $(du -sh /w/fastwan | cut -f1) in $(( $(date +%s) - T0 ))s"',
      f"curl -sf -o /w/fastwan-bin {shlex.quote(get(B, run + '/' + sys.argv[1]))} && chmod +x /w/fastwan-bin",
      'up $LOG "$LOGPUT"', "echo READY",
      "CMD=(" + " ".join(shlex.quote(get(B, f"{run}/cmd/{i}.sh")) for i in range(1, N + 1)) + ")",
      "OUT=(" + " ".join(shlex.quote(put(f"{run}/out/{i}.log")) for i in range(1, N + 1)) + ")",
      f"for i in $(seq 0 {N - 1}); do",
      '  until curl -sf -o /root/cmd.sh "${CMD[$i]}"; do sleep 10; done',
      '  echo "== cmd $((i+1)) $(date -u)"',
      '  (cd /w && timeout 3000 bash /root/cmd.sh) > /root/out.log 2>&1; echo "exit $?" >> /root/out.log',
      '  up /root/out.log "${OUT[$i]}"; up $LOG "$LOGPUT"', "done"]
path = "/tmp/fastwan-boot.sh"
open(path, "w").write("\n".join(L) + "\n")
s3.upload_file(path, B, run + "/boot.sh")
url = get(B, run + "/boot.sh")
print("export DEBIAN_FRONTEND=noninteractive; (apt-get update -qq && apt-get install -y -qq curl ca-certificates) >/root/pre.log 2>&1; "
      f"curl -sf -o /root/boot.sh '{url}' && nohup bash /root/boot.sh >/dev/null 2>&1 &")
