# 2026-11-04: prompt gallery, serve without the text encoder (RTX 4090)

Vast instance 59040749: RTX 4090, Michigan, $0.394/h. 4-bit DiT file
(`int4-hqq-g64-ffn`, 1.18 GB) for everything below.

## `fastwan encode`

Six prompts (fox, night market, sea cliffs, pour-over, koi, puppy), 28–35
tokens each: 11.6–14.8 s per prompt with the native UMT5 on the box's CPU.
The gallery is 1.5 MB in all (f16, real tokens only).

## Gallery embeddings vs the live encoder

Same prompt, seed 7, CUDA: `generate --embeds <gallery file>` vs
`generate --prompt` with the text encoder: **39.4 dB** (Y 35.5). The only
difference is f16 storage of the embeddings; for scale, CUDA vs WebGPU on the
same weights differ by 33.5 dB. Both 7.0 s per clip (1.39 s/step).

## `fastwan serve --gallery`

Engine ready in 4.8 s with no text encoder loaded (6.0 s for the full engine,
whose encoder is mmapped lazily). Reached from a browser through Cloudflare
quick tunnels: the page loads, `/api/info` lists the six prompts, and a prompt
outside the gallery gets HTTP 400 before it is queued.

`gallery-page-synthetic.png`: the page in gallery mode (local run against
`--synthetic --gallery`): read-only prompt, gallery chips, note.

Found on the way: the CSRF guard accepted only `http://<Host>` as same-origin,
so the engine's own page behind an https tunnel got 403 on generate (fixed in
0ac4556; this rental used `--allow-origin <tunnel URL>` instead).
