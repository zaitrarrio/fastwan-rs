#!/usr/bin/env bash
# Build web/site and deploy it to the Cloudflare Pages project `fastwan`
# (https://fastwan.pages.dev). Needs CLOUDFLARE_API_TOKEN (Pages: Edit) and
# CLOUDFLARE_ACCOUNT_ID in the environment. The project already exists, so no
# --force: that flag was only needed once, to create it as a Pages project.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CLOUDFLARE_API_TOKEN:?set CLOUDFLARE_API_TOKEN}"
: "${CLOUDFLARE_ACCOUNT_ID:?set CLOUDFLARE_ACCOUNT_ID}"
python3 web/build.py
npx -y wrangler@4 pages deploy web/site --project-name fastwan --branch main --commit-dirty=true
