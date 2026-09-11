#!/usr/bin/env bash
# 把 diagram-*.html 用 Chrome headless 截图导出为 PNG。
# 用法: ./render.sh <name> <width> <height>
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/../../public/assets/posts/pd-disaggregation"
mkdir -p "$OUT"
name="$1"; w="$2"; h="$3"
P="$(mktemp -d)"
timeout 30 google-chrome --headless=new --no-sandbox --disable-gpu \
  --disable-dev-shm-usage --hide-scrollbars --no-first-run \
  --virtual-time-budget=6000 --user-data-dir="$P" \
  --force-device-scale-factor=2 --window-size="$w,$h" \
  --screenshot="$OUT/$name.png" "file://$DIR/$name.html" >/dev/null 2>&1
python3 -c "from PIL import Image; print('$name', Image.open('$OUT/$name.png').size)"
