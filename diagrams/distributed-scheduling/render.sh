#!/usr/bin/env bash
# 把 diagram-*.html 用 Chrome headless 截图导出为 PNG。
# 用法: ./render.sh <name> <width> <height>
# Chrome 截图后进程会挂起，用 timeout 兜底，PNG 在超时前已写出。
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/../../public/assets/posts/distributed-scheduling"
mkdir -p "$OUT"
name="$1"; w="$2"; h="$3"
P="$(mktemp -d)"
timeout 30 google-chrome --headless=new --no-sandbox --disable-gpu \
  --disable-dev-shm-usage --hide-scrollbars --no-first-run \
  --virtual-time-budget=6000 --user-data-dir="$P" \
  --force-device-scale-factor=2 --window-size="$w,$h" \
  --screenshot="$OUT/$name.png" "file://$DIR/$name.html" >/dev/null 2>&1
python3 -c "from PIL import Image; print('$name', Image.open('$OUT/$name.png').size)"
