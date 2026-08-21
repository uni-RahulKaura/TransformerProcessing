#!/bin/bash
# One command to make a fresh box ready: install dq, fetch the pipeline, pull the documents
# from wherever they live, and list them.
#
#   curl -fsSL https://raw.githubusercontent.com/uni-RahulKaura/TransformerProcessing/main/go \
#     | bash -s -- s3://my-bucket/doc-index/docs.tgz
#
# The argument is any DQ_SRC: an S3 object or prefix, a mounted path, or a tarball URL. It is
# remembered in ~/.dq_src, so every later session is just:
#
#   curl -fsSL .../go | bash
#
# Then pick a document:  dq list   dq outline 13   dq run 13   dq show 13
set -euo pipefail
BASE="https://raw.githubusercontent.com/uni-RahulKaura/TransformerProcessing/main"

SRC="${1:-}"
if [ -z "$SRC" ] && [ -f "$HOME/.dq_src" ]; then SRC="$(tr -d ' \n\r' < "$HOME/.dq_src")"; fi
if [ -z "$SRC" ]; then
  echo "usage: curl -fsSL .../go | bash -s -- <where-the-documents-are>" >&2
  echo "  e.g. s3://my-bucket/doc-index/docs.tgz   or   /mnt/share/Test_Files" >&2
  exit 1
fi
printf '%s\n' "$SRC" > "$HOME/.dq_src"

BIN="$HOME/bin"; mkdir -p "$BIN"
curl -fsSL -o "$BIN/dq" "$BASE/dq"
chmod +x "$BIN/dq"
case ":$PATH:" in *":$BIN:"*) ;; *)
  export PATH="$BIN:$PATH"
  grep -qs "$BIN" "$HOME/.bashrc" 2>/dev/null || echo "export PATH=\"$BIN:\$PATH\"" >> "$HOME/.bashrc"
esac

# torch is needed for summaries but not for outlines, so a missing torch is a warning, not a stop
PY=""
for c in python3.12 python3.11 python3; do
  command -v "$c" >/dev/null 2>&1 || continue
  "$c" -c 'import sys;raise SystemExit(0 if sys.version_info>=(3,10) else 1)' 2>/dev/null && { PY="$c"; break; }
done
[ -z "$PY" ] && { echo "go: need python 3.10+" >&2; exit 1; }
if ! "$PY" -c 'import torch' 2>/dev/null; then
  echo "installing torch (a few minutes, once per container)"
  "$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
  "$PY" -m pip install --user --quiet torch transformers || \
    echo "go: torch install failed -- outlines will work, summaries will not"
fi

DQ_SRC="$SRC" DQ_PYTHON="$PY" "$BIN/dq" load
echo
"$BIN/dq" list
echo
echo "pick one:   dq outline 13      structure only, about a second"
echo "            dq run 13          outline + summary on the GPU"
echo "            dq run all         all of them"
echo "            dq show 13         print what it produced"
