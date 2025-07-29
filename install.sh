#!/bin/bash
set -e

sudo apt update
sudo apt upgrade -y
#sudo apt install -y build-essential nvidia-cuda-toolkit

curl -Ls https://astral.sh/uv/install.sh | bash
export PATH="$HOME/.cargo/bin:$PATH"

if ! command -v uv &> /dev/null; then
  echo "uv not found in PATH"
  exit 1
fi

uv venv --python python3.11 .venv --clear
source .venv/bin/activate

uv pip install -r pyproject.toml