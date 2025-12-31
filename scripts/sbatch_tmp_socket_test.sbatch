#!/bin/bash
#SBATCH --job-name=tmp-socket-smoke
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:02:00
#SBATCH --output=tulupipe_logs/tmp_socket_%A.out
#SBATCH --error=tulupipe_logs/tmp_socket_%A.err

set -euo pipefail

cd /home/hice1/sbhansali8/scratch/FederatedScope
mkdir -p tulupipe_logs

python - <<'PY'
import os
import socket
import tempfile

def try_socket(path):
    if os.path.exists(path):
        os.remove(path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(path)
        print("socket_ok:", path)
        return True
    except Exception as exc:
        print("socket_fail:", path, exc)
        return False
    finally:
        sock.close()
        if os.path.exists(path):
            os.remove(path)

tmpdir = tempfile.gettempdir()
print("tempfile.gettempdir():", tmpdir)
try_socket(os.path.join(tmpdir, "tmp_socket_smoke.sock"))

custom = os.path.join(os.getcwd(), "tulupipe_smoke", "tmp")
os.makedirs(custom, exist_ok=True)
try_socket(os.path.join(custom, "custom_socket_smoke.sock"))
PY
