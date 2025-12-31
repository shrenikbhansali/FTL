import os
import socket
import tempfile


def _check_unix_socket(tmpdir):
    os.makedirs(tmpdir, exist_ok=True)
    socket_path = os.path.join(tmpdir, "wandb_smoke.sock")
    if os.path.exists(socket_path):
        os.remove(socket_path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(socket_path)
    finally:
        sock.close()
    os.remove(socket_path)
    return socket_path


def _run_wandb():
    import wandb

    project = os.environ.get("WANDB_PROJECT", "FederatedScope-scripts-smoke")
    entity = os.environ.get("WANDB_ENTITY")
    run = wandb.init(project=project, entity=entity)
    wandb.log({"smoke/ok": 1})
    run.finish()


def main():
    tmpdir = tempfile.gettempdir()
    print(f"TMPDIR={os.environ.get('TMPDIR')}")
    print(f"tempfile.gettempdir()={tmpdir}")

    socket_path = _check_unix_socket(tmpdir)
    print(f"unix_socket_ok={socket_path}")

    _run_wandb()
    print("wandb_ok=True")


if __name__ == "__main__":
    main()
