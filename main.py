"""Entry point. This is the file PyInstaller builds into the .exe."""
import argparse
import multiprocessing
import os
import secrets
import socket
import sys

import uvicorn

from fastapi_server import create_app


def app_dir() -> str:
    """Folder containing the .exe (or this script when run with python)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def resolve_api_key(cli_key: str | None) -> str:
    """Priority: --api-key, then DESKTOP_AGENT_API_KEY env var, then api_key.txt (auto-created)."""
    if cli_key:
        return cli_key
    env_key = os.environ.get("DESKTOP_AGENT_API_KEY")
    if env_key:
        return env_key
    path = os.path.join(app_dir(), "api_key.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            key = f.read().strip()
        if key:
            return key
    key = "sk-local-" + secrets.token_urlsafe(24)
    with open(path, "w", encoding="utf-8") as f:
        f.write(key)
    return key


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # no traffic is sent; just picks the LAN interface
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Desktop Agent Server (OpenAI-compatible, powered by Ollama)")
    parser.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from other devices")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None, help="Use this key instead of the auto-generated one")
    parser.add_argument("--no-auth", action="store_true", help="Disable API key check (trusted networks only)")
    parser.add_argument("--no-auto-pull", action="store_true", help="Don't download missing models automatically")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds to wait for a model reply")
    args = parser.parse_args()

    api_key = None if args.no_auth else resolve_api_key(args.api_key)
    app = create_app(api_key=api_key, auto_pull=not args.no_auto_pull, request_timeout=args.timeout)

    ip = local_ip()
    print("=" * 60)
    print(" Desktop Agent Server")
    print("=" * 60)
    print(f" This device : http://127.0.0.1:{args.port}/v1")
    if args.host == "0.0.0.0":
        print(f" Other devices: http://{ip}:{args.port}/v1")
    print(f" API key      : {api_key if api_key else 'DISABLED (--no-auth)'}")
    print(f" Docs         : http://127.0.0.1:{args.port}/docs")
    print("=" * 60)
    print(" Starting... (first run may install Ollama / download a model)\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # required for PyInstaller on Windows
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\nFatal error: {exc}")
        if getattr(sys, "frozen", False):
            input("Press Enter to close...")  # keeps the window open so you can read the error
        raise