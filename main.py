
"""Desktop Agent Server entry point."""

import argparse
import multiprocessing
import os
import secrets
import sys

import uvicorn

from fastapi_server import create_app
from tailscale_manager import (
    TailscaleError,
    find_tailscale_cli,
    install_tailscale,
    get_tailscale_ipv4,
)


def app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)

    return os.path.dirname(os.path.abspath(__file__))


def resolve_api_key(cli_key: str | None) -> str:
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


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Desktop Agent Server"
    )

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--no-auto-pull", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)

    args = parser.parse_args()

    # Step 1: Check Tailscale installation.
    if not find_tailscale_cli():

        print("Tailscale is not installed.")

        install_tailscale()

        print(
            "\nTailscale setup has been launched."
            "\nComplete installation and sign in."
            "\nThen restart Desktop Agent Server."
        )

        return

    # Step 2: Check Tailscale connectivity.
    try:
        host = get_tailscale_ipv4()

    except TailscaleError as exc:
        print(f"\nTailscale error: {exc}")
        return

    # Step 3: Resolve mandatory API key.
    api_key = resolve_api_key(args.api_key)

    # Step 4: Create FastAPI application.
    app = create_app(
        api_key=api_key,
        auto_pull=not args.no_auto_pull,
        request_timeout=args.timeout,
    )

    # Step 5: Start server only on Tailscale interface.
    print("=" * 55)
    print(" Desktop Agent Server")
    print("=" * 55)

    print(f" Tailscale IP : {host}")
    print(f" Server URL   : http://{host}:{args.port}/v1")
    print(" Authentication: ENABLED")

    print("=" * 55)
    print(" Starting server...\n")

    uvicorn.run(
        app,
        host=host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":

    multiprocessing.freeze_support()

    try:
        main()

    except KeyboardInterrupt:
        print("\nServer stopped.")

    except Exception as exc:
        print(f"\nFatal error: {exc}")

        if getattr(sys, "frozen", False):
            input("Press Enter to close...")

        raise
