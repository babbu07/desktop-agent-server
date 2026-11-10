
"""Tailscale installation and connectivity management for Windows."""

import json
import ipaddress
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path


INSTALLER_URL = (
    "https://pkgs.tailscale.com/stable/tailscale-setup-latest.exe"
)

TAILSCALE_NETWORK = ipaddress.ip_network("100.64.0.0/10")

EXPECTED_SIGNER_SUBJECT = "PUT_THE_VERIFIED_SUBJECT_HERE"


class TailscaleError(RuntimeError):
    pass


def find_tailscale_cli() -> str | None:
    """Find the Tailscale CLI."""

    discovered = shutil.which("tailscale")
    if discovered:
        return discovered

    program_files = os.environ.get(
        "ProgramFiles", r"C:\Program Files"
    )

    candidates = [
        Path(program_files) / "Tailscale" / "tailscale.exe"
    ]

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return None


def verify_installer(path: str | Path) -> None:
    escaped = str(path).replace("'", "''")

    script = (
        f"$sig = Get-AuthenticodeSignature -LiteralPath '{escaped}'; "
        'Write-Output "$($sig.Status)|$($sig.SignerCertificate.Subject)"'
    )

    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TailscaleError(
            "Could not verify the Tailscale installer signature."
        ) from exc

    status, separator, signer = result.stdout.strip().partition("|")

    if (
        result.returncode != 0
        or not separator
        or status != "Valid"
        or signer.strip() != EXPECTED_SIGNER_SUBJECT
    ):
        raise TailscaleError(
            "Tailscale installer signature verification failed "
            "or the publisher did not match."
        )

def install_tailscale() -> None:
    """Download and launch the official Tailscale installer."""

    if os.name != "nt":
        raise TailscaleError(
            "Automatic installation is supported on Windows only."
        )

    install_dir = Path(tempfile.gettempdir()) / "DesktopAgentServer"
    install_dir.mkdir(parents=True, exist_ok=True)

    installer = install_dir / "tailscale-setup.exe"

    print("Downloading Tailscale installer...")

    request = urllib.request.Request(
        INSTALLER_URL,
        headers={"User-Agent": "DesktopAgentServer"},
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if not response.geturl().startswith(
                "https://pkgs.tailscale.com/"
            ):
                raise TailscaleError(
                    "Unexpected installer download location."
                )

            total = 0

            with installer.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    total += len(chunk)

                    if total > 100 * 1024 * 1024:
                        raise TailscaleError(
                            "Installer exceeds maximum allowed size."
                        )

                    output.write(chunk)

        verify_installer(installer)

        print("Installer signature verified.")
        print("Starting Tailscale setup...")

        subprocess.Popen([str(installer)], close_fds=True)

    except Exception:
        installer.unlink(missing_ok=True)
        raise


def get_tailscale_ipv4() -> str:
    if os.name != "nt":
        raise TailscaleError(
            "This server currently supports Windows only."
        )

    cli = find_tailscale_cli()

    if not cli:
        raise TailscaleError("Tailscale is not installed.")

    try:
        result = subprocess.run(
            [cli, "status", "--json"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TailscaleError(
            "Tailscale status check timed out."
        ) from exc
    except OSError as exc:
        raise TailscaleError(
            f"Could not execute Tailscale CLI: {exc}"
        ) from exc

    if result.returncode != 0:
        raise TailscaleError(
            f"Tailscale status command failed: {result.stderr.strip()}"
        )

    try:
        status = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise TailscaleError(
            "Tailscale returned invalid status data."
        ) from exc

    if status.get("BackendState") != "Running":
        raise TailscaleError(
            "Tailscale is not running. Please connect to your tailnet."
        )

    self_node = status.get("Self") or {}

    if not self_node.get("Online", False):
        raise TailscaleError(
            "This device is not online in Tailscale. "
            "Please check your connection."
        )

    tailscale_ips = self_node.get("TailscaleIPs", [])

    for address in tailscale_ips:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue

        if parsed.version == 4 and parsed in TAILSCALE_NETWORK:
            return str(parsed)

    raise TailscaleError(
        "No valid Tailscale IPv4 address was found."
    )