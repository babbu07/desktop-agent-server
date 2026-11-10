
"""Tailscale installation and connectivity management for Windows."""

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


def verify_installer(path: Path) -> None:
    """Verify the installer's Windows Authenticode signature."""

    escaped = str(path).replace("'", "''")

    script = (
        f"$sig = Get-AuthenticodeSignature -LiteralPath '{escaped}'; "
        'Write-Output "$($sig.Status)|$($sig.SignerCertificate.Subject)"'
    )

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
    )

    status, _, signer = result.stdout.strip().partition("|")

    if (
        result.returncode != 0
        or status != "Valid"
        or "Tailscale" not in signer
    ):
        raise TailscaleError(
            "Tailscale installer signature verification failed."
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
    """Get the connected device's Tailscale IPv4 address."""

    if os.name != "nt":
        raise TailscaleError(
            "This server currently supports Windows only."
        )

    cli = find_tailscale_cli()

    if not cli:
        raise TailscaleError("Tailscale is not installed.")

    result = subprocess.run(
        [cli, "ip", "-4"],
        capture_output=True,
        text=True,
        timeout=15,
    )

    address = (
        result.stdout.strip().splitlines()[0]
        if result.stdout.strip()
        else ""
    )

    if result.returncode != 0 or not address:
        raise TailscaleError(
            "Tailscale is installed but not connected. "
            "Please sign in and connect to your tailnet."
        )

    try:
        parsed = ipaddress.ip_address(address)
    except ValueError as exc:
        raise TailscaleError(
            "Invalid Tailscale IP address."
        ) from exc

    if parsed.version != 4 or parsed not in TAILSCALE_NETWORK:
        raise TailscaleError(
            "The returned address is not a valid Tailscale IPv4 address."
        )

    return str(parsed)
