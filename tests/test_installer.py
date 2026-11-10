
from tailscale_manager import verify_installer

installer = r"C:\Users\Manpreet Singh\Downloads\tailscale-setup-1.104.5.exe"

verify_installer(installer)

print("SUCCESS: Tailscale installer signature verified.")
