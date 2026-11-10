# Desktop Agent Server

OpenAI-compatible API server for local Ollama models, packaged as a single
Windows .exe. Run it on any Windows machine and it becomes an LLM server
for your agents.

## Features
- OpenAI-compatible: /v1/chat/completions, /v1/models, /v1/embeddings
- Streaming, tool calling, structured output (JSON schema)
- Installs Ollama and downloads models automatically
- API key authentication (auto-generated on first run)

## Quick start (exe)
1. Download DesktopAgentServer.exe from the Releases page
2. Run it and allow the Windows Firewall prompt
3. Copy the URL and API key shown in the console

## Run from source
    pip install -r requirements.txt
    python main.py

## Options
    --host, --port, --api-key, --no-auth, --no-auto-pull, --timeout

## Usage
(Paste the OpenAI SDK and LangChain examples here)

## Build the exe
    build.bat

## Security
- The server is protected by an API key. Do not use --no-auth on untrusted networks.
- Do not expose the port to the internet directly. Use a VPN or tunnel
  such as Tailscale or Cloudflare Tunnel.
- Some antivirus tools flag PyInstaller executables as false positives.
  You can build it yourself from source.

## License
MIT