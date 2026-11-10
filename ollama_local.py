"""Everything that talks to Ollama. No web-server code lives here."""
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Iterator

DEFAULT_URL = "http://127.0.0.1:11434"


class OllamaError(Exception):
    """kind: unavailable | bad_request | model_not_found | timeout | upstream"""

    def __init__(self, message: str, kind: str = "upstream"):
        super().__init__(message)
        self.message = message
        self.kind = kind


class OllamaHandler:
    FALLBACK_EXE = os.path.join(
        os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"
    )

    def __init__(self, base_url: str = DEFAULT_URL, auto_install: bool = True):
        self.base_url = base_url.rstrip("/")
        self.available_model: list[str] = []
        self._pull_lock = threading.Lock()
        self._serve_lock = threading.Lock()

        self.ollama_exe = self._find_exe()
        if self.ollama_exe is None and auto_install:
            self._install()
            self.ollama_exe = self._find_exe()

        if self.ollama_exe:
            try:
                self.ensure_server()
                self.refresh_models()
            except OllamaError as e:
                print(f"Ollama startup warning: {e.message}")

    # ------------------------------------------------------------------
    # Install / locate
    # ------------------------------------------------------------------
    def _find_exe(self) -> str | None:
        for exe in (shutil.which("ollama"), self.FALLBACK_EXE):
            if exe and os.path.exists(exe):
                try:
                    subprocess.run(
                        [exe, "--version"], check=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    )
                    return exe
                except (subprocess.CalledProcessError, OSError):
                    continue
        return None

    def _install(self) -> None:
        installer_url = "https://ollama.com/download/OllamaSetup.exe"
        installer_path = "OllamaSetup.exe"
        try:
            print("Ollama not found. Downloading installer...")
            urllib.request.urlretrieve(installer_url, installer_path)
            print("Installing Ollama...")
            subprocess.run(
                [installer_path, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"],
                check=True,
            )
            time.sleep(2)
        except urllib.error.URLError as e:
            print(f"Download failed (network/URL issue): {e}")
        except subprocess.CalledProcessError as e:
            print(f"Installer exited with an error (code {e.returncode}): {e}")
        except PermissionError as e:
            print(f"Permission denied (try running as administrator): {e}")
        except OSError as e:
            print(f"OS error while running the installer: {e}")
        except Exception as e:
            print(f"Unexpected error during Ollama installation: {e}")
        finally:
            if os.path.exists(installer_path):
                try:
                    os.remove(installer_path)
                except OSError as e:
                    print(f"Could not delete installer file: {e}")

    # ------------------------------------------------------------------
    # Background server (`ollama serve`) that the HTTP API needs
    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/version", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def version(self) -> str | None:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/version", timeout=2) as r:
                return json.loads(r.read().decode("utf-8")).get("version")
        except Exception:
            return None

    def ensure_server(self) -> None:
        if self.is_running():
            return
        if not self.ollama_exe:
            raise OllamaError("Ollama is not installed or could not be found.", "unavailable")
        with self._serve_lock:
            if self.is_running():
                return
            try:
                subprocess.Popen(
                    [self.ollama_exe, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as e:
                raise OllamaError(f"Could not start Ollama: {e}", "unavailable")
            for _ in range(30):
                time.sleep(1)
                if self.is_running():
                    return
            raise OllamaError("Ollama server did not start within 30 seconds.", "unavailable")

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _from_http_error(e: urllib.error.HTTPError) -> OllamaError:
        body = e.read().decode("utf-8", errors="replace")
        try:
            msg = json.loads(body).get("error", body)
        except Exception:
            msg = body
        if e.code == 404:
            kind = "model_not_found"
        elif e.code == 400:
            kind = "bad_request"
        else:
            kind = "upstream"
        return OllamaError(msg or f"Ollama returned HTTP {e.code}", kind)

    def _open(self, path: str, payload: dict | None = None, timeout: int = 300):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            raise self._from_http_error(e)
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise OllamaError(f"Ollama did not respond within {timeout} seconds.", "timeout")
            raise OllamaError(f"Could not reach Ollama: {e.reason}", "unavailable")
        except TimeoutError:
            raise OllamaError(f"Ollama did not respond within {timeout} seconds.", "timeout")

    def _post_json(self, path: str, payload: dict, timeout: int) -> dict:
        try:
            with self._open(path, payload, timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except TimeoutError:
            raise OllamaError(f"Model did not respond within {timeout} seconds.", "timeout")
        except json.JSONDecodeError:
            raise OllamaError("Ollama returned an invalid response.", "upstream")

    @staticmethod
    def _iter_ndjson(resp, timeout: int) -> Iterator[dict]:
        try:
            for raw in resp:
                line = raw.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in chunk:
                    raise OllamaError(str(chunk["error"]), "upstream")
                yield chunk
        except TimeoutError:
            raise OllamaError(f"Model did not respond within {timeout} seconds.", "timeout")
        finally:
            resp.close()  # also runs if the client disconnects, which stops generation

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    def list_models(self) -> list[dict]:
        self.ensure_server()
        try:
            with self._open("/api/tags", timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8")).get("models", [])
        except TimeoutError:
            raise OllamaError("Ollama did not respond.", "timeout")

    def refresh_models(self) -> list[str]:
        self.available_model = [m["name"] for m in self.list_models()]
        return self.available_model

    def has_model(self, model: str) -> bool:
        names = self.refresh_models()
        return model in names or f"{model}:latest" in names

    def pull_model(self, model: str) -> None:
        if not self.ollama_exe:
            raise OllamaError("Ollama is not installed or could not be found.", "unavailable")
        print(f"Pulling model '{model}'... this can take a while.")
        try:
            subprocess.run([self.ollama_exe, "pull", model], check=True)
        except subprocess.CalledProcessError as e:
            raise OllamaError(
                f"Could not pull model '{model}' (exit code {e.returncode}). "
                "Check the model name and the internet connection.",
                "model_not_found",
            )
        except OSError as e:
            raise OllamaError(f"Could not run ollama: {e}", "unavailable")
        self.refresh_models()

    def ensure_model(self, model: str) -> None:
        """Pull the model if missing. The lock makes concurrent requests wait, not double-download."""
        with self._pull_lock:
            if not self.has_model(model):
                self.pull_model(model)

    def _prepare(self, model: str, auto_pull: bool) -> None:
        self.ensure_server()
        if auto_pull:
            self.ensure_model(model)

    # ------------------------------------------------------------------
    # Chat / embeddings
    # ------------------------------------------------------------------
    @staticmethod
    def _chat_payload(model, messages, tools, response_format, options, stream) -> dict:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if tools:
            payload["tools"] = tools
        if response_format is not None:
            payload["format"] = response_format  # "json" or a JSON schema dict
        if options:
            payload["options"] = options
        return payload

    def chat(self, model: str, messages: list[dict], *, tools=None, response_format=None,
             options=None, timeout: int = 300, auto_pull: bool = True) -> dict:
        """Returns Ollama's raw /api/chat response dict."""
        self._prepare(model, auto_pull)
        payload = self._chat_payload(model, messages, tools, response_format, options, False)
        return self._post_json("/api/chat", payload, timeout)

    def chat_stream(self, model: str, messages: list[dict], *, tools=None, response_format=None,
                    options=None, timeout: int = 300, auto_pull: bool = True) -> Iterator[dict]:
        """Connects immediately (so errors raise here), then returns an iterator of raw chunks."""
        self._prepare(model, auto_pull)
        payload = self._chat_payload(model, messages, tools, response_format, options, True)
        resp = self._open("/api/chat", payload, timeout)
        return self._iter_ndjson(resp, timeout)

    def embed(self, model: str, inputs: list[str], *, timeout: int = 120,
              auto_pull: bool = True) -> dict:
        self._prepare(model, auto_pull)
        return self._post_json("/api/embed", {"model": model, "input": inputs}, timeout)