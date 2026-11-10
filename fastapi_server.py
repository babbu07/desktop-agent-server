"""HTTP server only. Exposes an OpenAI-compatible API on top of OllamaHandler."""
import json
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from ollama_local import OllamaError, OllamaHandler

KIND_TO_HTTP = {
    "unavailable": (503, "server_error"),
    "bad_request": (400, "invalid_request_error"),
    "model_not_found": (404, "invalid_request_error"),
    "timeout": (504, "timeout_error"),
    "upstream": (502, "api_error"),
}
STATUS_TO_TYPE = {400: "invalid_request_error", 401: "authentication_error",
                  404: "invalid_request_error", 405: "invalid_request_error"}


# ---------------------------------------------------------------------
# Request schemas (OpenAI field names; unknown fields are ignored)
# ---------------------------------------------------------------------
class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]] = Field(..., min_length=1)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    n: int | None = None


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]


# ---------------------------------------------------------------------
# OpenAI <-> Ollama translation helpers
# ---------------------------------------------------------------------
def _bad(message: str):
    raise HTTPException(status_code=400, detail=message)


def _convert_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    call_names: dict[str, str] = {}  # tool_call_id -> function name

    for m in messages:
        role = m.get("role")
        if role == "developer":
            role = "system"
        if role not in ("system", "user", "assistant", "tool"):
            _bad(f"Unsupported message role: {role!r}")

        content = m.get("content")
        images: list[str] = []
        if isinstance(content, list):
            texts = []
            for part in content:
                ptype = part.get("type")
                if ptype in ("text", "input_text"):
                    texts.append(part.get("text", ""))
                elif ptype == "image_url":
                    url = part.get("image_url")
                    url = url.get("url", "") if isinstance(url, dict) else (url or "")
                    if not url.startswith("data:"):
                        _bad("Only base64 data: URLs are supported for images.")
                    images.append(url.split(",", 1)[1])
                else:
                    _bad(f"Unsupported content part type: {ptype!r}")
            content = "\n".join(texts)

        msg: dict[str, Any] = {"role": role, "content": content or ""}
        if images:
            msg["images"] = images

        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args or "{}")
                    except json.JSONDecodeError:
                        args = {}
                if tc.get("id"):
                    call_names[tc["id"]] = fn.get("name", "")
                calls.append({"function": {"name": fn.get("name", ""), "arguments": args}})
            msg["tool_calls"] = calls

        if role == "tool":
            name = m.get("name") or call_names.get(m.get("tool_call_id", ""))
            if name:
                msg["tool_name"] = name

        out.append(msg)
    return out


def _build_format(rf: dict | None):
    if not rf:
        return None
    rtype = rf.get("type")
    if rtype == "text":
        return None
    if rtype == "json_object":
        return "json"
    if rtype == "json_schema":
        schema = (rf.get("json_schema") or {}).get("schema")
        if not isinstance(schema, dict):
            _bad("response_format.json_schema.schema is required and must be an object.")
        return schema
    _bad(f"Unsupported response_format type: {rtype!r}")


def _build_options(req: ChatRequest) -> dict:
    o: dict[str, Any] = {}
    if req.temperature is not None:
        o["temperature"] = req.temperature
    if req.top_p is not None:
        o["top_p"] = req.top_p
    if req.seed is not None:
        o["seed"] = req.seed
    if req.presence_penalty is not None:
        o["presence_penalty"] = req.presence_penalty
    if req.frequency_penalty is not None:
        o["frequency_penalty"] = req.frequency_penalty
    max_tokens = req.max_completion_tokens or req.max_tokens
    if max_tokens:
        o["num_predict"] = max_tokens
    if req.stop:
        o["stop"] = [req.stop] if isinstance(req.stop, str) else req.stop
    return o


def _tool_calls_out(calls: list[dict] | None, start: int = 0, with_index: bool = False) -> list[dict]:
    out = []
    for i, c in enumerate(calls or [], start=start):
        fn = c.get("function", {})
        item = {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": fn.get("name", ""), "arguments": json.dumps(fn.get("arguments", {}))},
        }
        if with_index:
            item["index"] = i
        out.append(item)
    return out


def _finish_reason(final: dict, has_tools: bool) -> str:
    if has_tools:
        return "tool_calls"
    return "length" if final.get("done_reason") == "length" else "stop"


def _usage(final: dict) -> dict:
    p, c = final.get("prompt_eval_count", 0) or 0, final.get("eval_count", 0) or 0
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def _completion_response(result: dict, model: str) -> dict:
    msg = result.get("message", {})
    tool_calls = _tool_calls_out(msg.get("tool_calls"))
    content = msg.get("content") or ""
    message: dict[str, Any] = {"role": "assistant", "content": content or (None if tool_calls else "")}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": _finish_reason(result, bool(tool_calls))}],
        "usage": _usage(result),
    }


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _stream(chunks, model: str, include_usage: bool):
    cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())

    def frame(delta: dict, finish: str | None = None) -> dict:
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    yield _sse(frame({"role": "assistant", "content": ""}))
    final: dict = {}
    n_tools = 0
    try:
        for ch in chunks:
            msg = ch.get("message", {})
            if msg.get("content"):
                yield _sse(frame({"content": msg["content"]}))
            if msg.get("tool_calls"):
                calls = _tool_calls_out(msg["tool_calls"], start=n_tools, with_index=True)
                n_tools += len(calls)
                yield _sse(frame({"tool_calls": calls}))
            if ch.get("done"):
                final = ch
    except OllamaError as e:
        _, etype = KIND_TO_HTTP.get(e.kind, (502, "api_error"))
        yield _sse({"error": {"message": e.message, "type": etype, "param": None, "code": e.kind}})
        yield "data: [DONE]\n\n"
        return

    yield _sse(frame({}, _finish_reason(final, n_tools > 0)))
    if include_usage:
        yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": model, "choices": [], "usage": _usage(final)})
    yield "data: [DONE]\n\n"


def _error_response(status: int, message: str, etype: str, code: str | None = None):
    return JSONResponse(status_code=status, content={
        "error": {"message": message, "type": etype, "param": None, "code": code}})


# ---------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------
def create_app(api_key: str | None = None, auto_pull: bool = True,
               request_timeout: int = 600) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ollama = OllamaHandler()
        yield

    app = FastAPI(title="Desktop Agent Server", version="2.0.0", lifespan=lifespan)

    # ---- error handlers (all in OpenAI's error format) ----
    @app.exception_handler(OllamaError)
    async def _ollama_error(_: Request, exc: OllamaError):
        status, etype = KIND_TO_HTTP.get(exc.kind, (502, "api_error"))
        return _error_response(status, exc.message, etype, exc.kind)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        return _error_response(exc.status_code, str(exc.detail),
                               STATUS_TO_TYPE.get(exc.status_code, "api_error"))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return _error_response(400, f"Invalid request: {details}", "invalid_request_error")

    # ---- auth ----
    def require_key(authorization: str | None = Header(None)):
        if not api_key:
            return
        token = (authorization or "").strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        if not secrets.compare_digest(token.encode(), api_key.encode()):
            raise HTTPException(401, "Invalid or missing API key. Send 'Authorization: Bearer <key>'.")

    def get_handler(request: Request) -> OllamaHandler:
        handler: OllamaHandler = request.app.state.ollama
        if not handler.ollama_exe:
            raise OllamaError("Ollama is not installed or could not be found.", "unavailable")
        return handler

    # ---- public ----
    @app.get("/health")
    def health(request: Request):
        h: OllamaHandler = request.app.state.ollama
        return {"status": "ok", "ollama_installed": bool(h.ollama_exe),
                "ollama_running": h.is_running(), "ollama_version": h.version(),
                "auth_required": bool(api_key)}

    # ---- OpenAI-compatible, protected ----
    v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_key)])

    @v1.get("/models")
    def list_models(request: Request):
        handler = get_handler(request)
        now = int(time.time())
        return {"object": "list", "data": [
            {"id": m["name"], "object": "model", "created": now, "owned_by": "ollama"}
            for m in handler.list_models()]}

    @v1.get("/models/{model_id:path}")
    def get_model(model_id: str, request: Request):
        handler = get_handler(request)
        names = [m["name"] for m in handler.list_models()]
        if model_id not in names and f"{model_id}:latest" not in names:
            raise OllamaError(f"The model '{model_id}' does not exist.", "model_not_found")
        return {"id": model_id, "object": "model", "created": int(time.time()), "owned_by": "ollama"}

    @v1.post("/chat/completions")
    def chat_completions(body: ChatRequest, request: Request):
        handler = get_handler(request)
        if body.n not in (None, 1):
            _bad("Only n=1 is supported.")

        messages = _convert_messages(body.messages)
        fmt = _build_format(body.response_format)
        options = _build_options(body)
        if fmt is not None:
            options.setdefault("temperature", 0)  # far more reliable structured output

        kwargs = dict(tools=body.tools, response_format=fmt, options=options,
                      timeout=request_timeout, auto_pull=auto_pull)

        if body.stream:
            chunks = handler.chat_stream(body.model, messages, **kwargs)  # errors raise here
            include_usage = bool((body.stream_options or {}).get("include_usage"))
            return StreamingResponse(
                _stream(chunks, body.model, include_usage),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        return _completion_response(handler.chat(body.model, messages, **kwargs), body.model)

    @v1.post("/embeddings")
    def embeddings(body: EmbeddingRequest, request: Request):
        handler = get_handler(request)
        inputs = [body.input] if isinstance(body.input, str) else body.input
        if not inputs:
            _bad("input must not be empty.")
        result = handler.embed(body.model, inputs, auto_pull=auto_pull)
        tokens = result.get("prompt_eval_count", 0) or 0
        return {"object": "list", "model": body.model,
                "data": [{"object": "embedding", "index": i, "embedding": e}
                         for i, e in enumerate(result.get("embeddings", []))],
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}

    app.include_router(v1)
    return app