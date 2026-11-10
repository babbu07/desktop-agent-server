# examples/desktop_agent_llm.py
"""
LangChain client for the Desktop Agent Server.

Install:
    pip install langchain-openai langgraph

Configure (Command Prompt):
    set DESKTOP_AGENT_HOST=192.168.1.25
    set DESKTOP_AGENT_API_KEY=sk-local-...

Configure (PowerShell):
    $env:DESKTOP_AGENT_HOST = "192.168.1.25"
    $env:DESKTOP_AGENT_API_KEY = "sk-local-..."

Run the demo:
    python examples/desktop_agent_llm.py
"""
import os

from langchain_openai import ChatOpenAI


class DesktopAgentChat(ChatOpenAI):
    """LangChain chat model that talks to a Desktop Agent Server.

    Inherits streaming, async, bind_tools() and with_structured_output()
    from ChatOpenAI, because the server speaks the OpenAI protocol.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        api_key: str = "no-key",
        model: str = "qwen2.5",
        port: int = 8000,
        **kwargs,
    ):
        kwargs.setdefault("timeout", 600)    # a first-time model download can be slow
        kwargs.setdefault("temperature", 0)
        super().__init__(
            base_url=f"http://{host}:{port}/v1",
            api_key=api_key,
            model=model,
            **kwargs,
        )

    @classmethod
    def from_env(cls, model: str = "qwen2.5", **kwargs) -> "DesktopAgentChat":
        """Build the client from DESKTOP_AGENT_HOST / _PORT / _API_KEY."""
        return cls(
            host=os.environ.get("DESKTOP_AGENT_HOST", "127.0.0.1"),
            port=int(os.environ.get("DESKTOP_AGENT_PORT", "8000")),
            api_key=os.environ.get("DESKTOP_AGENT_API_KEY", "no-key"),
            model=model,
            **kwargs,
        )


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
def demo() -> None:
    from langchain_core.tools import tool
    from pydantic import BaseModel

    llm = DesktopAgentChat.from_env()

    # 1. Plain call
    print("\n--- Plain call ---")
    print(llm.invoke("Say hello in one sentence.").content)

    # 2. Streaming
    print("\n--- Streaming ---")
    for chunk in llm.stream("Count from 1 to 5."):
        print(chunk.content, end="", flush=True)
    print()

    # 3. Structured output (the server enforces the JSON schema)
    print("\n--- Structured output ---")

    class City(BaseModel):
        city: str
        population_millions: float

    structured = llm.with_structured_output(City)
    print(structured.invoke("Tell me about Tokyo."))

    # 4. Tool-calling agent (LangGraph)
    print("\n--- Agent with a tool ---")

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(llm, [add])
    result = agent.invoke({"messages": [("user", "What is 17 + 25? Use the tool.")]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    demo()