import os
import json
import re
import inspect
import textwrap
import requests
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.graph import START, END, StateGraph, add_messages
from tools import read_files, read_slice_of_files, read_latest_logs, get_request, run_command, send_mail

load_dotenv()

MAX_LLM_CALLS = 20  # Only count LLM calls, not tool executions


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_HOST = os.getenv("LLM_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

tools = {
    "read_files": read_files,
    "read_slice_of_files": read_slice_of_files,
    "read_latest_logs": read_latest_logs,
    "get_request": get_request,
    "run_command": run_command,
    "send_mail": send_mail,
}

SYSTEM_PROMPT = textwrap.dedent("""\
You are a professional server monitoring AI assistant. Provide concise, technical status updates.

## Behavior Rules
1. ALWAYS gather real data using tools before making any analysis or decision — NEVER assume or invent values.
2. You MUST call at least one tool before ever giving a final_response. A final_response with no prior tool calls is FORBIDDEN.
3. Take exactly one step at a time: call one tool, wait for the result, then decide the next step.
4. Use as many tools as needed to build a complete picture before concluding — do NOT stop early.
5. Be concise and technical — no unnecessary explanations.
6. If uncertain, gather more data rather than guessing.
7. If critical issues are detected, use send_mail to alert the admin with a full report.
8. For normal health check you can report to the admin as well, so that they can be assured that the system is being monitored.
9. For any kind of operation performed please write a detailed email by using the send_mail only write the body text and send all the steps you have performed.
10. If the server is unhealthy, prioritize sending an email alert before giving the final_response.
11. Send mail before giving final_response, so that the admin can be assured that the system is being monitored and they will receive a detailed email report of all the steps performed by the agent.
## Available Tools

### File Tools
- `read_files(file_path)` — reads full content of a file as a list of lines.
- `read_slice_of_files(file_path, lines_to_read=10, start_line=0)` — reads a specific slice of lines.
- `read_latest_logs(file_path, max_lines)` — reads the most recent N lines from a file.

### API Tools
- `get_request(url, params=None)` — makes a GET request. Use for health endpoints and metrics.

### Command Tools
- `run_command(command)` — executes a shell command on the server.

### Notifying Tools
- `send_mail(body)` — sends an email notification to alert admins.

## Admin Contacts
- Siddhant Kadam (siddhantkadam.dev@gmail.com)
- siddhantkadam.personal@gmail.com

## Server Registry

### server1
- Host: localhost | Port: 3000
- Log file: C:/WorkFolder/LoadBalancer/servers/server1/server1.log
- Service: sudo systemctl (start | restart | reload | status) server1_lb.service
- Metrics: http://localhost:3000/metrics
- Health: http://localhost:3000/healthdata

## Command Execution Note
All systemctl commands must be prefixed with `wsl -u root`:
  wsl -u root systemctl status server1_lb.service
  wsl -u root systemctl restart server1_lb.service
Never call systemctl directly without wsl -u root — it will fail.

## Output Format — STRICT
Your entire response must be a single valid JSON object. No markdown, no code fences, no explanation before or after. ONLY the JSON object.

{
  "step": "<call_tool | final_response>",
  "tool_name": "<tool name, or null for final_response>",
  "tool_input": "<tool arguments as a dict, or final answer string>"
}

### Step Definitions
- `call_tool` — invoke a tool. Set tool_name and tool_input as a dict of named arguments.
- `final_response` — ONLY allowed after you have received real tool results. Set tool_name to null. Write the complete technical summary in tool_input based solely on observed data.

## Correct Workflow — Follow This Exactly
Step 1: Call get_request to fetch health data.
Step 2: Call get_request to fetch metrics.
Step 3: Call read_latest_logs to check recent log entries.
Step 4: Only after receiving all results, emit final_response with real observed values.

## Example Flow (populate all values from real tool results — never copy these placeholders)
{"step": "call_tool", "tool_name": "get_request", "tool_input": {"url": "http://localhost:3000/healthdata"}}
[wait for tool result]
{"step": "call_tool", "tool_name": "get_request", "tool_input": {"url": "http://localhost:3000/metrics"}}
[wait for tool result]
{"step": "call_tool", "tool_name": "read_latest_logs", "tool_input": {"file_path": "C:/WorkFolder/LoadBalancer/servers/server1/server1.log", "max_lines": 20}}
[wait for tool result]
{"step": "final_response", "tool_name": null, "tool_input": "server1 status: <STATUS>. Uptime: Xd Xh. CPU: X%. Memory: X%. Recent logs: X errors / No anomalies."}

IMPORTANT: The example above shows structure only. You MUST populate all values from real tool results.
""")


# ---------------------------------------------------------------------------
# LLM Callers
# ---------------------------------------------------------------------------

def call_ollama(messages: list) -> str:
    response = requests.post(
        url=f"{LLM_HOST}/api/chat",
        json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "No content from LLM")


def call_groq(messages: list) -> str:
    response = requests.post(
        url="https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": GROQ_MODEL, "messages": messages, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    msg = data["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning") or "No content from LLM"


def call_llm(formatted_messages: list) -> str:
    try:
        if LLM_PROVIDER == "groq":
            return call_groq(formatted_messages)
        return call_ollama(formatted_messages)
    except Exception as e:
        return (
            f'{{"step": "final_response", "tool_name": null, '
            f'"tool_input": "LLM connection error ({LLM_PROVIDER}): {str(e)}"}}'
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models like Qwen."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> dict | None:
    text = _strip_thinking(text)
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
    return None


def _tool_results_exist(state: AgentState) -> bool:
    """Return True if at least one tool result message exists in history."""
    return any(
        hasattr(m, "content") and m.content.startswith("[Tool result")
        for m in state["messages"]
    )


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

def init_llm(state: AgentState) -> dict:
    """Call the LLM with the full conversation history."""
    step_count = state.get("step_count", 0) + 1

    formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    for msg in state["messages"]:
        formatted_messages.append({
            "role": role_map.get(msg.type, msg.type),
            "content": msg.content,
        })

    llm_response = call_llm(formatted_messages)
    print(f"[LLM step {step_count}] {llm_response[:300]}{'...' if len(llm_response) > 300 else ''}")

    return {
        "messages": [{"role": "assistant", "content": llm_response}],
        "step_count": step_count,
    }


def execute_tool(state: AgentState) -> dict:
    """Parse the last LLM message and execute the requested tool."""
    last_message = state["messages"][-1]
    print(f"[Tool] Parsing: {last_message.content[:300]}")

    try:
        parsed = _extract_json(last_message.content)
        if not parsed:
            raise ValueError("No valid JSON found in LLM response")

        tool_name = parsed.get("tool_name")
        tool_input = parsed.get("tool_input")

        if tool_name not in tools:
            raise ValueError(
                f"Unknown tool: '{tool_name}'. Available tools: {list(tools.keys())}"
            )

        # Normalize tool_input to a dict of named kwargs
        if isinstance(tool_input, dict):
            tool_kwargs = tool_input
        elif isinstance(tool_input, str):
            sig = inspect.signature(tools[tool_name])
            first_param = next(iter(sig.parameters))
            tool_kwargs = {first_param: tool_input}
        else:
            raise ValueError(
                f"tool_input must be a dict or string, got {type(tool_input).__name__}. "
                f'Use a dict of named args e.g. {{"url": "http://..."}}'
            )

        print(f"[Tool] Calling {tool_name}({tool_kwargs})")
        tool_response = tools[tool_name](**tool_kwargs)
        result_text = f"[Tool result from {tool_name}]: {tool_response}"

    except Exception as e:
        result_text = (
            f"[Tool error]: {str(e)}. "
            "Reassess and try again — pass tool_input as a dict with named parameters. "
            "Do NOT give a final_response yet."
        )
        print(f"[Tool] Error: {e}")

    return {"messages": [{"role": "user", "content": result_text}]}


def retry_node(state: AgentState) -> dict:
    """Nudge the LLM back on track when it returns invalid JSON."""
    step_count = state.get("step_count", 0) + 1
    print(f"[Retry] Invalid JSON from LLM. Nudging. (step {step_count})")
    return {
        "messages": [{"role": "user", "content": (
            "Your last response was not valid JSON. "
            "You MUST reply with ONLY a single JSON object — no markdown, no code fences, no explanation. "
            "Tool call example: "
            '{"step": "call_tool", "tool_name": "get_request", "tool_input": {"url": "http://localhost:3000/healthdata"}} '
            "Final answer example: "
            '{"step": "final_response", "tool_name": null, "tool_input": "Your summary here."}'
        )}],
        "step_count": step_count,
    }


def force_tool_node(state: AgentState) -> dict:
    """Intercept a premature final_response and force the LLM to call a tool first."""
    step_count = state.get("step_count", 0) + 1
    print(f"[Force] Premature final_response intercepted. Forcing tool call. (step {step_count})")
    return {
        "messages": [{"role": "user", "content": (
            "You gave a final_response without calling any tools first. This is NOT allowed. "
            "You MUST gather real data before concluding. "
            "Start now — call get_request on the health endpoint: "
            '{"step": "call_tool", "tool_name": "get_request", "tool_input": {"url": "http://localhost:3000/healthdata"}}'
        )}],
        "step_count": step_count,
    }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def route_after_llm(state: AgentState) -> str:
    """Decide the next node based on the LLM's latest response."""
    if state.get("step_count", 0) >= MAX_LLM_CALLS:
        print(f"[Router] Max LLM calls ({MAX_LLM_CALLS}) reached. Ending.")
        return END

    last_message = state["messages"][-1]
    parsed = _extract_json(last_message.content)

    if parsed is None:
        print("[Router] Could not parse LLM JSON → retry_node")
        return "retry_node"

    step = parsed.get("step", "").strip()
    print(f"[Router] Parsed step='{step}'")

    if step == "call_tool":
        return "execute_tool"

    if step == "final_response":
        # Block final_response if no tools have been called yet
        if not _tool_results_exist(state):
            print("[Router] final_response before any tool calls → force_tool_node")
            return "force_tool_node"
        return END

    # Unexpected step value — treat as invalid JSON
    print(f"[Router] Unknown step value '{step}' → retry_node")
    return "retry_node"


# ---------------------------------------------------------------------------
# Graph Assembly
# ---------------------------------------------------------------------------

builder = StateGraph(AgentState)
builder.add_node("init_llm", init_llm)
builder.add_node("execute_tool", execute_tool)
builder.add_node("retry_node", retry_node)
builder.add_node("force_tool_node", force_tool_node)

builder.add_edge(START, "init_llm")
builder.add_conditional_edges("init_llm", route_after_llm, {
    "execute_tool": "execute_tool",
    "retry_node": "retry_node",
    "force_tool_node": "force_tool_node",
    END: END,
})
builder.add_edge("execute_tool", "init_llm")
builder.add_edge("retry_node", "init_llm")
builder.add_edge("force_tool_node", "init_llm")

agent = builder.compile()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    initial_state = {
        "messages": [{"role": "user", "content": (
            "Run a full diagnostic on server1. "
            "You MUST use tools to collect real data — check health, metrics, and recent logs. "
            "Do NOT give a final_response until you have called at least three tools and received their results."
        )}],
        "step_count": 0,
    }
    result = agent.invoke(initial_state, {"recursion_limit": 100})

    print("\n========== AGENT TRACE ==========")
    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.content:
            print(msg.content)
            print("---")