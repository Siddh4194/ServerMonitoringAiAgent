import os
import json
import re
import requests
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.graph import START, END, StateGraph, add_messages
from tools import read_files, read_slice_of_files, read_latest_logs, get_request, run_command, send_mail
load_dotenv()

MAX_STEPS = 20

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_HOST = os.getenv("LLM_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

tools = {
    "read_files": read_files,
    "read_slice_of_files": read_slice_of_files,
    "read_latest_logs": read_latest_logs,
    "get_request": get_request,
    "run_command": run_command,
    "send_mail":send_mail
}


def call_ollama(messages: list) -> str:
    response = requests.post(
        url=f"{LLM_HOST}/api/chat",
        json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
        timeout=40,
    )
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
        timeout=40,
    )
    data = response.json()
    msg = data["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning") or "No content from LLM"


def call_llm(formatted_messages: list):
    try:
        if LLM_PROVIDER == "groq":
            return call_groq(formatted_messages)
        return call_ollama(formatted_messages)
    except Exception as e:
        return f"Error connecting to LLM ({LLM_PROVIDER}): {str(e)}"

def init_llm(state: AgentState):
    step_count = state.get("step_count", 0) + 1

    system_prompt = {
        "role": "system",
        "content": """
You are a professional server monitoring AI assistant. You provide concise, technical status updates and make careful, data-driven decisions.

## Behavior rules
1. Always gather data before making any analysis or decision — never assume.
2. Take exactly one step at a time: call one tool, wait for the result, analyze it, then decide the next step.
3. Use as many tools as needed to build a complete picture before concluding.
4. Be concise and technical — no unnecessary explanations.
5. If uncertain, gather more data rather than guessing.
6. If anything is not working you can send mail with full report to the admin using send_mail tool.

## Available tools

### File tools
- `read_files(file_path)` — reads the full content of a file as a list of lines. Use for small config files or short logs.
- `read_slice_of_files(file_path, lines_to_read=10, start_line=0)` — reads a specific slice of lines from a file. Use when you need a specific section of a large file.
- `read_latest_logs(file_path, max_lines)` — reads the most recent N lines from a file. Prefer this for large log files.

### API tools
- `get_request(url, params=None)` — makes a GET request to a URL with optional query parameters. Use for health endpoints and Prometheus metrics.

### Command tools
- `run_command(command)` — executes a shell command on the server and returns the output. Use for service status checks, restarts, or system diagnostics.

## Notifying tools
- `send_mail(subject, body, to)` — sends an email notification. Use this to alert admins if you detect critical issues.

# admin details
siddhant kadam
siddhantkadam.dev@gmail.com
siddhantkadam.personal@gmail.com

## Server registry

### server1
- Host: localhost
- Port: 3000
- Log file: C:/WorkFolder/LoadBalancer/servers/server1/server1.log
- Service: `sudo systemctl (start | restart | reload | status) server1_lb.service`
- Metrics: `http://localhost:3000/metrics`
- Health: `http://localhost:3000/healthdata`

## Prometheus endpoint patterns
- Metrics: `http://{host}:{port}/metrics`
- Health data: `http://{host}:{port}/healthdata`

## Output format
Every response must be a single valid JSON object — no extra text, no markdown, no explanation outside the JSON.

{
  "step": "<call_tool | analyze | final_response>",
  "tool_name": "<tool name, or null if step is analyze or final_response>",
  "tool_input": "<tool arguments, or analysis text, or final answer>"
}

### Step definitions
- `call_tool` — you want to invoke a tool. Set `tool_name` and `tool_input` accordingly.
- `analyze` — you are reasoning about data already collected. Set `tool_name` to null and write your analysis in `tool_input`. Follow this with either another `call_tool` or a `final_response`.
- `final_response` — you have enough data to answer. Set `tool_name` to null and write your complete technical summary in `tool_input`.

## Example flow
{"step": "call_tool", "tool_name": "get_request", "tool_input": "http://localhost:3000/healthdata"}
{"step": "analyze", "tool_name": null, "tool_input": "Health endpoint returned 200. Uptime is 3d 4h. No reported issues. Proceeding to check metrics."}
{"step": "call_tool", "tool_name": "get_request", "tool_input": "http://localhost:3000/metrics"}
{"step": "final_response", "tool_name": null, "tool_input": "server1 is healthy. Uptime: 3d 4h. CPU: 34%. Memory: 61%. No anomalies detected in logs or metrics."}

## Command execution note
All systemctl commands must be prefixed with `wsl -u root` since services run inside WSL.
Always use this pattern:
  wsl -u root systemctl status server1_lb.service
  wsl -u root systemctl restart server1_lb.service
Never call `systemctl` directly without the `wsl -u root` prefix — it will fail silently.
        """
    }

    ollama_messages = [system_prompt]

    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    for msg in state["messages"]:
        ollama_messages.append({
            "role": role_map.get(msg.type, msg.type),
            "content": msg.content
        })

    llmresponse = call_llm(ollama_messages)

    return {"messages": [{"role": "assistant", "content": llmresponse}], "step_count": step_count}

def execute_tool(state: AgentState):
    step_count = state.get("step_count", 0) + 1
    try:
        last_message = state["messages"][-1]
        print("Executing tool with input: ", last_message.content)
        call_tool_response = _extract_json(last_message.content)
        if not call_tool_response:
            raise ValueError("No JSON found in LLM response")

        tool_name = call_tool_response["tool_name"]
        tool_input = call_tool_response["tool_input"]

        if isinstance(tool_input, dict):
            tool_response = tools[tool_name](**tool_input)
        else:
            tool_response = tools[tool_name](tool_input)

        result_text = f"[Tool result from {tool_name}]: {tool_response}"
        return {"messages": [{"role": "user", "content": result_text}], "step_count": step_count}
    except Exception as e:
        error_text = f"Tool execution failed: {str(e)}. Please provide a final response with this error."
        return {"messages": [{"role": "user", "content": error_text}], "step_count": step_count}
    
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
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    break
    return None

def route_next_step(state: AgentState):
    if state.get("step_count", 0) >= MAX_STEPS:
        return END

    last_message = state["messages"][-1]
    parsed = _extract_json(last_message.content)
    print("Parsed LLM response for routing: ", last_message)
    if parsed is None:
        print("[WARN] Could not parse LLM response, injecting correction...")
        state["messages"].append({
            "role": "user",
            "content": (
                "Your last response was not valid JSON. "
                "You must reply with ONLY a single JSON object. "
                "No <think> tags, no markdown, no explanation. "
                'Example: {"step": "call_tool", "tool_name": "run_command", "tool_input": {"command": "wsl -u root systemctl status server1_lb.service"}}'
            )
        })
        return "init_llm"

    step = parsed.get("step", "").strip()
    if step == "call_tool":
        return "execute_tool"
    if step == "analyze":
        return "init_llm"
    return END

builder = StateGraph(AgentState)
builder.add_node("init_llm", init_llm)
builder.add_node("execute_tool", execute_tool)

builder.add_edge(START, "init_llm")
builder.add_conditional_edges("init_llm", route_next_step, {
    "execute_tool": "execute_tool",
    "init_llm": "init_llm",
    END: END
})
builder.add_edge("execute_tool", "init_llm")

agent = builder.compile()

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "I'm not able to see the server status please check and let me know over the email the detailed report of this?"}]},
        {"recursion_limit": 50}
    )
    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.content:
            print(msg.content)
            print("---")
