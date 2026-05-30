import os
import requests
from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph, MessagesState

load_dotenv()

LLM_HOST = os.getenv("LLM_HOST")

# 1. Update this function to accept the structured message list instead of a single prompt string
def call_llm(formatted_messages: list):
    try:
        response = requests.post(
            url=f"{LLM_HOST}/api/chat",
            json={
                "model": "llama3.2:3b",
                "messages": formatted_messages, # Sends system instructions + full history
                "stream": False                  # Stops streaming so we get one JSON response
            },
            timeout=15
        )
        
        # Parse the standard Ollama chat format response
        data = response.json()
        print(data)
        return data.get("message", {}).get("content", "No content from LLM")
    except Exception as e:
        return f"Error connecting to LLM: {str(e)}"

def mock_llm(state: MessagesState):
    # 2. Define your system prompt here
    system_prompt = {
        "role": "system",
        "content": """You are a professional server monitoring AI assistant. Provide concise, technical status updates.
        You make every decision carefully by utilizing all the tools as much as you can
        And you take one step at a time, you call one tool, analyze the data, and then decide the next step.
        
        
        So basically you need to take the decisions based on the data you collect by calling the tools 
        available tools are 
        read_files(file_path) - reads the content of a file and returns it as a list of lines. Use this to read server logs or config files.
        read_slice_of_files(file_path, lines_to_read=10, start_line=0) - reads a specific slice of lines from a file. Use this to get the most recent entries from large log files without reading the entire file.
        read_latest_logs(file_path, max_lines) - reads the latest lines from a file. Use this to get the most recent entries from large log files.
        
        # api request tools
        get_request(url, params=None) - makes a GET request to the specified URL with optional query parameters. Use this to query server status endpoints or monitoring APIs.
        
        #command execution tools
        run_command(command) - executes a shell command on the server and returns the output. Use this to check service status, restart services, or gather system information.
        
        server informations are stored in the following files:
        servers\server1\server1.log - contains the latest logs from server1
        server1 :
        port : 3000
        host : localhost
        logs : /mnt/c/WorkFolder/LoadBalancer/servers/server1/server1.log - contains the latest logs from server1 (alternative path)
        service : sudo systemctl (start || restart || reload || status) server1_lb.service
        
        promithius endpoints
        http://host:port/metrics - contains the latest metrics from the server, including CPU usage, memory usage, and network traffic.
        http://host:port/healthData - contains the health status of the server, including uptime and any reported issues.
        
        
        output format 
        {"step":"call_tool || analyze || final_response","tool_name":"tool_name_here","tool_input":"tool_input_here"}
        
        Rules:
        1. One step at a time, call one tool, analyze the data, and then decide the next step.
        2. Always use the tools to gather data before making any analysis or final response.
        3. Be concise and technical in your responses, avoid unnecessary explanations.
        4. If you have gathered enough data to provide a final response, set the step to "final_response" and provide your answer in the tool_input field. Otherwise, set the step to "call_tool" and specify the next tool you want to call along with its input.
        5. Always follow the output format strictly to ensure proper communication with the system.
        """
    }
    
    # 3. Format LangGraph messages to fit Ollama's expected structure
    ollama_messages = [system_prompt]
    
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    for msg in state["messages"]:
        ollama_messages.append({
            "role": role_map.get(msg.type, msg.type),
            "content": msg.content
        })
    
    # Send the combined array to the server
    llmresponse = call_llm(ollama_messages)
    
    return {"messages": [{"role": "assistant", "content": llmresponse}]}

builder = StateGraph(MessagesState)
builder.add_node("mock_llm", mock_llm)
builder.add_edge(START, "mock_llm")
builder.add_edge("mock_llm", END)

agent = builder.compile()

if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "What is the status of the server?"}]})
    print("the response ",result["messages"][-1].content)
