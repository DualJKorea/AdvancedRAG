# agent_app.py  (의사코드)

class MCPClient:
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def list_tools(self):
        return ws_rpc(self.endpoint, method="mcp/listTools")

    def call_tool(self, name, arguments):
        return ws_rpc(self.endpoint, method="mcp/callTool",
                      params={"name": name, "arguments": arguments})

# LLM 공급자 추상화 (OpenAI/Anthropic 등 교체 가능)
class LLM:
    def __init__(self, provider):
        self.provider = provider  # "openai" | "anthropic" | ...

    def chat(self, messages, tools_spec):
        """
        tools_spec: MCP list_tools()에서 받은 JSON Schema를
        LLM의 '툴 호출 / 함수 호출' 기능에 전달 (포맷은 공급자별 변환)
        반환: {"content": "...", "tool_calls": [{"name":"...", "arguments":{...}}, ...]}
        """
        return provider_specific_chat(messages, tools_spec, self.provider)

# --- 에이전트 루프 ---
def agent_run(user_input):
    mcp = MCPClient("ws://localhost:9999")
    tools_spec = mcp.list_tools()                  # ← 동적 '발견'
    llm = LLM(provider="openai")                   # 공급자 교체 자유

    messages = [
        {"role":"system", "content":"You are a helpful assistant. Use tools if needed."},
        {"role":"user", "content": user_input}
    ]

    # 1차 LLM 호출 (툴 호출 여부 판단)
    first = llm.chat(messages, tools_spec)
    messages.append({"role":"assistant", "content": first.get("content","")})

    # LLM이 MCP 툴을 호출하라고 지시하는 경우
    for call in first.get("tool_calls", []):
        name = call["name"]
        args = call["arguments"]
        result = mcp.call_tool(name, args)         # ← 표준 호출
        # MCP 서버의 결과를 도구 메시지로 전달
        messages.append({"role":"tool", "name": name, "content": str(result)})

    # 툴 결과 반영하여 최종 답변
    final = llm.chat(messages, tools_spec=None)    # 이번엔 도구 스펙 없이 마무리
    return final["content"]

# 데모
print(agent_run("슬랙 general의 최근 메시지 한 줄 요약해줘"))
print(agent_run("드라이브 root의 파일 목록을 요약해줘"))
