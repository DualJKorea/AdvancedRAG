# mcp_tool_server.py  (의사코드)

class MCPServer:
    def __init__(self):
        self.tools = []

    def register_tool(self, name, description, input_schema, handler):
        self.tools.append({
            "name": name,
            "description": description,
            "inputSchema": input_schema,   # JSON Schema
            "handler": handler,            # 실제 실행 함수
        })

    def list_tools(self):
        # 클라이언트에게 스키마만 전달 (핸들러는 서버 내부)
        return [{k: t[k] for k in ["name","description","inputSchema"]} for t in self.tools]

    def call_tool(self, name, arguments):
        tool = find(self.tools, lambda t: t["name"] == name)
        validate(arguments, tool["inputSchema"])      # 서버가 표준 스키마로 검증
        try:
            return {"ok": True, "content": tool["handler"](**arguments)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

# ---- 실제 툴 구현 (Slack/Drive) ----
def slack_query_messages(token: str, channel: str) -> str:
    # 실제로는 slack_sdk 사용
    return f"[Slack:{channel}] 최근 메시지(요약)"

def drive_list_files(token: str, folder_id: str) -> str:
    # 실제로는 google-api-python-client 사용
    return f"[Drive:{folder_id}] 파일 목록(요약)"

SLACK_SCHEMA = {
  "type":"object",
  "properties":{
    "token":{"type":"string"},
    "channel":{"type":"string"}
  },
  "required":["token","channel"]
}

DRIVE_SCHEMA = {
  "type":"object",
  "properties":{
    "token":{"type":"string"},
    "folder_id":{"type":"string"}
  },
  "required":["token","folder_id"]
}

server = MCPServer()
server.register_tool(
  name="slack.query_messages",
  description="슬랙 채널 최근 메시지 요약",
  input_schema=SLACK_SCHEMA,
  handler=slack_query_messages
)
server.register_tool(
  name="drive.list_files",
  description="구글 드라이브 파일 목록 요약",
  input_schema=DRIVE_SCHEMA,
  handler=drive_list_files
)

serve_over_ws(server, host="0.0.0.0", port=9999)  # WS/STDIO 등 프로토콜로 노출
