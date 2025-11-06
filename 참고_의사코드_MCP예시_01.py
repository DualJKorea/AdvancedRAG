# pip install langchain langchain-openai openai mcp
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool

import asyncio
from mcp import Client # 예시: mcp 파이썬 클라이언트 (서버 구현/주소는 환경에 맞게)

# ---- (A) MCP 서버에서 툴을 동적으로 발견하고 LC Tool로 감싸기 ----
async def load_mcp_tools_as_langchain(server_url: str):
    lc_tools = []
    async with Client.connect(server_url) as client:
        # 1) MCP 서버가 제공하는 툴 목록 가져오기
        tool_list = await client.list_tools() # [{name, description, inputSchema, ...}, ...]

    for t in tool_list:
        name = t["name"]
        desc = t.get("description", "")
        schema = t.get("inputSchema") # JSON Schema

# 2) LangChain StructuredTool로 래핑
async def _mcp_caller(**kwargs):
    # MCP 표준 호출
    result = await client.call_tool(name=name, arguments=kwargs)
    # result 형식은 MCP 서버 구현에 따라 다를 수 있음
    return result.get("content", result)

# LangChain은 sync 호출을 기대하므로 간단 sync wrapper 제공
def sync_caller(**kwargs):
    return asyncio.run(_mcp_caller(**kwargs))

    lc_tools.append(
    StructuredTool.from_function(
            name=name,
            description=desc or f"MCP tool: {name}",
            func=sync_caller,
            args_schema=None, # 필요 시 pydantic 모델로 schema를 반영할 수 있음
            )
        )
    return lc_tools

# ---- (B) 에이전트 구성 (툴은 MCP로부터 주입) ----
def build_agent_with_mcp(tools):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a precise unit-conversion assistant. Use tools as needed."),
    ("human", "{input}")
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    # 예: mcp 서버가 'convert_length', 'convert_weight' 툴을 제공한다고 가정
    server = "ws://localhost:9999" # 환경에 맞게
    mcp_tools = asyncio.run(load_mcp_tools_as_langchain(server))
    executor = build_agent_with_mcp(mcp_tools)

    print(executor.invoke({"input": "2.54 inch를 meter로. 소수점 4자리."})["output"])
    print(executor.invoke({"input": "82 kg은 몇 파운드? 정수 반올림."})["output"])

