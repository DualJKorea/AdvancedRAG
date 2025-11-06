# pip install -U langchain langchain-openai openai

from langchain.agents import AgentExecutor, create_openai_tools_agent, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# ---- (A) 로컬 tool 정의 (@tool) ----
@tool
def convert_length(value: float, from_unit: str, to_unit: str) -> float:
    """길이 환산: meter<->inch"""
    fu, tu = from_unit.lower(), to_unit.lower()
    if fu in ["m", "meter", "meters"] and tu in ["inch", "inches", "in"]:
        return value * 39.3700787
    if fu in ["inch", "inches", "in"] and tu in ["m", "meter", "meters"]:
        return value / 39.3700787
    raise ValueError("지원 단위: m<->inch")

@tool
def convert_weight(value: float, from_unit: str, to_unit: str) -> float:
    """무게 환산: kg<->lb"""
    fu, tu = from_unit.lower(), to_unit.lower()
    if fu in ["kg", "kilogram", "kilograms"] and tu in ["lb", "lbs", "pound", "pounds"]:
        return value * 2.20462262185
    if fu in ["lb", "lbs", "pound", "pounds"] and tu in ["kg", "kilogram", "kilograms"]:
        return value / 2.20462262185
    raise ValueError("지원 단위: kg<->lb")

tools = [convert_length, convert_weight]

# ---- (B) LLM + 프롬프트 + 에이전트 구성 ----
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
("system", "You are a precise unit-conversion assistant. Use tools as needed. "
"Follow the user's formatting instructions (e.g., rounding, decimal places)."),
("human", "{input}"),
# 반드시 포함: 도구 호출/중간 추론을 담는 메시지 버퍼
MessagesPlaceholder("agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    print(executor.invoke({"input": "1.8 meter를 inch로. 소수점 2자리."})["output"])
    print(executor.invoke({"input": "70 kg을 파운드로. 정수 반올림."})["output"])

