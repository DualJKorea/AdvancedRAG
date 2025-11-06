"""
문서 요약 에이전트 (일반적인 Agent 흐름 반영)
- Goal/Task 수신 → Planning → Retrieve → Reasoning(의사결정)
→ Tool Use → Memory → Agent Loop(Plan→Act→Review→Observe) → Output
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import os

# --- LLM / VectorStore ---
from langchain_openai import ChatOpenAI          # 최신 경로
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# ====== 0) 환경 / 구성 ======
OPENAI_MODEL = "gpt-4o-mini"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ====== 1) 샘플 코퍼스 적재(ChromaDB Retriever) ======
docs = [
    Document(page_content=("LangChain은 LLM 앱 개발을 돕는 프레임워크입니다. "
                           "체인과 에이전트를 쉽게 구성할 수 있습니다.")),
    Document(page_content=("RAG는 외부 지식을 검색해 답변 품질을 높이는 기법입니다. "
                           "검색 결과를 컨텍스트로 활용합니다.")),
    Document(page_content=("ChromaDB는 가벼운 로컬 벡터 데이터베이스입니다. "
                           "임베딩을 저장하고 유사도 검색을 수행합니다.")),
    Document(page_content=("요약은 핵심을 간결히 전달해야 합니다. 사실 위주로 작성하고 과도한 추측은 금지합니다."))
]
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ====== 2) LLM / 프롬프트 ======
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
parser = StrOutputParser()

SUMMARY_PROMPT = PromptTemplate.from_template(
    """다음 컨텍스트를 한국어로 {style} 요약해 주세요.
- 사실 위주, 불필요한 수식어/추측 금지
- {sentences}문장 이내

[컨텍스트]
{context}
"""
)
REVIEW_PROMPT = PromptTemplate.from_template(
    """다음 요약이 지침을 지키는지 검토하고, 필요하면 수정본만 출력하세요.
지침: 사실 위주, 간결, {sentences}문장 이내, 컨텍스트 벗어난 내용 금지

[컨텍스트]
{context}

[초안]
{draft}
"""
)

# ====== 3) Tool (예: 단어 수 / 형식 검증) ======
def count_words(text: str) -> int:
    return len(text.split())

def format_check(summary: str, max_sent: int) -> Dict[str, Any]:
    # 매우 단순한 문장 수 점검
    sents = [s.strip() for s in summary.replace("\n", " ").split(".") if s.strip()]
    ok = len(sents) <= max_sent
    return {"ok": ok, "sentences": len(sents)}

# ====== 4) 요약/리뷰 함수(Act 단계에서 사용) ======
def summarize_with_llm(context: str, summary_type: str = "short") -> str:
    sentences = 2 if summary_type == "short" else 5
    style = "짧게" if summary_type == "short" else "자세히"

    draft = (SUMMARY_PROMPT | llm | parser).invoke({
        "context": context,
        "style": style,
        "sentences": sentences
    })
    final = (REVIEW_PROMPT | llm | parser).invoke({
        "context": context,
        "draft": draft,
        "sentences": sentences
    })
    return final.strip()

# ====== 5) 상태 구조 ======
@dataclass
class AgentState:
    goal: str
    plan: List[str] = field(default_factory=list)
    context: str = ""
    summary_type: str = "short"    # reasoning 결과
    draft: str = ""
    final_summary: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)

# ====== 6) 단계별 함수 ======
# Goal/Task
def receive_goal(user_input: str) -> AgentState:
    print(f"[Goal/Task] 입력: {user_input}")
    return AgentState(goal=user_input)

# Planning
def make_plan(state: AgentState) -> AgentState:
    state.plan = ["문서 검색", "요약 작성", "검토/수정", "결과 출력"]
    print(f"[Planning] {state.plan}")
    return state

# Retriever
def retrieve_context(state: AgentState) -> AgentState:
    query = "LangChain, RAG, ChromaDB 핵심 요약"
    docs = retriever.get_relevant_documents(query)
    state.context = " ".join(d.page_content for d in docs)
    state.metrics["context_words"] = count_words(state.context)
    print(f"[Retriever] 컨텍스트 단어 수: {state.metrics['context_words']}")
    return state

# Reasoning (요약 길이/톤 등 결정)
def decide_strategy(state: AgentState) -> AgentState:
    state.summary_type = "short" if any(k in state.goal for k in ["짧게", "간단"]) else "long"
    print(f"[Reasoning] summary_type = {state.summary_type}")
    return state

# Act (Tool + LLM 실행)
def act_summarize(state: AgentState) -> AgentState:
    state.draft = summarize_with_llm(state.context, state.summary_type)
    print("[Act] 초안 요약 생성")
    return state

# Review (Tool로 검증 + 필요시 재수정)
def review_and_fix(state: AgentState) -> AgentState:
    max_sent = 2 if state.summary_type == "short" else 5
    check = format_check(state.draft, max_sent)
    print(f"[Review] 문장 수={check['sentences']} (허용 {max_sent}) → {'OK' if check['ok'] else '수정'}")
    if not check["ok"]:
        # 조건 미달 시 다시 요약(간단 처리: 한 번 더 압축 요청)
        state.draft = summarize_with_llm(state.context, state.summary_type)
    state.final_summary = state.draft
    return state

# Observe (Memory 반영)
def observe_and_store(state: AgentState) -> AgentState:
    state.memory.update({
        "last_goal": state.goal,
        "last_summary_type": state.summary_type,
        "last_context_words": state.metrics.get("context_words", 0),
    })
    print(f"[Memory] {state.memory}")
    return state

# Output
def render_output(state: AgentState) -> None:
    print("\n[Output Generator] 최종 요약")
    print(state.final_summary)

# ====== 7) Agent Loop (Plan → Act → Review → Observe) ======
def run_agent(user_goal: str):
    # Goal/Task
    state = receive_goal(user_goal)
    # Planning
    state = make_plan(state)
    # Retrieval & Reasoning (계획 앞/뒤 모두 가능하지만 강의용으로 명시)
    state = retrieve_context(state)
    state = decide_strategy(state)

    # Loop over plan
    for step in state.plan:
        print(f"\n[Agent Loop] 단계: {step}")
        if step == "문서 검색":
            # 실제 구조에서는 이미 검색 완료. 예시로 재호출 가능
            state = retrieve_context(state)
        elif step == "요약 작성":
            state = act_summarize(state)
        elif step == "검토/수정":
            state = review_and_fix(state)
            state = observe_and_store(state)  # 리뷰 결과 메모리에 반영
        elif step == "결과 출력":
            render_output(state)

# ====== 실행 ======
if __name__ == "__main__":
    # 예) "…짧게 요약해줘" → short 전략, 그렇지 않으면 long
    user_goal = "문서 내용을 짧게 요약해줘"
    run_agent(user_goal)
