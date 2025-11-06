import streamlit as st
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
import io, sys, re, functools

# ---------------- 벡터 저장소 ----------------
def create_faiss_docs(embedding_model):
    documents = [
        Document(page_content="React는 사용자 인터페이스(UI)를 구축하는 JavaScript 라이브러리입니다."),
        Document(page_content="React는 브라우저에서 실행되며, 상태 관리와 컴포넌트 기반 설계가 핵심입니다."),
        Document(page_content="Spring Boot는 Java 기반의 백엔드 프레임워크로 REST API와 서버 사이드 애플리케이션 구축에 사용됩니다."),
        Document(page_content="Spring Boot는 데이터베이스 연동, 비즈니스 로직 처리, 서버 운영 환경 설정이 용이합니다."),
        Document(page_content="React는 프론트엔드 개발에 최적화되어 있으며, Spring Boot는 백엔드 개발에 최적화되어 있습니다."),
        Document(page_content="React와 Spring Boot는 서로 다른 레이어에서 동작하며, 함께 사용하여 풀스택 애플리케이션을 구성할 수 있습니다."),
    ]
    return FAISS.from_documents(documents, embedding_model)

def strip_ansi(text: str) -> str:
    return re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', text)

# ---------------- RAG 로깅 용 ----------------
def capture_logs(label):
    """데코레이터: stdout 캡처 + 프롬프트/결과 로깅"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            old_stdout = sys.stdout
            buf = io.StringIO()
            sys.stdout = buf
            try:
                print(f"\n[{label}] called")
                if args: print(f"args: {args}")
                if kwargs: print(f"kwargs: {kwargs}")

                result = func(*args, **kwargs)

                print(f"[{label}] result:\n{result}\n")

                logs = strip_ansi(buf.getvalue())
                st.session_state.logs.append(logs)   
            finally:
                sys.stdout = old_stdout
            return result
        return wrapper
    return decorator

@capture_logs("LLM")
def run_llm(llm_func, prompt):
    return llm_func(prompt)

@capture_logs("Agent Answer")
def answer_with_agent(agent, question):
    return agent.run(f"User question: {question}\nUse tools as needed to answer.")

# ---------------- 스트림릿  UI ----------------
st.set_page_config(page_title="FAISS + Multi-Tool Agent Demo", layout="wide")

st.markdown("""
<style>
.chat-bubble{padding:.6em 1em;border-radius:1em;max-width:70%;margin:.4em 0;word-wrap:break-word}
.user-bubble{background:#2C3E50;color:white;margin-left:auto;text-align:right}
.assistant-bubble{background:#E0E0E0;color:black;margin-right:auto;text-align:left}
.logs-bubble{background:#1E1E1E;color:#00FF00;padding:.5em;border-radius:.5em;white-space:pre-wrap;margin:.4em 0;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    OPENAI_API_KEY = st.text_input("OpenAI API key", type="password")

st.session_state.setdefault("chat", [])
st.session_state.setdefault("agent_ready", False)
st.session_state.setdefault("logs", [])   

st.title("💬 FAISS + Multi-Tool Agent with Logs Demo")

if OPENAI_API_KEY and not st.session_state.agent_ready:
    # LLM & FAISS
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_db = create_faiss_docs(emb)

    # Tools
    retriever_tool = Tool(
        name="RetrieverTool",
        func=lambda q: "\n".join([d.page_content for d in vector_db.similarity_search(q, k=3)]),
        description="Vector DB에서 관련 문서를 검색합니다."
    )
    search_tool = DuckDuckGoSearchRun()
    tools = [retriever_tool, search_tool]

    # 에이전트 - 제로샷
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=3
    )

    st.session_state.update({
        "llm": llm,
        "vector_db": vector_db,
        "agent": agent,
        "agent_ready": True
    })

    # 초기 질문
    init_question = "React와 Spring Boot의 주요 차이점을 알려줘."
    answer = answer_with_agent(agent, init_question)
    st.session_state.chat.extend([
        ("user", init_question),
        ("assistant", answer),
    ])

# ---------------- 챗 UI ----------------
st.markdown("---")
st.subheader("Chat")
for role, msg in st.session_state.chat:
    cls = "user-bubble" if role=="user" else "assistant-bubble"
    st.markdown(f"<div class='chat-bubble {cls}'><b>{role.title()}:</b> {msg}</div>", unsafe_allow_html=True)

# ---------------- 로그 UI ----------------
if st.session_state.logs:
    st.markdown("---")
    st.subheader("Execution Logs")
    full_logs = "\n".join(st.session_state.logs)
    st.markdown(f"<div class='logs-bubble'>{full_logs}</div>", unsafe_allow_html=True)

# ---------------- 챗 압력 ----------------
if st.session_state.get("agent_ready") and (prompt := st.chat_input("Your question")):
    agent = st.session_state.agent
    answer = answer_with_agent(agent, prompt)
    st.session_state.chat.extend([
        ("user", prompt),
        ("assistant", answer),
    ])
    st.experimental_rerun()

st.markdown("---")
st.caption("FAISS + Multi-Tool Agent demo with stdout logs and top-k retrieval.")
