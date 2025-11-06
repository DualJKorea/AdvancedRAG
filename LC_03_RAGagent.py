import streamlit as st
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.tools import tool
import io, sys, re

# ---------------- 계산 함수 ----------------
def simple_calculator(expression: str) -> str:
    try:
        # Fix exponentiation syntax
        expression = expression.replace("^", "**")
        return str(eval(expression))
    except Exception as e:
        return f"계산 오류: {str(e)}"

def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape sequences from stdout"""
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="RAG Agent Demo", layout="wide")

st.markdown("""
<style>
.chat-bubble{padding:.6em 1em;border-radius:1em;max-width:70%;margin:.4em 0;color:black;word-wrap:break-word}
.user-bubble{background:#2C3E50;margin-left:auto;text-align:right;color:white;}
.assistant-bubble{background:#3D3D3D;margin-right:auto;text-align:left;color:white;}
.logs-bubble{background:#1E1E1E;color:#00FF00;padding:.5em;border-radius:.5em;white-space:pre-wrap;margin:.4em 0;}
</style>
""", unsafe_allow_html=True)

# OpenAI API 키 설정
with st.sidebar:
    st.header("Settings")
    OPENAI_API_KEY = st.text_input("OpenAI API key", type="password")

st.session_state.setdefault("chat", [])
st.session_state.setdefault("agent_ready", False)
st.session_state.setdefault("initial_question_sent", False)

st.title("💬 RAG Agent Demo with langchain Logs")

# FAISS 준비
if OPENAI_API_KEY and not st.session_state.agent_ready:
    with st.spinner("Preparing FAISS vector store and Agent..."):
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
        embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        docs = [
            Document(page_content="지구는 태양으로부터 약 1억 5천만 km 떨어져 있습니다."),
            Document(page_content="1 광년은 약 9조 4600억 km입니다.")
        ]
        vectorstore = FAISS.from_documents(docs, embedding_model)
        retriever = vectorstore.as_retriever()
        retrieval_qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        @tool
        def SearchDocs(question: str) -> str:
            """사용자 질문에 관련된 문서를 검색합니다."""
            return retrieval_qa.run(question)

        tools = [
            Tool.from_function(name="SearchDocs", func=SearchDocs.run, description="문서 검색"),
            Tool.from_function(name="Calc", func=simple_calculator, description="계산 수행")
        ]

        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent="zero-shot-react-description",
            verbose=True   
        )

        st.session_state.update({"agent": agent, "agent_ready": True})

# ---------------- 챗 UI ----------------
st.markdown("---")
st.subheader("Chat")

for role, msg in st.session_state.chat:
    if role == "logs":
        st.markdown(f"<div class='logs-bubble'>{msg}</div>", unsafe_allow_html=True)
    else:
        bubble_class = "user-bubble" if role=="user" else "assistant-bubble"
        st.markdown(f"<div class='chat-bubble {bubble_class}'><b>{role.title()}:</b> {msg}</div>", unsafe_allow_html=True)

# ---------------- 에이전트 실행 ----------------
def run_agent(prompt: str):
    agent = st.session_state.agent

    buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer

    try:
        response = agent.run(prompt)
    except Exception as e:
        response = f"Error: {str(e)}"

    sys.stdout = sys_stdout
    logs_text = strip_ansi_codes(buffer.getvalue())

    st.session_state.chat.append(("user", prompt))
    st.session_state.chat.append(("assistant", response))
    st.session_state.chat.append(("logs", logs_text))
    st.experimental_rerun()

# ---------------- 챗 기본 질의 ----------------
if st.session_state.agent_ready and not st.session_state.initial_question_sent:
    st.session_state.initial_question_sent = True
    run_agent("지구에서 태양까지의 거리를 광년으로 환산하면?")

# ---------------- 챗 입력창 ----------------
if st.session_state.get("agent_ready") and (prompt := st.chat_input("Your question")):
    run_agent(prompt)

st.markdown("---")
st.caption("RAG Agent demo with clean stdout logs showing reasoning, tool calls, and LLM outputs.")
