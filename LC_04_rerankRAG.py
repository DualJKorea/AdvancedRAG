import streamlit as st
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import io, sys, re, functools

# ---------------- Decorator ----------------
def capture_logs(label):
    """데코레이터: 함수 실행 시 stdout 캡처 + 입력/출력 로깅"""
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
                logs = buf.getvalue()
            finally:
                sys.stdout = old_stdout

            # 로그 누적
            if "logs" not in st.session_state:
                st.session_state.logs = ""
            st.session_state.logs += logs
            return result
        return wrapper
    return decorator

# ---------------- 벡터 저장 ----------------
def create_faiss_docs(embedding_model):
    documents = [
        Document(page_content="React is a frontend framework for building UIs."),
        Document(page_content="Spring Boot is a Java backend framework for microservices."),
        Document(page_content="PostgreSQL is an open-source relational database."),
    ]
    return FAISS.from_documents(documents, embedding_model)

def strip_ansi(text: str) -> str:
    return re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', text)

@capture_logs("LLM Call")
def run_llm(llm, prompt):
    result = llm.predict(prompt)
    return result, strip_ansi("")

@capture_logs("Answer Pipeline")
def answer_question(llm, vector_db, question, top_k=2):
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)

    # rerank 점수 기반 정렬
    scored_docs = []
    for doc in docs:
        score, _ = run_llm(llm, f"Rate relevance of this doc to the question on a scale 0-1:\nQ: {question}\nDoc: {doc.page_content}")
        try:
            score = float(score[0].strip()) if isinstance(score, tuple) else float(str(score).strip())
        except:
            score = 0.0
        scored_docs.append((doc, score))

    top_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:top_k]
    context = "\n".join([d.page_content for d, _ in top_docs])
    answer, _ = run_llm(llm, f"Answer the question below using the context:\nQ: {question}\nContext:\n{context}")
    return answer

# ---------------- 스트림릿 UI ----------------
st.set_page_config(page_title="FAISS + ReRank Demo", layout="wide")

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
st.session_state.setdefault("faiss_ready", False)
st.session_state.setdefault("logs", "")

st.title("💬 FAISS + ReRank Demo Chat with Logs")

if OPENAI_API_KEY and not st.session_state.faiss_ready:
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_db = create_faiss_docs(emb)

    st.session_state.update({
        "llm": llm,
        "vector_db": vector_db,
        "faiss_ready": True
    })

    # 최초 질문
    init_question = "백엔드 개발자에게 필요한 기술은?"
    answer = answer_question(llm, vector_db, init_question)
    st.session_state.chat.extend([
        ("user", init_question),
        ("assistant", answer)
    ])

# 챗 UI
st.markdown("---")
st.subheader("Chat")
for role, msg in st.session_state.chat:
    cls = "user-bubble" if role=="user" else "assistant-bubble"
    st.markdown(f"<div class='chat-bubble {cls}'><b>{role.title()}:</b> {msg}</div>", unsafe_allow_html=True)

# 로그 출력
if st.session_state.logs:
    st.markdown("---")
    st.markdown("**📝 Logs:**")
    st.text(st.session_state.logs)

# 챗 입력
if st.session_state.get("faiss_ready") and (prompt := st.chat_input("Your question")):
    llm = st.session_state.llm
    vector_db = st.session_state.vector_db
    answer = answer_question(llm, vector_db, prompt)
    st.session_state.chat.extend([
        ("user", prompt),
        ("assistant", answer)
    ])
    st.experimental_rerun()

st.markdown("---")
st.caption("FAISS + ReRank demo with decorator-based logging shown at the bottom.")
