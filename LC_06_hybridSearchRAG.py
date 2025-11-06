import streamlit as st
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import io, sys, re, functools

def strip_ansi(text: str) -> str:
    return re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', text)

# ---------------- 로깅용 ----------------
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
                st.session_state.logs.append(logs)  # 로그 누적
            finally:
                sys.stdout = old_stdout
            return result
        return wrapper
    return decorator

# ---------------- Retriever 생성 ----------------
@capture_logs("Hybrid Retriever Init")
def create_hybrid_retriever(embedding_model):
    documents = [
        Document(page_content="React is a frontend JavaScript framework."),
        Document(page_content="Spring Boot is a backend Java framework."),
        Document(page_content="PostgreSQL is a powerful relational database."),
        Document(page_content="ElasticSearch enables full-text search using inverted index."),
    ]

    # Dense (FAISS) 벡터 검색
    dense_vector_db = FAISS.from_documents(documents, embedding_model)
    dense_retriever = dense_vector_db.as_retriever(search_kwargs={"k": 3})
    print(f"Dense retriever initialized: {dense_retriever}")
    print(f"  - type: {type(dense_retriever)}")
    print(f"  - k: {dense_retriever.search_kwargs['k']}")

    # Sparse (BM25) 키워드 가중치
    sparse_retriever = BM25Retriever.from_documents(documents)
    sparse_retriever.k = 3
    print(f"Sparse BM25 retriever initialized: {sparse_retriever}")
    print(f"  - type: {type(sparse_retriever)}")
    print(f"  - k: {sparse_retriever.k}")

    # Hybrid 검색
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.5, 0.5]
    )
    print(f"Hybrid Ensemble Retriever created combining dense + sparse retrievers.")
    print(f"  - type: {type(hybrid_retriever)}")
    print(f"  - retrievers count: {len(hybrid_retriever.retrievers)}")

    return hybrid_retriever

# ---------------- LLM 호출 ----------------
@capture_logs("LLM Call")
def run_llm(llm, prompt):
    result = llm.predict(prompt)
    print(f"Prompt:\n{prompt}\n\nAnswer:\n{result}")
    return result

def answer_with_llm(llm, retriever, question):
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""
Answer the question using the following context:

Question: {question}
Context:
{context}
"""
    answer = run_llm(llm, prompt)
    return answer

# ---------------- 스트림릿 UI ----------------
st.set_page_config(page_title="Hybrid Retriever RAG Demo", layout="wide")
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

st.title("💬 Hybrid Retriever RAG Demo with Logs")

if OPENAI_API_KEY and not st.session_state.agent_ready:
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    emb_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    hybrid_retriever = create_hybrid_retriever(emb_model)

    st.session_state.update({
        "llm": llm,
        "retriever": hybrid_retriever,
        "agent_ready": True
    })

    # 초기 질문 
    init_question = "리액트와 스프링부트의 가장 큰 차이는 뭐야?"
    answer = answer_with_llm(llm, hybrid_retriever, init_question)
    st.session_state.chat.extend([
        ("user", init_question),
        ("assistant", answer)
    ])

# ---------------- 챗 UI ----------------
st.markdown("---")
st.subheader("Chat")
for role, msg in st.session_state.chat:
    if role == "logs":
        st.markdown(f"<div class='logs-bubble'>{msg}</div>", unsafe_allow_html=True)
    else:
        cls = "user-bubble" if role=="user" else "assistant-bubble"
        st.markdown(f"<div class='chat-bubble {cls}'><b>{role.title()}:</b> {msg}</div>", unsafe_allow_html=True)

# ---------------- 챗 입력 ----------------
if st.session_state.get("agent_ready") and (prompt := st.chat_input("Your question")):
    llm = st.session_state.llm
    retriever = st.session_state.retriever
    answer = answer_with_llm(llm, retriever, prompt)
    st.session_state.chat.extend([
        ("user", prompt),
        ("assistant", answer)
    ])
    st.experimental_rerun()

# ---------------- 로그 출력 ----------------
if st.session_state.logs:
    st.markdown("---")
    st.subheader("📝 Logs")
    st.text("\n".join(st.session_state.logs))

st.markdown("---")
st.caption("Hybrid Retriever RAG demo using FAISS + BM25 with logs displayed at the end.")
