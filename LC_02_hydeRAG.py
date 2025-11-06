import streamlit as st
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import functools
import io
import sys

# ---------------- RAG 로깅용 ----------------
def capture_logs(label):
    """데코레이터: 함수 실행 시 stdout 캡처 + 프롬프트/결과 로깅"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            old_stdout = sys.stdout
            buf = io.StringIO()
            sys.stdout = buf
            try:
                print(f"\n[{label}] called")
                if kwargs:
                    print(f"→ kwargs: {kwargs}")

                result = func(*args, **kwargs)

                if "question" in kwargs:
                    print(f"[{label}] question: {kwargs['question']}")
                if "context" in kwargs:
                    print(f"[{label}] context: {kwargs['context']}")

                print(f"[{label}] result:\n{result}\n")

                logs = buf.getvalue()    
            finally:
                sys.stdout = old_stdout   
            return result, logs
        return wrapper
    return decorator

# ---------------- 벡터 적재 및 hyde RAG 처리 ----------------
def create_faiss_docs(embedding_model):
    documents = [
        Document(page_content="React is a frontend framework for building UIs."),
        Document(page_content="Spring Boot is a Java backend framework for microservices."),
        Document(page_content="PostgreSQL is an open-source relational database."),
    ]
    return FAISS.from_documents(documents, embedding_model)


@capture_logs("HyDE Answer")
def get_hyde_answer(llm, question):
    prompt = f"Answer the question: {question}"
    return llm.predict(prompt)


@capture_logs("Retriever")
def logging_retriever(vectorstore, question):
    docs = vectorstore.similarity_search(question, k=2)
    return docs


@capture_logs("Final Answer")
def get_final_answer(llm, question, context):
    final_prompt = f"""Answer the question below using the following context. 
Question: {question}
Context: {context}"""
    return llm.predict(final_prompt)


# ---------------- 스트림릿 UI ----------------
st.set_page_config(page_title="RAG + HyDE Logs", page_icon="🤖", layout="centered")

if "chat" not in st.session_state:
    st.session_state.chat = []

st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

if api_key:
    llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = create_faiss_docs(embeddings)

    init_question = "리액트와 스프링 부트의 가장 큰 차이는 뭐야?"
    st.session_state.chat.append(("user", init_question))

    # HyDE 처리
    hyde_answer, logs_hyde = get_hyde_answer(llm=llm, question=init_question)

    # Retriever 처리
    retrieved_docs, logs_retriever = logging_retriever(vectorstore=vectorstore, question=hyde_answer)

    # Final answer 처리
    context = " ".join([doc.page_content for doc in retrieved_docs])
    final_answer, logs_final = get_final_answer(llm=llm, question=init_question, context=context)

    # 로그 합치기
    full_logs = logs_hyde + logs_retriever + logs_final

    # 세션 기록
    st.session_state.chat.append(("assistant", final_answer))
    st.session_state.chat.append(("logs", full_logs))


# ---------------- 챗 UI ----------------
st.title("💬 RAG with HyDE + Logs")

for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"<div style='text-align:right; background:#DCF8C6; padding:8px; border-radius:10px; margin:4px; color:black;'>{msg}</div>", unsafe_allow_html=True)
    elif role == "assistant":
        st.markdown(f"<div style='text-align:left; background:#E6E6E6; padding:8px; border-radius:10px; margin:4px; color:black;'>{msg}</div>", unsafe_allow_html=True)
    elif role == "logs":
        st.markdown("---")
        st.markdown("**📝 Logs:**")
        st.text(msg)
