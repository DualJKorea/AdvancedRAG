import streamlit as st, tempfile, os, time, shutil
from PyPDF2 import PdfReader
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# ---------------- 벡터 적재 및 검색 ----------------
def load_pdf(path): 
    return "\n".join(
        [(p.extract_text() or "") for p in PdfReader(path).pages])

def chunk_text(txt, size=1000, overlap=200): 
    return RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap).split_text(txt)

def create_vectordb(texts, key, persist): 
    vdb = Chroma.from_texts(
        texts, OpenAIEmbeddings(openai_api_key=key), persist_directory=persist)
    vdb.persist(); return vdb

def make_chain(key, retriever): 
    return ConversationalRetrievalChain.from_llm(
        ChatOpenAI(openai_api_key=key, model="gpt-4o-mini", temperature=0), retriever)

# ---------------- 스트림릿 UI ---------------------
st.set_page_config(page_title="Simple RAG Demo", layout="wide")

st.markdown("""
<style>
.chat-bubble{padding:.6em 1em;border-radius:1em;max-width:70%;margin:.4em 0;color:white;word-wrap:break-word}
.user-bubble{background:#2C3E50;margin-left:auto;text-align:right}
.assistant-bubble{background:#3D3D3D;margin-right:auto;text-align:left}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    key = st.text_input("OpenAI API key", type="password")
    pdf = st.file_uploader("Upload PDF", type="pdf")
    persist_dir = st.text_input("Chroma persist dir", "./chroma_db")
    if st.button("Clear DB") and os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        st.session_state.clear()

st.title("💬 Simple RAG — Chat with your PDF")
st.session_state.setdefault("chat", [])
st.session_state.setdefault("qa", None)

# PDF 처리
if pdf and key and st.session_state.qa is None:
    with st.spinner("Indexing PDF..."):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf.read()); tmp.close()
        text = load_pdf(tmp.name)
        if not text.strip(): st.error("No text extracted.")
        else:
            chunks = chunk_text(text)
            vdb = create_vectordb(chunks, key, os.path.join(persist_dir, f"chroma_{int(time.time())}"))
            st.session_state.qa = make_chain(key, vdb.as_retriever(search_kwargs={"k":4}))
            st.success(f"Indexed {len(chunks)} chunks.")

# 채팅 UI
st.markdown("---"); st.subheader("Chat")

for role, msg in st.session_state.chat:
    bubble = "user-bubble" if role=="user" else "assistant-bubble"
    st.markdown(f"<div class='chat-bubble {bubble}'><b>{role.title()}:</b> {msg}</div>", unsafe_allow_html=True)

# 챗 입력 
if prompt := st.chat_input("Your question"):
    st.session_state.chat.append(("user", prompt))
    if st.session_state.qa:
        try: answer = st.session_state.qa({"question": prompt, "chat_history": st.session_state.chat})["answer"]
        except Exception as e: answer = f"Error: {e}"
    else: answer = "Upload a PDF and provide API key first."
    st.session_state.chat.append(("assistant", answer))
    st.experimental_rerun()

st.markdown("---")
st.caption("Simple RAG demo with dark chat bubbles.")
