# AdvancedRAG

# One Day "AdvancedRAG"
Inflearn 강의 [하루만에 끝내는 Advanced RAG 핵심 정리] 에서 데모에 사용된 파이썬 소스 코드입니다.


## 강의 내용 중에서...


## 소스 코드 목차

    LC_01_simpleRAG.py
    LC_02_hydeRAG.py
    LC_03_RAGagent.py
    LC_04_rerankRAG.py
    LC_05_multiToolAgentRAG.py
    LC_06_hybridSearchRAG.py
    LF_03_server-everything.json
    requirements.txt


# README - 데모 안내

## 1. RAG langchain 환경 셋업
```bash
conda -V
conda info --envs
conda create -n aidemo python=3.11.7
conda activate aidemo
python --version
pip install -r requirements.txt
```

### open api key 설정
open api 호출을 위해 key를 준비함

## 2. RAG langchain 데모 
```bash
streamlit run LC_01_simpleRAG.py
streamlit run LC_02_hydeRAG.py
streamlit run LC_03_RAGagent.py
streamlit run LC_04_rerankRAG.py
streamlit run LC_05_multiToolAgentRAG.py
streamlit run LC_06_hybridSearchRAG.py
```

## 3. RAG langflow 환경 셋업

### uv 환경 셋업
```bash
cd ../langflow
uv init  
uv pip install langflow 
cd ..
```

## 4. RAG langflow 데모 

### langflow 구동
```bash
uv run langflow run
```
