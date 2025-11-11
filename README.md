# 한번에 끝내는 LLM부터 AI Agent까지: RAG, MCP로 확장되는 생성형 AI의 진화
Inflearn 강의 [한번에 끝내는 LLM부터 AI Agent까지: RAG, MCP로 확장되는 생성형 AI의 진화] 에서 데모에 사용된 소스 코드입니다.


## 강의 내용 중에서..


## 소스 코드 목차

|코드|설명|     
|---|----|
|requirements.txt | langchain 패키지 셋업 |
|LC_01_simpleRAG.py | 심플RAG |
|LC_02_hydeRAG.py | hydeRAG|
|LC_03_RAGagent.py | RAG agent(tool call) |
|LC_04_rerankRAG.py | rerank RAG |
|LC_05_multiToolAgentRAG.py | multi-tool RAG Agent(multi tool) |
|LC_06_hybridSearchRAG.py | hybrid search RAG|
|LF_01_flow.json | langflow기반 simple RAG |
|LF_02_flow.json | langflow기반 AI Agent |
|LF_03_flow.json | langflow기반 MCP연동 AI Agent|
|LF_03_mcp_run.sh | everything mcp서버 구동|
|LF_03_server-everything.json | langflow MCP서버 등록 |
|참고_의사코드_MCP예시_01.py | 강의 이론 설명용 의사코드(흐름만 참조) |
|참고_의사코드_MCP예시_02.py | 강의 이론 설명용 의사코드(흐름만 참조) |
|참고_의사코드_MCP예시_03.py | 강의 이론 설명용 의사코드(흐름만 참조) |
|참고_의사코드_non-MCP예시_01.py | 강의 이론 설명용 의사코드(흐름만 참조) |
|참고_의사코드_non-MCP예시_02.py | 강의 이론 설명용 의사코드(흐름만 참조) |
|참고_의사코드_문서요약에이전트.py | 강의 이론 설명용 의사코드(흐름만 참조) |
|강의작성 참조자료.md | 강의 작성 참고자료 |


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

langflow 데스크탑 설치 

## 4. RAG langflow 데모 

langflow 데스크탑 구동 

langflow에 LF_0x_flow.json을 import 하여 실습 진행

### mcp 환경 셋업

```bash
npx @modelcontextprotocol/server-everything sse
```
