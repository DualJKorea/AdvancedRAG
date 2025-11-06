# 단순 예시: 각 LLM + 각 Tool 연결 코드를 직접 작성

# ---- Slack API 연결 ----
def query_slack_messages(token, channel):
    # 실제로는 slack_sdk 사용
    return f"[Slack:{channel}] 최근 메시지"

# ---- Google Drive API 연결 ----
def list_drive_files(token, folder_id):
    # 실제로는 google-api-python-client 사용
    return f"[Drive:{folder_id}] 파일 목록"

# ---- OpenAI LLM ----
def ask_openai(prompt):
    # openai.ChatCompletion.create(...) 가정
    return f"[OpenAI 응답: {prompt}]"

# ---- Anthropic LLM ----
def ask_anthropic(prompt):
    # anthropic.Client().messages.create(...) 가정
    return f"[Anthropic 응답: {prompt}]"

# ---- NxM 문제 발생: LLM × Tool 조합 직접 연결 ----
def openai_with_slack():
    msgs = query_slack_messages("slack-token", "general")
    return ask_openai(f"슬랙 메시지 요약: {msgs}")

def openai_with_drive():
    files = list_drive_files("drive-token", "root")
    return ask_openai(f"드라이브 파일 요약: {files}")

def anthropic_with_slack():
    msgs = query_slack_messages("slack-token", "general")
    return ask_anthropic(f"슬랙 메시지 요약: {msgs}")

def anthropic_with_drive():
    files = list_drive_files("drive-token", "root")
    return ask_anthropic(f"드라이브 파일 요약: {files}")

if __name__ == "__main__":
    print(openai_with_slack())
    print(openai_with_drive())
    print(anthropic_with_slack())
    print(anthropic_with_drive())

