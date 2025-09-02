# -*- coding: utf-8 -*-
from dotenv import load_dotenv
import os
import json
import re
import requests
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from bs4 import BeautifulSoup


# =========================
# State
# =========================
class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: Optional[str]             # 取得した本文の束
    sources: Optional[List[str]]       # 参照URL一覧


# =========================
# Constants / Utils
# =========================
# Tavily 検索向けの「サイト限定」クエリ（URLではなく検索フィルタ文字列）
SITE_FILTER = 'https://learn.microsoft.com/ja-jp/dotnet/maui/?view=net-maui-9.0#-net-maui----'

SYSTEM_PROMPT = """あなたは .NET MAUI 9.0 (ja-jp) ドキュメントのQ&Aアシスタントです。
必ず以下を守ってください：
- 対象は learn.microsoft.com の .NET MAUI 9.0 (ja-jp) のみ。
- 不明点は推測せず、根拠のある部分のみ回答。
- 回答の末尾に "参考URL:" として使用したURLを列挙（最大5件）。
- コードは必要最小限。日本語で簡潔に。"""

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAG-MAUI/1.0; +https://example.org)"
}

def extract_main_text(html: str) -> str:
    """<main> を優先し、余分なナビ/フッターを排除してテキスト抽出（簡易版）"""
    soup = BeautifulSoup(html, "html.parser")

    # remove boilerplate
    for sel in ["nav", "header", "footer", "aside", ".sidebar", ".navigation", ".toc"]:
        for tag in soup.select(sel):
            tag.decompose()

    main = soup.find("main") or soup
    for tag in main(["script", "style"]):
        tag.decompose()

    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:18000]  # 安全のため上限

def fetch_urls(urls: List[str], limit: int = 3) -> Dict[str, str]:
    """上位URLを取得→本文抽出。{url: text} を返す"""
    out = {}
    for url in urls[:limit]:
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200 and r.text:
                out[url] = extract_main_text(r.text)
        except Exception:
            continue
    return out

def build_context_text(pages: Dict[str, str]) -> str:
    """LLMへ渡すコンテキスト本文を作成"""
    chunks = []
    for url, body in pages.items():
        chunks.append(f"[SOURCE] {url}\n{body}\n")
    return "\n\n".join(chunks)[:45000]


# =========================
# Tool node
# =========================
class BasicToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No messages found in inputs")

        outputs = []
        for tool_call in getattr(message, "tool_calls", []):
            tool = self.tools_by_name[tool_call["name"]]
            tool_result = tool.invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


# =========================
# Agent
# =========================
class Agent:
    def __init__(self):
        # OpenAI チャットモデル
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1, streaming=True)

        # Tavily 検索インスタンス
        self._tavily = TavilySearch(max_results=5)

        # Callable な LangChain Tool を @tool で定義（bind_tools 対応）
        @tool("maui_search")
        def maui_search(query: str) -> dict:
            """Search .NET MAUI 9.0 (ja-jp) docs on learn.microsoft.com with site restriction."""
            q = f"{query} {SITE_FILTER}".strip()
            return self._tavily.invoke({"query": q})

        self.tools = [maui_search]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def build(self):
        graph_builder = StateGraph(State)

        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.add_node("tools", BasicToolNode(tools=self.tools))
        graph_builder.add_node("fetch_and_clean", self.fetch_and_clean)

        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_conditional_edges("chatbot", self.route_tools, {"tools": "tools", END: END})
        graph_builder.add_edge("tools", "fetch_and_clean")
        graph_builder.add_edge("fetch_and_clean", "chatbot")
        return graph_builder.compile()

    # ---- nodes ----
    def chatbot(self, state: State):
        messages = state["messages"]
        # system を先頭に挿入
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        # 取得済みコンテキストがあれば、補助入力として渡す
        context = state.get("context")
        sources = state.get("sources") or []
        if context:
            context_note = (
                "以下は検索で収集した参照本文です。必要な部分のみ使って回答してください。\n\n"
                f"{context}\n\n"
                f"(参照URL候補: {', '.join(sources[:5])})"
            )
            messages = messages + [{"role": "user", "content": context_note}]

        return {"messages": [self.llm_with_tools.invoke(messages)]}

    def fetch_and_clean(self, state: State):
        """検索結果のURLを取得→本文抽出→stateに格納"""
        tool_msg = state["messages"][-1]
        try:
            payload = json.loads(tool_msg.content)
        except Exception:
            payload = {}

        urls = []
        for r in payload.get("results", []):
            url = r.get("url")
            if not url:
                continue
            if "learn.microsoft.com" in url:
                # 念のためフィルタ（9.0/ja-jp優先）
                if "/ja-jp/" in url and ("view=net-maui-9.0" in url or "net-maui-9.0" in url):
                    urls.append(url)
                else:
                    urls.append(url)  # Tavily の要約が9.0起点のことがあるため残す

        urls = list(dict.fromkeys(urls))  # 重複排除
        pages = fetch_urls(urls, limit=3)
        context_text = build_context_text(pages)
        return {"context": context_text, "sources": urls}

    # ---- router ----
    def route_tools(self, state: State):
        messages = state["messages"] if isinstance(state, dict) else state
        ai_message = messages[-1] if messages else None
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return END


# =========================
# CLI Runner
# =========================
def stream_graph_updates(user_input: str):
    agent = Agent()
    graph = agent.build()

    init = {"messages": [{"role": "user", "content": user_input}], "context": None, "sources": None}

    for event in graph.stream(init):
        for value in event.values():
            if "messages" in value and value["messages"]:
                message = value["messages"][-1]
                # ツール応答や中間はスキップ。最終LLM出力だけ表示
                if hasattr(message, "content") and message.content and not hasattr(message, "tool_call_id"):
                    print("\nAssistant:\n", message.content)


# =========================
# Streamlit Runner
# =========================
def run_streamlit():
    import streamlit as st

    load_dotenv()
    st.set_page_config(page_title="MAUI RAG (search-only)", page_icon="🟣", layout="wide")
    st.title("💻 .NET MAUI 9.0 ドキュメント Q&A")

    # --- Sidebar ---
    with st.sidebar:
        # st.markdown("### 設定")
        # st.caption("環境変数の読み込み: `.env` の OPENAI_API_KEY / TAVILY_API_KEY")
        clear = st.button("履歴をクリア")

    # --- Session State 初期化 ---
    if "agent" not in st.session_state:
        st.session_state.agent = Agent()
        st.session_state.graph = st.session_state.agent.build()
        st.session_state.history: List[Dict[str, str]] = []

    if clear:
        st.session_state.history = []
        st.rerun()

    # --- 既存履歴の描画 ---
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- ユーザー入力 & 実行 ---
    user_prompt = st.chat_input("MAUI について質問してください")
    if user_prompt:
        st.session_state.history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        placeholder = st.chat_message("assistant").empty()
        acc_text = ""

        init_state = {"messages": [{"role": "user", "content": user_prompt}], "context": None, "sources": None}

        # LangGraph のイベントを逐次反映（簡易ストリーミング）
        for event in st.session_state.graph.stream(init_state):
            for value in event.values():
                if "messages" in value and value["messages"]:
                    message = value["messages"][-1]
                    if hasattr(message, "content") and message.content and not hasattr(message, "tool_call_id"):
                        acc_text = message.content
                        placeholder.markdown(acc_text)

        # 応答を履歴に保存
        st.session_state.history.append({"role": "assistant", "content": acc_text})

    # フッター
    st.caption("ソースは MS Learn / .NET MAUI 9.0（ja-jp）に限定して検索・抽出しています。")


# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    load_dotenv()
    import sys

    # --- 今が「Streamlitランタイム配下」かを堅牢に判定 ---
    under_streamlit = False
    try:
        # 公式API。ランタイムが初期化済みなら True
        from streamlit.runtime import exists as _st_exists
        under_streamlit = _st_exists()
    except Exception:
        under_streamlit = False

    # 環境変数でのヒント（pdm scriptsで UI=streamlit を付けているなら True 扱い）
    if os.environ.get("UI", "").lower() == "streamlit":
        under_streamlit = True

    # --- 分岐 ---
    if under_streamlit:
        # すでに streamlit run から呼ばれているので、UI関数だけ実行（★二重起動禁止）
        run_streamlit()
        sys.exit(0)

    # ここは通常実行（EXE/CLI）。自己起動で Streamlit を立ち上げる
    try:
        from streamlit.web import cli as stcli

        # 再入ループ防止フラグ（streamlit がスクリプトを再実行するときに使う）
        os.environ["APP_LAUNCHED_BY_WRAPPER"] = "1"

        script_path = os.path.abspath(__file__)
        sys.argv = [
            "streamlit", "run", script_path,
            "--server.headless=true",
            "--server.port=8501",
        ]
        stcli.main()  # ← ブロッキング（^Cで KeyboardInterrupt）

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
