# Agent Sandbox Documentation

## 概要
Agent Sandboxは、LangGraphとLangChainを使用した様々なAIエージェント実装のサンプル集です。基本的なチャットボットから高度な人間支援機能まで、段階的に学習できる構成になっています。

## モジュール一覧

### 1. [chatbot.py](./chatbot.md) - シンプルAIチャットボット
**最も基本的な実装**
- 外部ツール不使用
- 純粋なLLMとの対話
- LangGraphの基礎概念学習に最適

### 2. [agent.py](./agent.md) - Web検索機能付きAIエージェント
**ツール使用の基本実装**
- Tavily検索ツール統合
- 条件分岐による動的ツール使用
- ステートフルな処理フロー

### 3. [memory_chatbot.py](./memory_chatbot.md) - メモリ機能付きAIチャットボット
**永続的会話履歴の実装**
- MemorySaverによる状態保存
- スレッドベースのセッション管理
- 継続的な対話体験

### 4. [human_assistance.py & human_assistance_pattern2.py](./human_assistance.md) - 人間支援機能付きAIエージェント
**Human-in-the-Loop実装**
- interrupt機能による実行中断
- 人間からの専門知識取得
- 2つの実装パターン（関数ベース・クラスベース）

## アーキテクチャ比較

| 機能 | chatbot | agent | memory_chatbot | human_assistance |
|------|---------|-------|----------------|------------------|
| 基本対話 | ✅ | ✅ | ✅ | ✅ |
| Web検索 | ❌ | ✅ | ✅ | ✅ |
| メモリ保存 | ❌ | ❌ | ✅ | ✅ |
| 人間支援 | ❌ | ❌ | ❌ | ✅ |
| 複雑度 | 低 | 中 | 中 | 高 |

## 学習順序の推奨

### 1. 初学者
1. **chatbot.py** - LangGraphの基本概念
2. **agent.py** - ツール統合とグラフフロー
3. **memory_chatbot.py** - 状態管理

### 2. 中級者
1. **human_assistance.py** - 高度な制御フロー
2. 各実装の詳細比較・カスタマイズ

## 共通技術要素

### LangGraph
- **StateGraph**: 状態管理グラフ
- **ToolNode**: ツール実行ノード
- **tools_condition**: 動的分岐
- **MemorySaver**: 状態永続化

### LangChain
- **ChatOpenAI**: GPT-4o統合
- **TavilySearch**: Web検索ツール
- **StructuredTool**: カスタムツール定義

### 設計パターン
- **条件分岐**: 動的な処理フロー制御
- **ツール統合**: 外部機能の組み込み
- **状態管理**: 会話履歴の保持
- **エラーハンドリング**: 堅牢な実行制御

## 実行環境セットアップ

### 必要な環境変数
```bash
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 依存関係
- Python 3.12+
- langchain-openai
- langchain-community
- langchain-tavily
- langgraph
- その他（pyproject.tomlを参照）

## 各実装の詳細ドキュメント

詳細な処理フローと実装解説は、各モジュールの専用ドキュメントを参照してください：

- [Chatbot詳細](./chatbot.md)
- [Agent詳細](./agent.md)
- [Memory Chatbot詳細](./memory_chatbot.md)
- [Human Assistance詳細](./human_assistance.md)

## 拡張可能性

### 新しいツールの追加
```python
# 例: 計算ツールの追加
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """数式を計算します"""
    return str(eval(expression))

# LLMにバインド
llm_with_tools = llm.bind_tools([existing_tools, calculator])
```

### カスタムノードの実装
```python
# 例: ログ記録ノード
def logging_node(state: State):
    print(f"Processing: {len(state['messages'])} messages")
    return {}

graph_builder.add_node("logger", logging_node)
```

### 永続化の強化
```python
# 例: ファイルベース永続化
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string("checkpoints.db")
```

## トラブルシューティング

### よくある問題
1. **API Key未設定**: 環境変数を確認
2. **メモリ不足**: 長時間実行時の履歴制限
3. **ツール呼び出しエラー**: ツール定義の確認

### デバッグ支援
- memory_chatbot.pyの状態可視化機能を活用
- pretty_print()でメッセージ構造を確認
- ログ出力による実行フロー追跡