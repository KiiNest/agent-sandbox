# Chatbot.py - シンプルAIチャットボット

## 概要
`chatbot.py`は、最もシンプルなAIチャットボットの実装です。外部ツールを使用せず、純粋にLLMとの対話のみを行います。LangGraphを使用してステートフルな会話を管理します。

## アーキテクチャ

### クラス構成
- **State**: メッセージ履歴を管理するTypeDict
- **Chatbot**: メインのチャットボットクラス

### グラフ構造
```
START → chatbot → END
```

## 処理フロー

### 1. 初期化フェーズ
```python
def __init__(self):
    self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1, streaming=True)
```

**処理ステップ:**
1. OpenAI GPT-4oモデルの初期化
2. 温度設定0.1（決定論的な応答）
3. ストリーミング有効化

### 2. グラフ構築フェーズ
```python
def build(self):
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", self.chatbot)
    graph_builder.add_edge(START, "chatbot")
    return graph_builder.compile()
```

**処理ステップ:**
1. StateGraphインスタンスの作成
2. chatbotノードの追加
3. START → chatbot エッジの定義
4. グラフのコンパイル

### 3. 実行時処理フロー

#### 3.1 ユーザー入力受信
- ユーザーの質問がStateの`messages`配列に追加
- STARTノードからchatbotノードへ直接遷移

#### 3.2 Chatbotノード処理
```python
def chatbot(self, state: State):
    return {"messages": [self.llm.invoke(state["messages"])]}
```

**処理ステップ:**
1. 現在のメッセージ履歴をLLMに送信
2. LLMが内容を分析して応答を生成
3. 生成された応答をStateに追加
4. 処理終了（自動的にEND）

#### 3.3 応答出力
```python
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
```

**処理ステップ:**
1. グラフをストリーミングモードで実行
2. イベントを順次処理
3. 最新メッセージの内容を出力

## 実際の使用例

### 質問: 「LangGraphについて教えて」

1. **START** → `{"messages": [{"role": "user", "content": "LangGraphについて教えて"}]}`

2. **chatbot** → LLMが知識ベースから回答生成
   - 出力: `{"messages": [AIMessage(content="LangGraphは...")]}`

3. **END** → 処理完了

## 主要機能

### ストリーミング出力
- リアルタイムでの応答表示
- ユーザーエクスペリエンスの向上

### インタラクティブループ
```python
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # フォールバック処理
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
```

**処理ステップ:**
1. ユーザー入力の待機
2. 終了コマンドの検出
3. 入力が利用できない場合のフォールバック
4. デフォルト質問での動作テスト

## 技術的特徴

1. **最小構成**: 必要最小限の機能のみ実装
2. **高速応答**: 外部ツール不使用による高速処理
3. **ストリーミング**: リアルタイム応答表示
4. **エラー処理**: 入力エラー時のフォールバック
5. **拡張可能**: 他の実装のベースとして使用可能

## 制限事項

1. **外部情報アクセス不可**: 最新情報や検索機能なし
2. **ツール使用不可**: 計算や外部API呼び出し不可
3. **メモリ非永続化**: 会話履歴は実行中のみ保持

## 適用場面

- **プロトタイピング**: 新機能の基盤として
- **教育目的**: LangGraphの基本概念学習
- **軽量な対話**: 簡単な質問応答システム
- **テスト用途**: 基本的な動作確認