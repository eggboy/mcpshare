# mcpshare

コーディングエージェント間で MCP 設定を共有するための CLI ツールです。

[Model Context Protocol](https://modelcontextprotocol.io/) サーバー定義を
**VSCode**、**GitHub Copilot CLI**、**Claude Code**、**OpenAI Codex**、
**Google Gemini CLI**、**OpenCode** 間で同期します。

## クイックスタート

### `uv` を使う場合（推奨）

```bash
# PEP 723 のインラインメタデータで直接実行
uv run mcpshare.py init   # デフォルト設定を作成
uv run mcpshare.py pull   # 各ターゲットから master にサーバーを収集
uv run mcpshare.py sync   # master をすべてのターゲットへ配布
uv run mcpshare.py status # 現在の状態を表示
```

### `uv tool install` を使う場合

```bash
# グローバル CLI ツールとしてインストール
uv tool install git+https://github.com/eggboy/mcpshare.git

# 以降はどこからでも実行可能
mcpshare init
mcpshare pull
mcpshare sync
mcpshare status

# 最新バージョンへ更新
uv tool upgrade mcpshare
```

## 設定

設定ファイルは `~/.config/mcpshare/config.yaml` にあります:

```yaml
source: /path/to/master          # 正本 mcp.json を置くディレクトリ
mode: merge                      # merge | overwrite
targets:
  claude:
    path: ~/.claude
  vscode:
    path: ~/Library/Application Support/Code/User  # macOS。その他 OS は docs を参照
  copilot:
    path: ~/.copilot
  codex:
    path: ~/.codex
  gemini:
    path: ~/.gemini
  opencode:
    path: ~/.config/opencode
```

### モード

| モード | 挙動 |
|------|-----------|
| `merge` | まずすべてのターゲットからサーバーを収集して master にマージし、その後配布 |
| `overwrite` | master を唯一の正とし、すべてのターゲットを上書き |

## 動作概要

1. **`mcpshare init`** – 設定ファイルと master ディレクトリを作成します。
2. **`mcpshare pull`** – 設定された各ターゲットから MCP サーバー定義を取得し、
   master の `mcp.json` にマージします（収集のみで、ターゲットへは書き戻しません）。
3. **`mcpshare sync`** – master の `mcp.json` を設定済みの全ターゲットに配布し、
   各ツールのネイティブ形式へ変換します。
4. **`mcpshare status`** – master のサーバー一覧と、各ターゲット設定ファイルの存在有無を表示します。

典型的なワークフロー: `mcpshare pull` → `mcpshare sync`。

### サーバーの無効化

master から削除せずに、個別の MCP サーバーを無効化できます:

```bash
mcpshare disable <server>   # サーバーを無効としてマーク
mcpshare enable <server>    # 無効化したサーバーを再有効化
mcpshare status             # 無効なサーバーは [disabled] と表示
```

`mcpshare sync` 時の無効化サーバーの扱いはターゲットごとに異なります:

| ターゲット | 挙動 |
|--------|-----------|
| Copilot CLI | `"disabled": true` を書き込み — Copilot はこのサーバーをネイティブにスキップ |
| Claude Code | `"disabled": true` を書き込み — Claude は未知のフィールドを無視。スキップには外部で `--disallowedTools` または `--strict-mcp-config` を利用 |
| VSCode | `disabled` フィールドを除去（有効/無効は VSCode 側 UI で管理） |
| Codex | `enabled = false` を書き込み — サーバーは表示されるがツールは読み込まれない |
| Gemini, OpenCode | `disabled` フィールドを除去 |

### 対応フォーマット

| ツール | ファイル | 最上位キー | 備考 |
|------|------|---------------|-------|
| Claude Code | `mcp_servers.json` | `mcpServers` | |
| VSCode | `mcp.json` | `servers` | `"type": "stdio"` を追加 |
| Copilot CLI | `mcp-config.json` | `mcpServers` | `"type": "stdio"` を追加 |
| Codex | `config.toml` | `[mcp_servers.*]` | TOML 形式 |
| Gemini CLI | `settings.json` | `mcpServers` | MCP 以外の設定を保持 |
| OpenCode | `opencode.json` | `mcp` | `command` 配列、`environment` を使用 |

## 開発

```bash
pip install -e .
pip install pytest
python -m pytest tests/ -v
```
