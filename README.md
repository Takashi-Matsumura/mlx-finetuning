# MLX Fine-tuning

Apple Silicon Mac専用のリアルなMLXファインチューニングアプリケーション

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![MLX](https://img.shields.io/badge/MLX-supported-orange.svg)

## 📋 概要

**MLX Fine-tuning**は、Apple Silicon Mac上でリアルなニューラルネットワークの重み更新を実行するStreamlitベースのWebアプリケーションです。従来のシミュレーションベースのアプローチとは異なり、このアプリはMLXフレームワークを使用して実際の.safetensors LoRAアダプターファイルを生成します。

### ✨ 主要機能

- 🔥 **リアルなMLXファインチューニング**: 実際のニューラルネットワーク重み更新
- 📊 **Streamlit Webインターフェース**: 7つのメインページを持つ直感的なUI
- 🎯 **LoRA適応**: 効率的な学習のためのLow-Rank Adaptation
- 🔄 **完全なパイプライン**: データ前処理 → ファインチューニング → 量子化 → Ollama統合
- 🇯🇵 **日本語サポート**: 日本語テキスト処理と検証
- 📈 **実験追跡**: 包括的なメトリクス可視化と実験履歴
- 💾 **リアルタイムメモリ監視**: バッチサイズ推奨とパフォーマンス最適化

### 🏆 実証済みの成果

- **Google Gemma-2-2b-it**: 48イテレーション、26.2秒、損失 2.785→0.324
- **企業情報注入**: 33.5秒、損失 0.5、知識転移成功
- **Ollama統合**: 両モデルとも正常に統合・テスト完了

## 🔧 システム要件

### 必須要件
- **Apple Silicon Mac** (M1/M2/M3/M4) - MLX専用
- **macOS 13.0以上**
- **Python 3.11以上**
- **メモリ 16GB以上推奨** (実際の重み更新のため)

### サポートモデル
- ELYZA Japanese models (Llama-3-ELYZA-JP-8B)
- Google Gemma 2 (2B, 9B variants)
- Meta Llama 3.1 (8B Instruct)

## 🚀 クイックスタート

### 1. リポジトリのクローン
```bash
git clone https://github.com/your-username/mlx-finetuning.git
cd mlx-finetuning
```

### 2. MLX環境の設定
```bash
# 仮想環境作成
python3 -m venv mlx_env
source mlx_env/bin/activate

# MLX依存関係のインストール（Apple Silicon専用）
pip install mlx mlx-lm

# アプリケーション依存関係のインストール
pip install streamlit pandas numpy transformers
pip install scikit-learn psutil plotly torch pyyaml jinja2
```

### 3. アプリケーション起動
```bash
# MLXモード（リアルファインチューニング）
source mlx_env/bin/activate
streamlit run app.py --server.port 8506 --server.address 0.0.0.0
```

ブラウザで `http://localhost:8506` にアクセス

## 🎯 使用方法

### 基本ワークフロー
1. **データセット準備**: CSV/JSONファイルをアップロードして前処理
2. **モデル設定**: ベースモデルとLoRAパラメータを選択
3. **ファインチューニング実行**: リアルなMLX学習を開始
4. **結果確認**: 学習メトリクスと生成されたLoRAアダプターを確認
5. **量子化とデプロイ**: GGUF形式に変換してOllamaに統合

### HuggingFace認証（ゲートモデル用）
```bash
# HuggingFace Token設定
export HUGGINGFACE_TOKEN="your-token-here"
# または
export HF_TOKEN="your-token-here"
```

## 📁 プロジェクト構造

```
mlx-finetuning/
├── app.py                    # メインアプリケーション
├── src/                      # コアソースコード
│   ├── data_processor.py     # データ前処理・検証
│   ├── mlx_trainer.py        # MLXファインチューニング（コア）
│   ├── trainer.py            # トレーニング統合
│   ├── quantizer.py          # GGUF量子化
│   ├── ollama_integration.py # Ollama統合
│   ├── experiment_tracker.py # 実験追跡
│   └── utils/                # ユーティリティ
├── config/                   # 設定ファイル
│   ├── default_config.yaml   # デフォルト設定
│   └── models.yaml          # サポートモデル定義
├── tests/                    # テストファイル
├── data/
│   └── templates/           # データテンプレート
├── sample_data.csv          # サンプルデータ
└── requirements.txt         # 依存関係
```

## 🔬 技術的詳細

### MLXファインチューニング実装
- **コアメソッド**: `python -m mlx_lm lora`による安定化
- **ユニークパス**: タイムスタンプベースでキャッシュ競合を回避
- **HuggingFace認証**: ゲートモデル対応
- **.safetensorsファイル生成**: 実際の訓練済み重み

### 量子化ワークフロー
- **プライマリ**: llama.cpp GGUF変換
- **フォールバック**: MLX変換未対応時のPyTorch
- **フォーマット**: Q4_K_M, Q5_K_M, Q8_0 自動最適化

### 実験追跡
- **ユニークID**: タイムスタンプベース実験管理
- **リアルタイムメトリクス**: 損失、困惑度、継続時間
- **JSON永続化**: 完全な実験履歴と比較

## 🚨 トラブルシューティング

### 一般的な問題

**依存関係エラー**:
```bash
pip install scikit-learn psutil plotly
```

**Streamlit接続問題**:
```bash
pkill -f streamlit
nohup streamlit run app.py --server.port 8506 --server.address 0.0.0.0 > streamlit.log 2>&1 &
```

**MLXモデルキャッシュ競合**:
- タイムスタンプベースユニークディレクトリで解決済み

## 🧪 テスト

```bash
# 全テスト実行
pytest

# 特定テスト実行
pytest tests/test_data_processor.py

# カバレッジ付き実行
pytest --cov=src tests/
```

## 📊 開発ツール

```bash
# コードフォーマット
black . && isort .

# タイプチェック
mypy .

# リンティング
flake8
```

## 🤝 コントリビューション

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. Pull Requestを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## ✨ アクノレッジメント

- [MLX](https://ml-explore.github.io/mlx/) - Apple Silicon用機械学習フレームワーク
- [Streamlit](https://streamlit.io/) - 美しいWebアプリケーション
- [HuggingFace](https://huggingface.co/) - モデルハブとTransformersライブラリ
- [Ollama](https://ollama.com/) - ローカル言語モデル実行

---

**注意**: このアプリケーションは教育および研究目的で開発されました。プロダクション環境での使用前に十分なテストを実施してください。