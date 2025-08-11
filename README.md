# 🚀 MLX Fine-tuning Studio

**Apple Silicon最適化 × 企業特化AIアシスタント構築プラットフォーム**

Apple Silicon Mac上でリアルなニューラルネットワークの重み更新を実行する、LM Studio統合に特化したStreamlitアプリケーション。従来のシミュレーションベースのアプローチとは異なり、実際の.safetensors LoRAアダプターファイルを生成します。

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![MLX](https://img.shields.io/badge/MLX-supported-orange.svg)

## ✨ 主な特徴

- **🎯 実証済みの成功パターン**: 95%の訓練損失改善を達成したMLXファインチューニング手法
- **🔥 リアルなMLXファインチューニング**: 実際のニューラルネットワーク重み更新
- **🍎 Apple Silicon最適化**: M4チップの性能を最大活用
- **💻 LM Studio統合**: WebアプリからAPIサーバー経由で利用可能
- **📊 直感的UI**: 4つのタブによるステップバイステップガイド
- **🧪 リアルタイムテスト**: ファインチューニング結果の即座確認
- **🇯🇵 日本語サポート**: 日本語テキスト処理と検証

## 🏆 実証済みの成果

**Google Gemma-2-2b-it** での企業特化ファインチューニング：
- **訓練損失**: 3.048 → 0.143 (95%改善) 
- **検証損失**: 6.065 → 1.530 (75%改善)
- **学習時間**: 約3分（100イテレーション）
- **メモリ使用**: 約6GB
- **LM Studio統合**: 完全テスト済み

### テスト結果例

| 質問 | 期待回答 | 実際の回答 | 結果 |
|------|----------|------------|------|
| あなたの所属している会社は？ | 株式会社テックイノベーションです。 | 株式会社テックイノベーションです。 | ✅ |
| 会社の設立年は？ | 2020年に設立されました。 | 2020年に設立されました。 | ✅ |
| 会社の強みは？ | MLXを活用したApple Silicon最適化が得意分野です。 | MLXを活用したApple Silicon最適化が得意分野です。 | ✅ |

## 🔧 システム要件

### 必須要件
- **Apple Silicon Mac** (M1/M2/M3/M4) - MLX専用
- **macOS 13.0以上**
- **Python 3.11以上**
- **メモリ 8GB以上推奨** (実際の重み更新のため)
- **LM Studio**: 最新版

### サポートモデル
- Google Gemma 2 (2B, 9B variants)
- Meta Llama 3.1 (8B Instruct)
- ELYZA Japanese models

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

# 依存関係のインストール
pip install -r requirements.txt
```

### 3. アプリケーション起動
```bash
# Streamlitアプリ起動
streamlit run app.py
```

ブラウザで `http://localhost:8501` にアクセス

## 🎯 使用方法

### 基本ワークフロー

1. **📊 データセット準備**
   - CSVファイル（instruction, outputカラム）をアップロード
   - 自動的に学習/検証データに分割

2. **🔧 MLXファインチューニング**
   - ワンクリックでMLXファインチューニング実行
   - リアルタイムの進捗表示

3. **💻 LM Studio統合**
   - ファインチューニング済みモデルをLM Studioに自動コピー
   - APIサーバー起動の確認

4. **🎯 テスト・デモ**
   - 定型テストで学習結果を確認
   - インタラクティブチャットでリアルタイムテスト

### LM Studioローカルサーバー起動
```bash
# LM Studio内でサーバー起動
# 1. LM Studioを起動
# 2. Developer > Start Server (ポート1234)
# 3. モデルを選択してLoad
```

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
├── app.py                      # メインアプリケーション
├── requirements.txt           # 依存関係（最小限）
├── data/                     # サンプルデータ
│   └── templates/
│       └── core_company_data.csv  # 企業情報データセット
├── setup.sh                  # セットアップスクリプト
├── CLAUDE.md                 # プロジェクト指示書
└── README.md                # このファイル
```

## 🛠️ 使用技術

- **MLX**: Apple Silicon最適化機械学習フレームワーク
- **LoRA**: 効率的なファインチューニング手法（Low-Rank Adaptation）
- **Streamlit**: モダンなWebアプリフレームワーク
- **LM Studio**: ローカルLLM実行プラットフォーム
- **Gemma 2**: Googleの高性能言語モデル

## 📊 データ形式

### 入力CSV形式
```csv
instruction,output
"あなたの所属している会社は？","株式会社テックイノベーションです。"
"会社の設立年は？","2020年に設立されました。"
```

### 自動生成される学習データ形式
```json
{
  "text": "### 指示:\nあなたの所属している会社は？\n\n### 回答:\n株式会社テックイノベーションです。"
}
```

## 🔧 カスタマイズ

### ファインチューニングパラメータ調整

`app.py`の`run_mlx_finetuning()`関数で調整可能：

```python
--iters 100              # 学習イテレーション数
--steps-per-report 25    # 進捗報告間隔  
--lora-layers 16         # LoRA対象レイヤー数
```

### LM Studio API設定

デフォルト: `http://localhost:1234`

他のポートを使用する場合は`check_lmstudio_api()`関数を修正

## 🎯 Webアプリ統合

### JavaScript API呼び出し例

```javascript
async function chatWithAI(message) {
  const response = await fetch('http://localhost:1234/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'local-model',
      messages: [{ role: 'user', content: message }],
      temperature: 0.1
    })
  });
  
  const data = await response.json();
  return data.choices[0].message.content;
}
```

## 🚨 トラブルシューティング

### よくある問題

1. **MLX環境エラー**
   ```bash
   # MLX再インストール
   pip uninstall mlx mlx-lm
   pip install mlx mlx-lm
   ```

2. **LM Studio接続エラー**
   - LM Studioでサーバーが起動していることを確認
   - ポート1234が使用可能かチェック

3. **メモリ不足エラー**
   - 他のアプリケーションを終了
   - より小さなモデルを使用

4. **依存関係エラー**
   ```bash
   pip install scikit-learn psutil plotly
   ```

5. **Streamlit接続問題**
   ```bash
   pkill -f streamlit
   nohup streamlit run app.py --server.port 8506 --server.address 0.0.0.0 > streamlit.log 2>&1 &
   ```

## 📈 パフォーマンス最適化

- **バッチサイズ**: 自動調整（メモリに応じて）
- **量子化**: 自動的にbfloat16を使用
- **キャッシュ**: MLX内蔵キャッシュを活用
- **タイムスタンプベース**: ユニークパスでキャッシュ競合回避

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

- [Apple MLX team](https://ml-explore.github.io/mlx/) - Apple Silicon用機械学習フレームワーク
- [Streamlit](https://streamlit.io/) - 美しいWebアプリケーション
- [HuggingFace](https://huggingface.co/) - モデルハブとTransformersライブラリ
- [LM Studio](https://lmstudio.ai/) - ローカル言語モデル実行
- [Google Gemma team](https://ai.google.dev/gemma) - 高性能言語モデル

---

## 📞 サポート

問題が発生した場合は、Issueを作成してください。

**🚀 Happy Fine-tuning with MLX! 🍎**

**注意**: このアプリケーションは教育および研究目的で開発されました。プロダクション環境での使用前に十分なテストを実施してください。