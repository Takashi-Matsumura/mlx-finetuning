# 🚀 MLX Fine-tuning Server 起動ガイド

このドキュメントでは、MLXファインチューニングサーバーの起動・停止方法を説明します。

## 📋 クイックスタート

### **推奨方法（スクリプト使用）**

```bash
# サーバー起動
./start_server.sh

# サーバー停止
./stop_server.sh
```

### **手動起動**

```bash
cd /Users/matsbaccano/Projects/clone/mlx-finetuning
source mlx_env/bin/activate
streamlit run app.py --server.port 8507 --server.address 0.0.0.0
```

## 🌐 アクセス方法

- **ローカル**: http://localhost:8507
- **ネットワーク**: http://0.0.0.0:8507

## 🛠️ 起動スクリプトの機能

### **start_server.sh**
- ✅ 仮想環境の存在確認
- ✅ 依存関係の自動チェック・インストール
- ✅ ポート競合の自動解決
- ✅ サーバーの安全な起動

### **stop_server.sh**
- ✅ 実行中のサーバープロセス検出
- ✅ 安全な停止処理
- ✅ 強制停止のフォールバック
- ✅ ポート解放の確認

## 🚨 トラブルシューティング

### **仮想環境エラー**
```bash
python3 -m venv mlx_env
source mlx_env/bin/activate
pip install -r requirements.txt
pip install mlx mlx-lm
```

### **ポート競合エラー**
```bash
# 既存プロセス確認
lsof -i :8507

# 強制停止
./stop_server.sh

# 別ポートで起動
streamlit run app.py --server.port 8508
```

### **依存関係エラー**
```bash
source mlx_env/bin/activate
pip install --upgrade streamlit mlx mlx-lm pandas numpy transformers
```

## 🔧 カスタマイズ

### **ポート変更**
`start_server.sh` の `PORT=8507` を変更

### **ネットワーク設定**
- ローカルのみ: `--server.address localhost`
- 全ネットワーク: `--server.address 0.0.0.0`

### **バックグラウンド起動**
```bash
nohup ./start_server.sh > server.log 2>&1 &
```

## 📊 システム要件

- **OS**: macOS 13.0以上
- **CPU**: Apple Silicon (M1/M2/M3/M4)
- **メモリ**: 16GB以上推奨
- **Python**: 3.11以上
- **ディスク**: 20GB以上の空き容量

## 🆘 サポート

問題が発生した場合：

1. **ログ確認**: `server.log` または端末出力
2. **プロセス確認**: `ps aux | grep streamlit`
3. **ポート確認**: `lsof -i :8507`
4. **仮想環境確認**: `which python` (mlx_env内のPythonを指しているか)

## 🎯 使用例

```bash
# 開発時の起動
./start_server.sh

# 作業完了後の停止
./stop_server.sh

# 長時間稼働（バックグラウンド）
nohup ./start_server.sh > server.log 2>&1 &
tail -f server.log  # ログ監視
```

---

**Happy Fine-tuning! 🤖**