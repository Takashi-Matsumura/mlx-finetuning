#!/bin/bash

# MLX Fine-tuning Server Stop Script
# MLXファインチューニングサーバー停止スクリプト

echo "⏹️  Stopping MLX Fine-tuning Server..."
echo "MLXファインチューニングサーバーを停止中..."

# Kill all streamlit processes related to this app
PROCESSES=$(ps aux | grep "streamlit.*app.py" | grep -v grep | awk '{print $2}')

if [ -z "$PROCESSES" ]; then
    echo "ℹ️  No running server found."
    echo "実行中のサーバーが見つかりません。"
else
    echo "🔍 Found running processes: $PROCESSES"
    pkill -f "streamlit.*app.py"
    
    # Wait and verify
    sleep 2
    REMAINING=$(ps aux | grep "streamlit.*app.py" | grep -v grep | wc -l)
    
    if [ "$REMAINING" -eq 0 ]; then
        echo "✅ Server stopped successfully."
        echo "サーバーを正常に停止しました。"
    else
        echo "⚠️  Force killing remaining processes..."
        pkill -9 -f "streamlit.*app.py"
        echo "✅ Server force stopped."
        echo "サーバーを強制停止しました。"
    fi
fi

# Check port status
PORT=8507
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port $PORT is still in use by another process."
else
    echo "✅ Port $PORT is now free."
    echo "ポート$PORTが解放されました。"
fi

echo "👋 Stop script completed."