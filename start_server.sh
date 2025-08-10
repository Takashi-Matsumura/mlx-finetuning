#!/bin/bash

# MLX Fine-tuning Server Startup Script
# MLXファインチューニングサーバー起動スクリプト

echo "🚀 Starting MLX Fine-tuning Server..."
echo "MLXファインチューニングサーバーを起動中..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "mlx_env" ]; then
    echo "❌ Virtual environment 'mlx_env' not found!"
    echo "Please run: python3 -m venv mlx_env && source mlx_env/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source mlx_env/bin/activate

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import streamlit, mlx, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies are missing. Installing..."
    pip install -r requirements.txt
    pip install mlx mlx-lm
fi

# Check if port is already in use
PORT=8507
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port $PORT is already in use. Trying to stop existing process..."
    pkill -f "streamlit.*app.py"
    sleep 2
fi

# Start server
echo "🌐 Starting server on http://localhost:$PORT"
echo "サーバーを起動中: http://localhost:$PORT"

streamlit run app.py --server.port $PORT --server.address 0.0.0.0

echo "👋 Server stopped."