#!/bin/bash

echo "🚀 LLMファインチューニングアプリのセットアップを開始します..."

# Apple Silicon Macのチェック
if [[ $(uname -m) != "arm64" ]]; then
    echo "❌ Error: Apple Silicon Mac が必要です"
    exit 1
fi

# macOSバージョンチェック
if [[ $(sw_vers -productVersion | cut -d. -f1) -lt 13 ]]; then
    echo "❌ Error: macOS 13.0以上が必要です"
    exit 1
fi

# Homebrewの確認
if ! command -v brew &> /dev/null; then
    echo "❌ Error: Homebrewがインストールされていません"
    echo "まず https://brew.sh からHomebrewをインストールしてください"
    exit 1
fi

# Condaの確認とインストール
if ! command -v conda &> /dev/null; then
    echo "📦 Minicondaをインストール中..."
    brew install miniconda
    echo 'eval "$(conda shell.bash hook)"' >> ~/.bashrc
    source ~/.bashrc
fi

# Python環境の作成
echo "🐍 Python環境を作成中..."
conda create -n llm-finetuning python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate llm-finetuning

# 依存関係のインストール
echo "📚 依存関係をインストール中..."
pip install --upgrade pip
pip install -r requirements.txt

# llama.cppのビルド
echo "🔨 llama.cppをビルド中..."
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
fi
cd llama.cpp
LLAMA_METAL=1 make -j$(sysctl -n hw.ncpu)
cd ..

# Ollamaのインストール
echo "🤖 Ollamaをインストール中..."
if ! command -v ollama &> /dev/null; then
    brew install ollama
fi

# ディレクトリ権限の設定
chmod -R 755 .

echo "✅ セットアップが完了しました！"
echo ""
echo "🚀 アプリを起動するには以下のコマンドを実行してください："
echo "conda activate llm-finetuning"
echo "streamlit run app.py"
echo ""
echo "📝 詳細な使用方法はREADME.mdを参照してください"