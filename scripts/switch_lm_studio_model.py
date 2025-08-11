#!/usr/bin/env python3
"""
LM Studioのモデル切り替えを自動化するスクリプト
"""

import subprocess
import time
import requests
import json
from pathlib import Path

FINETUNED_MODEL_PATH = "/Users/matsbaccano/Projects/clone/mlx-finetuning/works/mlx_finetuning_1754901424/fused_model"
LM_STUDIO_API_BASE = "http://localhost:1234/v1"

def check_lm_studio_running():
    """LM Studio APIサーバーが起動しているかチェック"""
    try:
        response = requests.get(f"{LM_STUDIO_API_BASE}/models", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_current_models():
    """現在ロードされているモデル一覧を取得"""
    try:
        response = requests.get(f"{LM_STUDIO_API_BASE}/models")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def test_model_knowledge():
    """モデルの学習済み知識をテスト"""
    test_questions = [
        "あなたの所属している会社は？",
        "会社の設立年は？"
    ]
    
    results = {}
    for question in test_questions:
        try:
            response = requests.post(f"{LM_STUDIO_API_BASE}/chat/completions", 
                json={
                    "model": "fused_model",
                    "messages": [{"role": "user", "content": question}],
                    "temperature": 0.1,
                    "max_tokens": 100
                }, timeout=30)
            
            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"]
                results[question] = answer
            else:
                results[question] = f"エラー: {response.status_code}"
        except Exception as e:
            results[question] = f"エラー: {str(e)}"
    
    return results

def main():
    print("🔍 LM Studio モデル切り替えスクリプト")
    print("=" * 50)
    
    # LM Studio API確認
    if not check_lm_studio_running():
        print("❌ LM Studio APIサーバーが起動していません")
        print("LM Studioを開いて、ローカルサーバーを起動してください")
        return
    
    print("✅ LM Studio APIサーバーが起動中")
    
    # 現在のモデル確認
    current_models = get_current_models()
    if current_models:
        print(f"📋 現在のモデル: {[m['id'] for m in current_models['data']]}")
    
    # ファインチューニング済みモデルパス確認
    model_path = Path(FINETUNED_MODEL_PATH)
    if not model_path.exists():
        print(f"❌ ファインチューニング済みモデルが見つかりません: {FINETUNED_MODEL_PATH}")
        return
    
    print(f"✅ ファインチューニング済みモデル確認: {model_path}")
    
    # 現在のモデルの知識テスト
    print("\n🧪 現在のモデルの知識テスト:")
    results = test_model_knowledge()
    for question, answer in results.items():
        print(f"Q: {question}")
        print(f"A: {answer}")
        print()
    
    # ファインチューニング済み知識の判定
    has_company_knowledge = any("テックイノベーション" in str(answer) for answer in results.values())
    has_year_knowledge = any("2020" in str(answer) for answer in results.values())
    
    if has_company_knowledge and has_year_knowledge:
        print("✅ 現在のモデルは正しいファインチューニング済み知識を持っています")
        return
    else:
        print("❌ 現在のモデルはファインチューニング済み知識を持っていません")
    
    print("\n📝 手動でモデルを切り替えてください:")
    print("1. LM Studioを開く")
    print("2. 現在のモデルをアンロード")
    print(f"3. 以下のパスからモデルをロード:")
    print(f"   {FINETUNED_MODEL_PATH}")
    print("4. モデル名を 'fused_model' に設定")
    print("5. このスクリプトを再実行して確認")

if __name__ == "__main__":
    main()