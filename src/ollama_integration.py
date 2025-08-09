import os
import subprocess
import json
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import time


class OllamaIntegrator:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.logger = logging.getLogger(__name__)
        
        # デフォルトのシステムプロンプトテンプレート
        self.system_prompt_templates = {
            'japanese_assistant': {
                'name': '日本語アシスタント',
                'prompt': '''あなたは親切で知識豊富な日本語アシスタントです。
ユーザーの質問に対して、正確で有益な回答を日本語で提供してください。
回答は分かりやすく簡潔にまとめ、必要に応じて具体例を示してください。'''
            },
            'code_assistant': {
                'name': 'コーディングアシスタント',
                'prompt': '''あなたは経験豊富なプログラマーです。
コードの問題解決、最適化、説明を得意としています。
回答にはコード例を含め、日本語でわかりやすく説明してください。'''
            },
            'translator': {
                'name': '翻訳アシスタント',
                'prompt': '''あなたは高品質な翻訳を提供する翻訳アシスタントです。
自然で読みやすい翻訳を心がけ、文脈を理解した適切な表現を使用してください。'''
            },
            'custom': {
                'name': 'カスタム',
                'prompt': ''
            }
        }
    
    def check_ollama_status(self) -> Dict[str, Any]:
        """Ollamaサーバーの状態をチェック"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                return {
                    'status': 'running',
                    'url': self.ollama_url,
                    'models_count': len(models),
                    'available_models': [m['name'] for m in models]
                }
            else:
                return {
                    'status': 'error',
                    'error': f'HTTP {response.status_code}'
                }
                
        except requests.exceptions.ConnectionError:
            return {
                'status': 'not_running',
                'error': 'Ollamaサーバーに接続できません'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def start_ollama_server(self) -> bool:
        """Ollamaサーバーを起動"""
        try:
            # Ollamaがインストールされているかチェック
            result = subprocess.run(['which', 'ollama'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error("Ollamaがインストールされていません")
                return False
            
            # Ollamaサーバーを起動
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # 起動を待つ
            for _ in range(30):  # 30秒待機
                status = self.check_ollama_status()
                if status['status'] == 'running':
                    self.logger.info("Ollamaサーバー起動完了")
                    return True
                time.sleep(1)
            
            self.logger.error("Ollamaサーバー起動タイムアウト")
            return False
            
        except Exception as e:
            self.logger.error(f"Ollamaサーバー起動エラー: {e}")
            return False
    
    def create_modelfile(
        self,
        base_model: str = "llama2",
        system_prompt: str = "",
        parameters: Dict[str, Any] = None
    ) -> str:
        """Modelfileを生成"""
        
        if parameters is None:
            parameters = {
                'temperature': 0.7,
                'top_p': 0.9,
                'num_ctx': 4096,
                'repeat_penalty': 1.1
            }
        
        modelfile_content = f"FROM {base_model}\n\n"
        
        # システムプロンプト
        if system_prompt:
            modelfile_content += f'SYSTEM """{system_prompt}"""\n\n'
        
        # パラメータ設定
        for param, value in parameters.items():
            modelfile_content += f"PARAMETER {param} {value}\n"
        
        return modelfile_content
    
    def create_model_from_gguf(
        self,
        gguf_path: str,
        model_name: str,
        system_prompt: str = "",
        parameters: Dict[str, Any] = None,
        template: str = None
    ) -> Dict[str, Any]:
        """GGUFファイルからOllamaモデルを作成"""
        
        # Ollamaサーバーの状態チェック
        status = self.check_ollama_status()
        if status['status'] != 'running':
            return {
                'success': False,
                'error': 'Ollamaサーバーが実行されていません'
            }
        
        try:
            # 一時的なModelfileを作成
            modelfile_content = f"FROM {gguf_path}\n\n"
            
            # システムプロンプト
            if system_prompt:
                modelfile_content += f'SYSTEM """{system_prompt}"""\n\n'
            
            # チャットテンプレート
            if template:
                modelfile_content += f'TEMPLATE """{template}"""\n\n'
            
            # パラメータ設定
            if parameters is None:
                parameters = {
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'num_ctx': 4096,
                    'repeat_penalty': 1.1
                }
            
            for param, value in parameters.items():
                modelfile_content += f"PARAMETER {param} {value}\n"
            
            # 一時ファイルに保存
            modelfile_path = f"/tmp/Modelfile_{model_name}"
            with open(modelfile_path, 'w', encoding='utf-8') as f:
                f.write(modelfile_content)
            
            self.logger.info(f"Modelfile作成: {modelfile_path}")
            self.logger.info(f"Modelfile内容:\n{modelfile_content}")
            
            # Ollamaでモデルを作成
            cmd = ['ollama', 'create', model_name, '-f', modelfile_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # 一時ファイルを削除
                os.remove(modelfile_path)
                
                return {
                    'success': True,
                    'model_name': model_name,
                    'message': 'モデル作成完了'
                }
            else:
                return {
                    'success': False,
                    'error': f'モデル作成エラー: {result.stderr}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_model(
        self,
        model_name: str,
        test_prompt: str = "こんにちは！調子はいかがですか？"
    ) -> Dict[str, Any]:
        """モデルをテスト"""
        
        try:
            # APIでテスト実行
            data = {
                'model': model_name,
                'prompt': test_prompt,
                'stream': False
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'prompt': test_prompt,
                    'response': result.get('response', ''),
                    'eval_count': result.get('eval_count', 0),
                    'eval_duration': result.get('eval_duration', 0)
                }
            else:
                return {
                    'success': False,
                    'error': f'テスト失敗: HTTP {response.status_code}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """利用可能なモデル一覧を取得"""
        
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
            else:
                self.logger.error(f"モデル一覧取得エラー: HTTP {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"モデル一覧取得エラー: {e}")
            return []
    
    def delete_model(self, model_name: str) -> Dict[str, Any]:
        """モデルを削除"""
        
        try:
            cmd = ['ollama', 'rm', model_name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'message': f'モデル {model_name} を削除しました'
                }
            else:
                return {
                    'success': False,
                    'error': f'削除エラー: {result.stderr}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """モデル情報を取得"""
        
        try:
            # モデル情報をAPIで取得
            data = {'name': model_name}
            response = requests.post(
                f"{self.ollama_url}/api/show",
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'error': f'モデル情報取得エラー: HTTP {response.status_code}'
                }
                
        except Exception as e:
            return {
                'error': str(e)
            }
    
    def optimize_parameters(
        self,
        model_name: str,
        use_case: str = 'general'
    ) -> Dict[str, Any]:
        """用途に応じたパラメータ最適化提案"""
        
        parameter_presets = {
            'creative': {
                'temperature': 0.9,
                'top_p': 0.95,
                'repeat_penalty': 1.05,
                'description': '創作や発想に適した設定'
            },
            'precise': {
                'temperature': 0.3,
                'top_p': 0.8,
                'repeat_penalty': 1.2,
                'description': '正確性を重視した設定'
            },
            'general': {
                'temperature': 0.7,
                'top_p': 0.9,
                'repeat_penalty': 1.1,
                'description': '一般的な用途向け設定'
            },
            'translation': {
                'temperature': 0.5,
                'top_p': 0.85,
                'repeat_penalty': 1.15,
                'description': '翻訳タスクに適した設定'
            },
            'coding': {
                'temperature': 0.4,
                'top_p': 0.8,
                'repeat_penalty': 1.1,
                'description': 'コーディング支援向け設定'
            }
        }
        
        preset = parameter_presets.get(use_case, parameter_presets['general'])
        
        return {
            'use_case': use_case,
            'recommended_parameters': preset,
            'available_presets': list(parameter_presets.keys())
        }
    
    def export_model_config(
        self,
        model_name: str,
        output_path: str
    ) -> Dict[str, Any]:
        """モデル設定をエクスポート"""
        
        try:
            # モデル情報を取得
            model_info = self.get_model_info(model_name)
            
            if 'error' in model_info:
                return {
                    'success': False,
                    'error': model_info['error']
                }
            
            # 設定ファイルとして保存
            config = {
                'model_name': model_name,
                'exported_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_info': model_info,
                'ollama_url': self.ollama_url
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            return {
                'success': True,
                'output_path': output_path,
                'message': '設定エクスポート完了'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_system_prompt_templates(self) -> Dict[str, Dict[str, str]]:
        """システムプロンプトテンプレートを取得"""
        return self.system_prompt_templates
    
    def validate_model_name(self, model_name: str) -> Dict[str, Any]:
        """モデル名の検証"""
        
        errors = []
        warnings = []
        
        # 文字チェック
        if not model_name:
            errors.append('モデル名が空です')
        elif len(model_name) < 2:
            errors.append('モデル名が短すぎます')
        elif len(model_name) > 50:
            errors.append('モデル名が長すぎます')
        
        # 文字種チェック
        if not model_name.replace('-', '').replace('_', '').replace('.', '').isalnum():
            warnings.append('英数字、ハイフン、アンダースコア、ドット以外の文字が含まれています')
        
        # 既存モデルとの重複チェック
        existing_models = self.list_models()
        existing_names = [m['name'] for m in existing_models]
        
        if model_name in existing_names:
            warnings.append('同名のモデルが既に存在します')
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }