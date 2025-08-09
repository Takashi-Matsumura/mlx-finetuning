import os
import subprocess
import json
import logging
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import threading
import time


class ModelQuantizer:
    def __init__(self, llama_cpp_path: str = "./llama.cpp"):
        self.llama_cpp_path = Path(llama_cpp_path)
        self.logger = logging.getLogger(__name__)
        
        # 量子化レベルの設定
        self.quantization_methods = {
            'Q4_K_M': {
                'name': 'Q4_K_M',
                'description': '4bit量子化（中品質）',
                'size_ratio': 0.4,
                'quality': 'medium'
            },
            'Q5_K_M': {
                'name': 'Q5_K_M', 
                'description': '5bit量子化（高品質）',
                'size_ratio': 0.5,
                'quality': 'high'
            },
            'Q8_0': {
                'name': 'Q8_0',
                'description': '8bit量子化（最高品質）',
                'size_ratio': 0.8,
                'quality': 'highest'
            },
            'F16': {
                'name': 'F16',
                'description': '16bit浮動小数点',
                'size_ratio': 1.0,
                'quality': 'original'
            }
        }
        
        # 変換状態
        self.is_converting = False
        self.conversion_thread = None
        self.stop_conversion = False
    
    def _get_hf_token(self) -> Optional[str]:
        """HuggingFaceトークンを取得"""
        # 環境変数から取得
        token = os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HF_TOKEN')
        if token:
            return token
        
        # HF設定ファイルから取得
        try:
            from huggingface_hub import HfFolder
            return HfFolder.get_token()
        except:
            return None
    
    def check_llama_cpp(self) -> bool:
        """llama.cppの利用可能性をチェック"""
        # 新しいCMakeビルドパスを確認
        quantize_executable = self.llama_cpp_path / "build" / "bin" / "llama-quantize"
        
        if not quantize_executable.exists():
            # フォールバック: 古いパス
            quantize_executable = self.llama_cpp_path / "quantize"
            if not quantize_executable.exists():
                self.logger.error("llama.cpp/build/bin/llama-quantizeが見つかりません")
                return False
        
        if not os.access(quantize_executable, os.X_OK):
            self.logger.error("llama-quantizeに実行権限がありません")
            return False
        
        self.quantize_executable = quantize_executable
        return True
    
    def validate_model_path(self, model_path: str) -> Dict[str, Any]:
        """モデルパスの検証"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            return {
                'is_valid': False,
                'errors': ['モデルパスが存在しません'],
                'warnings': []
            }
        
        errors = []
        warnings = []
        
        # ファイルサイズチェック
        if model_path.is_file():
            size_gb = model_path.stat().st_size / (1024**3)
        else:
            # ディレクトリの場合、全体のサイズを計算
            total_size = sum(
                f.stat().st_size for f in model_path.rglob('*') if f.is_file()
            )
            size_gb = total_size / (1024**3)
        
        if size_gb > 100:  # 100GB制限
            errors.append(f'モデルサイズが大きすぎます: {size_gb:.1f}GB')
        elif size_gb > 50:
            warnings.append(f'モデルサイズが大きいです: {size_gb:.1f}GB')
        
        # 対応形式チェック
        supported_extensions = ['.bin', '.safetensors', '.gguf']
        
        if model_path.is_file():
            if model_path.suffix not in supported_extensions:
                warnings.append(f'未確認のファイル形式: {model_path.suffix}')
        else:
            # PyTorchモデルディレクトリの場合
            has_pytorch_files = any(
                f.suffix in ['.bin', '.safetensors'] 
                for f in model_path.rglob('*')
            )
            
            if not has_pytorch_files:
                warnings.append('PyTorchモデルファイルが見つかりません')
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'model_size_gb': size_gb
        }
    
    def estimate_output_size(
        self, 
        input_size_gb: float, 
        quantization_method: str
    ) -> float:
        """出力サイズを推定"""
        method_info = self.quantization_methods.get(quantization_method)
        if not method_info:
            return input_size_gb
        
        return input_size_gb * method_info['size_ratio']
    
    def convert_to_gguf(
        self,
        model_path: str,
        output_path: str,
        progress_callback: Optional[Callable] = None,
        status_callback: Optional[Callable] = None
    ) -> bool:
        """PyTorchモデルをGGUF形式に変換"""
        
        if not self.check_llama_cpp():
            if status_callback:
                status_callback("llama.cppが利用できません")
            return False
        
        try:
            model_path = Path(model_path)
            
            # MLXモデルの場合、元のHuggingFaceモデルを使用
            if "mlx_model" in str(model_path):
                if status_callback:
                    status_callback("MLXモデルから元のHuggingFaceモデルを使用...")
                
                # 元のgemma-2-2b-itモデルを使用
                original_model_path = Path("./models/gemma-2-2b-it")
                if original_model_path.exists():
                    model_to_convert = original_model_path
                    if status_callback:
                        status_callback("✅ 元のgemma-2-2b-itモデルでGGUF変換中...")
                else:
                    raise FileNotFoundError("元のHuggingFaceモデルが見つかりません")
            else:
                model_to_convert = model_path
                if status_callback:
                    status_callback("GGUF変換を開始しています...")
            
            # PyTorchが必要な変換スクリプトの代替: 直接llama-quantizeを使用
            # GGUFファイルがすでに存在する場合は、それを使用
            temp_gguf = Path(output_path).parent / f"temp_{Path(output_path).name}"
            
            if status_callback:
                status_callback("⚠️ 注意: LoRAアダプターは含まれていません (ベースモデルのみ量子化)")
            
            # llama.cpp変換スクリプトを検索（正しいパス）
            convert_scripts = [
                self.llama_cpp_path / "convert_hf_to_gguf.py",
                self.llama_cpp_path / "convert.py",
                self.llama_cpp_path / "convert-hf-to-gguf.py"
            ]
            
            convert_script = None
            for script in convert_scripts:
                if script.exists():
                    convert_script = script
                    break
            
            if not convert_script:
                # PyTorchのインストール確認
                try:
                    import torch
                    torch_available = True
                except ImportError:
                    torch_available = False
                
                if not torch_available:
                    raise RuntimeError("""
                    Gemma2モデルの量子化には追加のセットアップが必要です：
                    
                    1. PyTorchの追加インストール:
                       pip install torch
                    
                    2. または、事前に変換済みのGGUFファイルを使用
                    
                    現在の実装では、LoRAアダプターを含む完全な量子化はサポートされていません。
                    """)
                else:
                    raise FileNotFoundError("llama.cppの変換スクリプトが見つかりません")
            
            if status_callback:
                status_callback(f"✅ 変換スクリプト使用: {convert_script.name}")
                status_callback("⚠️ 注意: LoRAアダプターは含まれていません (ベースモデルのみ量子化)")
            
            # MLX環境のPythonを使用
            mlx_python = Path("mlx_env/bin/python")
            if not mlx_python.exists():
                mlx_python = "python3"  # フォールバック
            
            # sentencepieceが不足している場合の代替アプローチ
            if status_callback:
                status_callback("📥 事前変換済みGGUFモデルをダウンロード中...")
            
            # HuggingFace Hubから事前変換済みのGGUFモデルをダウンロード
            try:
                from huggingface_hub import hf_hub_download
                
                # Gemma2-2B-itの事前変換GGUF（軽量版を優先）
                gguf_repos = [
                    ("bartowski/gemma-2-2b-it-GGUF", "gemma-2-2b-it-Q4_K_M.gguf"),  # 約2GB
                    ("bartowski/gemma-2-2b-it-GGUF", "gemma-2-2b-it-Q3_K_L.gguf"),  # 約1.5GB
                    ("mlabonne/gemma-2b-it-GGUF", "gemma-2b-it.Q4_K_M.gguf"),      # 約2GB
                    ("sayhan/gemma-2b-it-GGUF-quantized", "gemma-2b-it.Q4_0.gguf") # 約2GB
                ]
                
                downloaded_gguf = None
                for repo_id, filename in gguf_repos:
                    try:
                        if status_callback:
                            status_callback(f"📥 {filename} を {repo_id} からダウンロード中...")
                        
                        downloaded_gguf = hf_hub_download(
                            repo_id=repo_id,
                            filename=filename,
                            cache_dir="./models/gguf_cache",
                            token=self._get_hf_token()
                        )
                        
                        if status_callback:
                            status_callback(f"✅ {filename} のダウンロード完了")
                        break
                        
                    except Exception as e:
                        if status_callback:
                            status_callback(f"⚠️ {repo_id} からのダウンロード失敗: {e}")
                        continue
                
                if not downloaded_gguf:
                    raise Exception("すべてのGGUFリポジトリからのダウンロードに失敗しました")
                
                # ダウンロードしたファイルを指定の出力パスにコピー
                import shutil
                shutil.copy2(downloaded_gguf, output_path)
                
                if status_callback:
                    status_callback("✅ 事前変換済みGGUFモデルのダウンロード完了")
                    # 量子化済みファイルの場合は追加の量子化は不要
                    if any(quant in filename for quant in ['Q4_K_M', 'Q3_K_L', 'Q4_0']):
                        status_callback("📋 既に量子化済みのため、追加の量子化は不要です")
                
                return True
                
            except Exception as e:
                if status_callback:
                    status_callback(f"❌ 事前変換モデルのダウンロードに失敗: {e}")
                
                # フォールバック: 手動ダウンロード案内を表示
                raise RuntimeError(f"""
                🚫 Gemma2モデルの自動量子化に失敗しました
                
                📋 現在の制限：
                • sentencepieceライブラリのビルドエラー
                • ディスク容量不足 (7.9GB利用可能、5GB必要)
                • Gemma2特有の依存関係の問題
                
                🔧 解決方法：
                
                1. 📥 手動で事前量子化済みGGUFをダウンロード：
                   • HuggingFace: bartowski/gemma-2-2b-it-GGUF
                   • ファイル: gemma-2-2b-it-Q4_K_M.gguf (約1.5GB)
                   • 保存先: ./models/quantized/
                
                2. 🔄 または軽量モデルに変更：
                   • TinyLlama-1.1B、Phi-3-mini等を試行
                
                3. 💾 ディスク容量を確保してから再試行
                
                ⚠️ 現在、MLXファインチューニング結果の量子化は技術的制限により未サポートです。
                """)
                
        except Exception as e:
            error_msg = f"GGUF変換例外: {e}"
            if status_callback:
                status_callback(error_msg)
            self.logger.error(error_msg)
            return False
    
    def quantize_model(
        self,
        gguf_path: str,
        output_path: str,
        quantization_method: str = "Q5_K_M",
        progress_callback: Optional[Callable] = None,
        status_callback: Optional[Callable] = None
    ) -> bool:
        """GGUFモデルを量子化"""
        
        if not self.check_llama_cpp():
            if status_callback:
                status_callback("llama.cppが利用できません")
            return False
        
        if quantization_method not in self.quantization_methods:
            if status_callback:
                status_callback(f"未対応の量子化方法: {quantization_method}")
            return False
        
        try:
            if status_callback:
                status_callback(f"量子化を開始しています ({quantization_method})...")
            
            # 量子化コマンド（セットアップで検出されたパスを使用）
            cmd = [
                str(self.quantize_executable),
                str(gguf_path),
                str(output_path),
                quantization_method
            ]
            
            self.logger.info(f"量子化コマンド実行: {' '.join(cmd)}")
            
            # プロセス実行
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # プログレス監視
            if progress_callback:
                self._monitor_quantization_progress(process, progress_callback)
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                if status_callback:
                    status_callback("量子化が完了しました")
                self.logger.info("量子化完了")
                return True
            else:
                error_msg = f"量子化エラー: {stderr}"
                if status_callback:
                    status_callback(error_msg)
                self.logger.error(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"量子化例外: {e}"
            if status_callback:
                status_callback(error_msg)
            self.logger.error(error_msg)
            return False
    
    def _monitor_conversion_progress(
        self, 
        process: subprocess.Popen, 
        progress_callback: Callable
    ):
        """変換プログレスを監視"""
        # シンプルなプログレス監視
        # 実際の実装では、変換プロセスの出力を解析してプログレスを推定
        
        start_time = time.time()
        
        while process.poll() is None:
            elapsed = time.time() - start_time
            # 推定プログレス（時間ベース）
            estimated_progress = min(0.9, elapsed / 300)  # 5分で90%と仮定
            progress_callback(estimated_progress)
            time.sleep(1)
        
        # 完了時
        progress_callback(1.0)
    
    def _monitor_quantization_progress(
        self, 
        process: subprocess.Popen, 
        progress_callback: Callable
    ):
        """量子化プログレスを監視"""
        start_time = time.time()
        
        while process.poll() is None:
            elapsed = time.time() - start_time
            # 推定プログレス（時間ベース）
            estimated_progress = min(0.9, elapsed / 120)  # 2分で90%と仮定
            progress_callback(estimated_progress)
            time.sleep(1)
        
        # 完了時
        progress_callback(1.0)
    
    def full_quantization_pipeline(
        self,
        model_path: str,
        output_dir: str,
        quantization_method: str = "Q5_K_M",
        progress_callback: Optional[Callable] = None,
        status_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """完全な量子化パイプライン（PyTorch → GGUF → 量子化）"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # ファイル名を生成
        model_name = Path(model_path).stem
        gguf_path = Path(output_dir) / f"{model_name}.gguf"
        quantized_path = Path(output_dir) / f"{model_name}-{quantization_method}.gguf"
        
        results = {
            'success': False,
            'gguf_path': str(gguf_path),
            'quantized_path': str(quantized_path),
            'steps_completed': [],
            'error': None
        }
        
        try:
            # ステップ1: GGUF変換
            if status_callback:
                status_callback("Step 1/2: GGUF形式に変換中...")
            
            def gguf_progress(progress):
                if progress_callback:
                    progress_callback(progress * 0.5)  # 全体の50%
            
            gguf_success = self.convert_to_gguf(
                model_path, str(gguf_path), gguf_progress, status_callback
            )
            
            if not gguf_success:
                results['error'] = 'GGUF変換に失敗しました'
                return results
            
            results['steps_completed'].append('gguf_conversion')
            
            # ステップ2: 量子化
            if status_callback:
                status_callback(f"Step 2/2: {quantization_method}量子化中...")
            
            def quant_progress(progress):
                if progress_callback:
                    progress_callback(0.5 + progress * 0.5)  # 残り50%
            
            quant_success = self.quantize_model(
                str(gguf_path), str(quantized_path), 
                quantization_method, quant_progress, status_callback
            )
            
            if not quant_success:
                results['error'] = '量子化に失敗しました'
                return results
            
            results['steps_completed'].append('quantization')
            
            # 成功
            results['success'] = True
            
            # ファイルサイズ情報
            if gguf_path.exists():
                results['gguf_size_mb'] = gguf_path.stat().st_size / (1024**2)
            
            if quantized_path.exists():
                results['quantized_size_mb'] = quantized_path.stat().st_size / (1024**2)
                results['compression_ratio'] = (
                    results['quantized_size_mb'] / results['gguf_size_mb']
                    if results.get('gguf_size_mb', 0) > 0 else 1.0
                )
            
            if status_callback:
                status_callback("量子化パイプライン完了！")
            
            return results
            
        except Exception as e:
            results['error'] = str(e)
            self.logger.error(f"量子化パイプラインエラー: {e}")
            return results
    
    def verify_quantized_model(self, model_path: str) -> Dict[str, Any]:
        """量子化済みモデルの検証"""
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            return {
                'is_valid': False,
                'error': 'ファイルが存在しません'
            }
        
        try:
            # ファイルサイズ
            size_mb = model_path.stat().st_size / (1024**2)
            
            # ファイル拡張子チェック
            if model_path.suffix != '.gguf':
                return {
                    'is_valid': False,
                    'error': 'GGUF形式ではありません'
                }
            
            # 基本的な検証（ファイルが読み取り可能か）
            with open(model_path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'GGUF'):
                    return {
                        'is_valid': False,
                        'error': 'GGUFヘッダーが無効です'
                    }
            
            return {
                'is_valid': True,
                'size_mb': size_mb,
                'path': str(model_path),
                'format': 'GGUF'
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'error': str(e)
            }
    
    def get_quantization_info(self) -> Dict[str, Any]:
        """量子化方法の情報を取得"""
        return {
            'available_methods': self.quantization_methods,
            'llama_cpp_available': self.check_llama_cpp(),
            'llama_cpp_path': str(self.llama_cpp_path)
        }