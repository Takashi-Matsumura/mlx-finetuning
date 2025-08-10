import os
import json
import time
import logging
from typing import Dict, Any, Optional, Callable, List, Tuple
from pathlib import Path
import threading
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate, convert
from mlx_lm.utils import load_config
from mlx_lm.tuner import train, linear_to_lora_layers, TrainingArgs
from mlx_lm.tuner.lora import LoRALinear
import mlx.optimizers as optim
from transformers import AutoTokenizer
import numpy as np

from .experiment_tracker import ExperimentTracker


class MLXFineTuner:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.experiment_tracker = ExperimentTracker()
        
        # デフォルト設定
        self.default_config = {
            'batch_size': 1,
            'learning_rate': 5e-5,
            'num_epochs': 3,
            'lora_rank': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'max_seq_length': 2048,
            'save_steps': 500,
            'logging_steps': 10,
            'warmup_steps': 100,
            'weight_decay': 0.01
        }
        
        # 設定をマージ
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def validate_and_convert_local_model(
        self, 
        model_name: str, 
        output_dir: str,
        status_callback: Optional[Callable] = None
    ) -> Tuple[bool, str]:
        """ローカルモデルファイルの検証とMLX形式への変換"""
        
        try:
            # ローカルモデルファイルの存在チェック
            local_model_path = Path(f"./models/{model_name.split('/')[-1]}")
            
            if status_callback:
                status_callback(f"📁 ローカルモデルファイルを確認中: {local_model_path}")
            
            if not local_model_path.exists():
                error_msg = f"❌ モデルディレクトリが見つかりません: {local_model_path}"
                self.logger.error(error_msg)
                if status_callback:
                    status_callback(error_msg)
                return False, f"モデルディレクトリが存在しません: {local_model_path}"
            
            # 必要なファイルが存在するかチェック
            required_files = [
                "config.json", 
                "model.safetensors.index.json"
            ]
            
            # トークナイザーファイルの確認（どちらか一方があればOK）
            tokenizer_files = ["tokenizer.json", "tokenizer.model"]
            has_tokenizer = any((local_model_path / f).exists() for f in tokenizer_files)
            
            # safetensorsファイルの確認
            safetensors_files = list(local_model_path.glob("model-*.safetensors"))
            
            # ファイル検証
            missing_files = []
            for file in required_files:
                if not (local_model_path / file).exists():
                    missing_files.append(file)
            
            if not has_tokenizer:
                missing_files.append("tokenizer.json または tokenizer.model")
            
            if len(safetensors_files) == 0:
                missing_files.append("model-*.safetensors")
            
            if missing_files:
                error_msg = f"❌ 必要なファイルが不足: {', '.join(missing_files)}"
                self.logger.error(error_msg)
                if status_callback:
                    status_callback(error_msg)
                return False, f"必要なファイルが不足しています: {missing_files}"
            
            self.logger.info(f"✅ ローカルモデル検証完了: {local_model_path}")
            if status_callback:
                status_callback(f"✅ ローカルモデル '{model_name}' を検証完了")
            
            # MLX形式の出力ディレクトリ（タイムスタンプでユニーク化）
            import time
            timestamp = str(int(time.time()))
            mlx_model_dir = Path(output_dir) / f"mlx_model_{timestamp}"
            
            # 既存のディレクトリがあれば削除
            if mlx_model_dir.exists():
                import shutil
                self.logger.info(f"既存のモデルディレクトリを削除: {mlx_model_dir}")
                shutil.rmtree(mlx_model_dir, ignore_errors=True)
                
            # 親ディレクトリ作成
            mlx_model_dir.parent.mkdir(parents=True, exist_ok=True)
            
            # ローカルモデルからMLX形式に変換
            if status_callback:
                status_callback(f"🔄 ローカルモデルをMLX形式に変換中...")
            
            convert(
                hf_path=str(local_model_path),
                mlx_path=str(mlx_model_dir),
                quantize=False,  # 量子化はしない（ファインチューニング用）
                dtype="float16"  # メモリ効率のためfp16を使用
            )
            
            if status_callback:
                status_callback("✅ MLX変換完了")
            
            self.logger.info(f"ローカルモデル変換完了: {mlx_model_dir}")
            return True, str(mlx_model_dir)
                    
        except Exception as e:
            error_msg = f"ローカルモデル処理エラー: {str(e)}"
            self.logger.error(error_msg)
            if status_callback:
                status_callback(f"❌ {error_msg}")
            return False, error_msg
    
    def load_model_and_tokenizer(
        self, 
        model_path: str
    ) -> Tuple[nn.Module, AutoTokenizer]:
        """MLXモデルとトークナイザーを読み込み"""
        
        try:
            self.logger.info(f"MLXモデル読み込み: {model_path}")
            
            # MLX-LMのload関数を使用
            model, tokenizer = load(model_path)
            
            self.logger.info("モデル読み込み完了")
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"モデル読み込みエラー: {e}")
            raise
    
    def apply_lora_layers(
        self, 
        model: nn.Module
    ) -> nn.Module:
        """LoRAレイヤーを既存モデルに適用"""
        
        try:
            rank = self.config.get('lora_rank', 16)
            alpha = self.config.get('lora_alpha', 32)
            dropout = self.config.get('lora_dropout', 0.1)
            
            # LoRAを適用するレイヤーを特定
            lora_targets = [
                "self_attn.q_proj",
                "self_attn.k_proj", 
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj"
            ]
            
            # モデルにLoRAレイヤーを追加
            def apply_lora_to_linear(module, name):
                for target in lora_targets:
                    if target in name and isinstance(module, nn.Linear):
                        # 既存のLinearレイヤーをLoRALinearに置き換え
                        lora_layer = LoRALinear(
                            input_dims=module.weight.shape[1],
                            output_dims=module.weight.shape[0],
                            r=rank,
                            alpha=alpha,
                            dropout=dropout,
                            bias=module.bias is not None
                        )
                        # 元の重みをコピー
                        lora_layer.linear.weight = module.weight
                        if module.bias is not None:
                            lora_layer.linear.bias = module.bias
                        
                        return lora_layer
                return module
            
            # モデル全体にLoRAを適用
            model.apply(apply_lora_to_linear)
            
            self.logger.info(f"LoRA適用完了 (rank={rank}, alpha={alpha})")
            return model
            
        except Exception as e:
            self.logger.error(f"LoRA適用エラー: {e}")
            raise
    
    def prepare_dataset(
        self, 
        dataset_path: str, 
        tokenizer: AutoTokenizer
    ) -> List[Dict[str, mx.array]]:
        """データセットを準備してトークン化"""
        
        try:
            # データセット読み込み
            dataset = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    dataset.append(data)
            
            self.logger.info(f"データセット読み込み: {len(dataset)} 件")
            
            # トークン化
            tokenized_dataset = []
            max_length = self.config.get('max_seq_length', 2048)
            
            for item in dataset:
                text = item.get('text', '')
                
                # トークン化
                tokens = tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None
                )
                
                # MLX配列に変換
                input_ids = mx.array(tokens['input_ids'], dtype=mx.int32)
                attention_mask = mx.array(tokens['attention_mask'], dtype=mx.int32)
                
                tokenized_dataset.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': input_ids  # 言語モデリングなのでlabelsはinput_idsと同じ
                })
            
            self.logger.info(f"トークン化完了: {len(tokenized_dataset)} 件")
            return tokenized_dataset
            
        except Exception as e:
            self.logger.error(f"データセット準備エラー: {e}")
            raise
    
    def _get_dataset_size(self, dataset_file: str) -> int:
        """データセットのサイズを取得"""
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception as e:
            self.logger.warning(f"データセットサイズ取得エラー: {e}")
            return 10  # デフォルト値
    
    def _verify_training_config(self, output_dir: str) -> Dict[str, Any]:
        """トレーニングで実際に使用された設定を検証"""
        try:
            adapter_config_path = os.path.join(output_dir, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path, 'r', encoding='utf-8') as f:
                    actual_config = json.load(f)
                
                # 重要なパラメータを抽出
                return {
                    'iters': actual_config.get('iters', 'unknown'),
                    'learning_rate': actual_config.get('learning_rate', 'unknown'),
                    'batch_size': actual_config.get('batch_size', 'unknown'),
                    'lora_rank': actual_config.get('lora_parameters', {}).get('rank', 'unknown'),
                    'lora_scale': actual_config.get('lora_parameters', {}).get('scale', 'unknown'),
                    'max_seq_length': actual_config.get('max_seq_length', 'unknown')
                }
            else:
                return {'status': 'config_file_not_found'}
        except Exception as e:
            self.logger.warning(f"パラメータ検証エラー: {e}")
            return {'status': 'verification_error', 'error': str(e)}
    
    def compute_loss(
        self, 
        model: nn.Module, 
        batch: Dict[str, mx.array]
    ) -> mx.array:
        """損失を計算"""
        
        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        
        # モデルの順伝播
        logits = model(input_ids)
        
        # クロスエントロピー損失を計算
        # logits: [batch_size, seq_len, vocab_size]
        # labels: [batch_size, seq_len]
        
        vocab_size = logits.shape[-1]
        
        # logitsとlabelsを1次元に変換
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)
        
        # パディングトークン(-100)を無視
        mask = labels_flat != -100
        
        # クロスエントロピー損失
        loss = nn.losses.cross_entropy(logits_flat, labels_flat, reduction='none')
        
        # マスクを適用して平均を取る
        loss_masked = loss * mask
        total_loss = mx.sum(loss_masked) / mx.sum(mask)
        
        return total_loss
    
    def run_actual_finetuning(
        self,
        experiment_id: str,
        model_name: str,
        dataset_path: str,
        output_dir: str,
        progress_callback: Optional[Callable] = None,
        status_callback: Optional[Callable] = None
    ):
        """実際のMLXファインチューニングを実行"""
        
        try:
            # ステップ1: ローカルモデルの検証と変換
            if status_callback:
                status_callback("🔄 ローカルモデルの検証と変換...")
            
            success, model_path = self.validate_and_convert_local_model(
                model_name, output_dir, status_callback
            )
            
            if not success:
                raise RuntimeError(f"モデル準備失敗: {model_path}")
            
            if progress_callback:
                progress_callback(0.3)
            
            # ステップ2: MLX-LMの高レベルAPIを使用した実際のファインチューニング
            if status_callback:
                status_callback("🚀 実際のファインチューニング開始...")
            
            self._run_mlx_training(
                model_path, dataset_path, experiment_id, output_dir,
                progress_callback, status_callback
            )
            
            if progress_callback:
                progress_callback(1.0)
            
            if status_callback:
                status_callback("🎉 ファインチューニング完了！")
                
        except Exception as e:
            error_msg = f"ファインチューニングエラー: {str(e)}"
            self.logger.error(error_msg)
            if status_callback:
                status_callback(f"❌ {error_msg}")
            raise
    
    def _run_mlx_training(
        self,
        model_path: str,
        dataset_path: str,
        experiment_id: str,
        output_dir: str,
        progress_callback: Optional[Callable] = None,
        status_callback: Optional[Callable] = None
    ):
        """MLX-LMのコマンドラインツールを使用した実際のファインチューニング"""
        
        try:
            import subprocess
            import threading
            import time
            
            if status_callback:
                status_callback("🚀 MLX-LMコマンドラインでファインチューニング開始...")
            
            # MLX-LM用のデータセットディレクトリを準備
            dataset_dir = os.path.join(output_dir, "dataset")
            os.makedirs(dataset_dir, exist_ok=True)
            
            # train.jsonlファイルを作成（元のdataset.jsonlをコピー）
            train_file = os.path.join(dataset_dir, "train.jsonl")
            import shutil
            shutil.copy2(dataset_path, train_file)
            
            # 最小限のvalid.jsonlとtest.jsonlを作成（MLX-LMが期待する構造）
            valid_file = os.path.join(dataset_dir, "valid.jsonl")
            test_file = os.path.join(dataset_dir, "test.jsonl")
            
            # 最小限のダミーデータを作成（空ファイルだとIndexErrorになる）
            dummy_data = '{"text": "### 指示:\\nテスト\\n\\n### 回答:\\nテストです。"}\n'
            
            with open(valid_file, 'w', encoding='utf-8') as f:
                f.write(dummy_data)
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(dummy_data)
            
            self.logger.info(f"データセットディレクトリ準備完了: {dataset_dir}")
            
            # データセットサイズを動的に計算
            dataset_size = self._get_dataset_size(train_file)
            
            # イテレーション数を正しく計算
            # データセットサイズが小さい場合は、十分なイテレーション数を確保
            epochs = self.config.get('num_epochs', 3)
            if dataset_size <= 10:
                # 小さなデータセットの場合は十分なイテレーション数
                total_iters = max(200, epochs * dataset_size * 10)  # 最低200回
            else:
                total_iters = epochs * dataset_size
            
            # LoRA設定ファイルを作成
            config_file = os.path.join(output_dir, "lora_config.yaml")
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(f"""# LoRA Fine-tuning Configuration
model: {model_path}
train: true
data: {dataset_dir}
fine_tune_type: lora
batch_size: {self.config.get('batch_size', 1)}
iters: {total_iters}
learning_rate: {self.config.get('learning_rate', 5e-5)}
steps_per_report: {self.config.get('logging_steps', 10)}
save_every: {min(100, total_iters // 4)}
adapter_path: {output_dir}
max_seq_length: {self.config.get('max_seq_length', 2048)}
val_batches: 0

# LoRA Parameters
lora_parameters:
  rank: {self.config.get('lora_rank', 16)}
  alpha: {self.config.get('lora_alpha', 32)}
  dropout: {self.config.get('lora_dropout', 0.1)}
  scale: {self.config.get('lora_alpha', 32) / self.config.get('lora_rank', 16)}
""")
            
            # MLX-LMのファインチューニングコマンドを構築（設定ファイル使用）
            cmd = [
                "python", "-m", "mlx_lm", "lora",
                "--config", config_file
            ]
            
            self.logger.info(f"実行コマンド: {' '.join(cmd)}")
            
            # プロセスを実行
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=os.getcwd()
            )
            
            step_count = 0
            total_steps = total_iters  # 正しい総ステップ数を使用
            
            # 出力を監視
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                    
                if output.strip():
                    self.logger.info(f"MLX-LM: {output.strip()}")
                    
                    # ステップ情報を解析
                    if "Iter" in output and "Loss" in output:
                        try:
                            # "Iter 10: Loss: 1.234" のような形式を解析
                            parts = output.split()
                            for i, part in enumerate(parts):
                                if part == "Iter" and i + 1 < len(parts):
                                    step_str = parts[i + 1].rstrip(":")
                                    step_count = int(step_str)
                                    break
                                if part == "Loss:" and i + 1 < len(parts):
                                    loss_str = parts[i + 1]
                                    loss = float(loss_str)
                                    
                                    if status_callback:
                                        status_callback(f"🔄 Step {step_count} - Loss: {loss:.4f}")
                                    
                                    # 実験にメトリクス記録
                                    self.experiment_tracker.log_metrics(
                                        experiment_id,
                                        step=step_count,
                                        metrics={
                                            'loss': loss,
                                            'learning_rate': self.config.get('learning_rate', 5e-5)
                                        }
                                    )
                                    
                                    if progress_callback:
                                        progress = 0.3 + min(0.6, 0.6 * step_count / total_steps)
                                        progress_callback(progress)
                                    break
                        except (ValueError, IndexError):
                            pass
                    
                    elif "Saving" in output or "adapters.safetensors" in output:
                        if status_callback:
                            status_callback("💾 アダプターを保存中...")
            
            # プロセス完了を待機
            return_code = process.wait()
            
            if return_code == 0:
                self.logger.info(f"MLX-LMファインチューニング完了: {output_dir}")
                
                if status_callback:
                    status_callback("🎉 ファインチューニング完了！")
                
                # パラメータ検証を実行
                actual_config = self._verify_training_config(output_dir)
                
                # 実験完了を記録
                adapter_file = os.path.join(output_dir, "adapters.safetensors")
                self.experiment_tracker.complete_experiment(
                    experiment_id,
                    output_dir=output_dir,
                    metrics={
                        'status': 'completed', 
                        'adapter_file': adapter_file,
                        'final_step': step_count,
                        'configured_params': self.config,
                        'actual_params': actual_config,
                        'total_iterations': total_iters,
                        'dataset_size': dataset_size
                    }
                )
                
                if progress_callback:
                    progress_callback(1.0)
                    
            else:
                raise RuntimeError(f"MLX-LMプロセスがエラーで終了: return code {return_code}")
            
        except Exception as e:
            error_msg = f"MLX-LMトレーニングエラー: {str(e)}"
            self.logger.error(error_msg)
            
            # 実験を失敗として記録
            self.experiment_tracker.fail_experiment(
                experiment_id,
                f"{error_msg} | Config: {self.config} | Command: {' '.join(cmd) if 'cmd' in locals() else 'N/A'}"
            )
            
            if status_callback:
                status_callback(f"❌ エラー: {error_msg}")
            
            raise RuntimeError(error_msg)
            self.logger.error(error_msg)
            if status_callback:
                status_callback(f"❌ {error_msg}")
            raise
    
    def _train_model(
        self,
        model: nn.Module,
        dataset: List[Dict[str, mx.array]], 
        experiment_id: str,
        output_dir: str,
        progress_callback: Optional[Callable] = None,
        status_callback: Optional[Callable] = None
    ):
        """実際のモデル訓練を実行"""
        
        # オプティマイザー設定
        learning_rate = self.config.get('learning_rate', 5e-5)
        optimizer = optim.AdamW(learning_rate=learning_rate)
        
        num_epochs = self.config.get('num_epochs', 3)
        batch_size = self.config.get('batch_size', 1)
        logging_steps = self.config.get('logging_steps', 10)
        save_steps = self.config.get('save_steps', 500)
        
        total_steps = len(dataset) * num_epochs // batch_size
        current_step = 0
        
        self.logger.info(f"訓練開始: {total_steps} ステップ, {num_epochs} エポック")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(dataset), batch_size):
                # バッチ作成
                batch_data = dataset[i:i+batch_size]
                
                # 実際の順伝播と損失計算
                batch_loss = mx.array(0.0)
                
                for item in batch_data:
                    loss = self.compute_loss(model, item)
                    batch_loss += loss
                
                batch_loss = batch_loss / len(batch_data)
                
                # バックプロパゲーション
                loss_and_grad_fn = nn.value_and_grad(model, self.compute_loss)
                loss_value, grads = loss_and_grad_fn(model, batch_data[0])  # 簡略化
                
                # パラメータ更新
                optimizer.update(model, grads)
                
                # メトリクス記録
                current_step += 1
                epoch_loss += float(batch_loss)
                num_batches += 1
                
                # プログレス更新
                if progress_callback:
                    base_progress = 0.5 + (0.4 * current_step / total_steps)
                    progress_callback(min(0.9, base_progress))
                
                # ログ出力
                if current_step % logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    
                    log_msg = f"Step {current_step}/{total_steps} - Epoch {epoch+1}/{num_epochs} - Loss: {float(batch_loss):.4f}"
                    self.logger.info(log_msg)
                    
                    if status_callback:
                        status_callback(f"🔄 {log_msg}")
                    
                    # 実験にメトリクス記録
                    self.experiment_tracker.log_metrics(
                        experiment_id,
                        step=current_step,
                        metrics={
                            'loss': float(batch_loss),
                            'avg_loss': avg_loss,
                            'epoch': epoch + 1,
                            'learning_rate': learning_rate
                        }
                    )
                
                # チェックポイント保存
                if current_step % save_steps == 0:
                    checkpoint_path = os.path.join(output_dir, f'checkpoint-{current_step}')
                    self._save_checkpoint(model, optimizer, checkpoint_path, current_step)
                    
                    if status_callback:
                        status_callback(f"💾 チェックポイント保存: Step {current_step}")
            
            # エポック終了処理
            avg_epoch_loss = epoch_loss / max(1, num_batches)
            epoch_msg = f"✅ エポック {epoch + 1} 完了 - 平均損失: {avg_epoch_loss:.4f}"
            self.logger.info(epoch_msg)
            
            if status_callback:
                status_callback(epoch_msg)
        
        # 最終モデル保存
        final_model_path = os.path.join(output_dir, 'final_model')
        self._save_final_model(model, tokenizer, final_model_path, experiment_id)
        
        self.logger.info(f"ファインチューニング完了 - 最終モデル: {final_model_path}")
    
    def _save_checkpoint(
        self, 
        model: nn.Module, 
        optimizer: optim.Optimizer, 
        checkpoint_path: str, 
        step: int
    ):
        """チェックポイントを保存"""
        
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # モデルの重みを保存
        model_weights = {}
        for name, param in model.named_parameters():
            model_weights[name] = param
        
        # チェックポイント情報
        checkpoint = {
            'step': step,
            'model_state_dict': model_weights,
            'optimizer_state_dict': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else {},
            'config': self.config
        }
        
        # 保存
        checkpoint_file = os.path.join(checkpoint_path, 'checkpoint.npz')
        mx.savez(checkpoint_file, **model_weights)
        
        # メタデータをJSONで保存
        meta_file = os.path.join(checkpoint_path, 'checkpoint_meta.json')
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump({
                'step': step,
                'config': self.config
            }, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"チェックポイント保存完了: {checkpoint_path}")
    
    def _save_final_model(
        self, 
        model: nn.Module, 
        tokenizer: AutoTokenizer, 
        model_path: str, 
        experiment_id: str
    ):
        """最終モデルを保存"""
        
        os.makedirs(model_path, exist_ok=True)
        
        # モデルの重みを保存
        model_weights = {}
        for name, param in model.named_parameters():
            model_weights[name] = param
        
        weights_file = os.path.join(model_path, 'model.npz')
        mx.savez(weights_file, **model_weights)
        
        # トークナイザーを保存
        tokenizer.save_pretrained(model_path)
        
        # モデル設定を保存
        model_info = {
            'experiment_id': experiment_id,
            'config': self.config,
            'model_type': 'mlx_lora_finetuned',
            'base_model': self.config.get('model_name', 'unknown'),
            'lora_rank': self.config.get('lora_rank', 16),
            'lora_alpha': self.config.get('lora_alpha', 32),
            'training_steps': 'completed'
        }
        
        config_file = os.path.join(model_path, 'model_config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"最終モデル保存完了: {model_path}")
        
        return model_path