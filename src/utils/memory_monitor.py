import psutil
import time
from typing import Dict, Optional, Tuple
import logging


class MemoryMonitor:
    def __init__(self, max_memory_gb: int = 32, safety_margin_gb: int = 2):
        self.max_memory_gb = max_memory_gb
        self.safety_margin_gb = safety_margin_gb
        self.available_memory_gb = max_memory_gb - safety_margin_gb
        self.logger = logging.getLogger(__name__)
    
    def get_memory_info(self) -> Dict[str, float]:
        """現在のメモリ使用状況を取得"""
        memory = psutil.virtual_memory()
        
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent,
            'free_gb': memory.free / (1024**3)
        }
    
    def check_available_memory(self) -> bool:
        """利用可能メモリが十分かチェック"""
        memory_info = self.get_memory_info()
        available_gb = memory_info['available_gb']
        
        return available_gb >= self.safety_margin_gb
    
    def estimate_model_memory(self, model_name: str, batch_size: int = 1) -> float:
        """モデルの推定メモリ使用量を計算（GB）"""
        # モデルサイズの推定マップ
        model_sizes = {
            'elyza/Llama-3-ELYZA-JP-8B': 16.0,
            'google/gemma-2-9b-it': 18.0,
            'google/gemma-2-2b-it': 4.0,
            'meta-llama/Llama-3.1-8B-Instruct': 16.0,
        }
        
        # ベースモデルサイズ
        base_size = model_sizes.get(model_name, 8.0)  # デフォルト8GB
        
        # ファインチューニング時の追加メモリ
        # - オプティマイザー状態: モデルサイズの約2倍
        # - 勾配: モデルサイズと同等
        # - アクティベーション: バッチサイズに比例
        training_overhead = base_size * 3.0  # オプティマイザー + 勾配
        activation_memory = batch_size * 0.5  # バッチサイズあたり500MB
        
        total_memory = base_size + training_overhead + activation_memory
        
        return total_memory
    
    def can_run_training(
        self, 
        model_name: str, 
        batch_size: int = 1
    ) -> Tuple[bool, str]:
        """トレーニングが実行可能かチェック"""
        current_memory = self.get_memory_info()
        estimated_memory = self.estimate_model_memory(model_name, batch_size)
        
        available_memory = current_memory['available_gb']
        
        if estimated_memory > available_memory:
            return False, (
                f"メモリ不足: 推定必要量 {estimated_memory:.1f}GB > "
                f"利用可能量 {available_memory:.1f}GB"
            )
        
        if estimated_memory > self.available_memory_gb:
            return False, (
                f"メモリ制限超過: 推定必要量 {estimated_memory:.1f}GB > "
                f"制限 {self.available_memory_gb:.1f}GB"
            )
        
        return True, f"OK: 推定使用量 {estimated_memory:.1f}GB"
    
    def suggest_batch_size(self, model_name: str) -> int:
        """推奨バッチサイズを提案"""
        current_memory = self.get_memory_info()
        available_memory = current_memory['available_gb']
        
        # バッチサイズ1から開始して、メモリ制限内で最大を見つける
        max_batch_size = 1
        
        for batch_size in range(1, 9):  # 最大8まで試す
            estimated_memory = self.estimate_model_memory(model_name, batch_size)
            
            if estimated_memory <= available_memory * 0.8:  # 80%を上限とする
                max_batch_size = batch_size
            else:
                break
        
        return max_batch_size
    
    def monitor_training(self, interval: int = 10) -> None:
        """トレーニング中のメモリ監視"""
        start_time = time.time()
        max_memory = 0
        
        while True:
            try:
                memory_info = self.get_memory_info()
                current_used = memory_info['used_gb']
                
                if current_used > max_memory:
                    max_memory = current_used
                
                # メモリ使用量が危険レベルに達した場合
                if memory_info['percent'] > 90:
                    self.logger.warning(
                        f"メモリ使用量が危険レベル: {memory_info['percent']:.1f}%"
                    )
                
                # 定期ログ
                elapsed = time.time() - start_time
                if elapsed % 60 < interval:  # 1分ごとにログ
                    self.logger.info(
                        f"メモリ使用量: {current_used:.1f}GB "
                        f"({memory_info['percent']:.1f}%), "
                        f"最大: {max_memory:.1f}GB"
                    )
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                self.logger.info("メモリ監視を停止しました")
                break
            except Exception as e:
                self.logger.error(f"メモリ監視エラー: {e}")
                time.sleep(interval)
    
    def get_recommendation(self, model_name: str) -> Dict[str, any]:
        """設定推奨値を取得"""
        memory_info = self.get_memory_info()
        recommended_batch_size = self.suggest_batch_size(model_name)
        can_run, message = self.can_run_training(model_name, recommended_batch_size)
        
        return {
            'current_memory': memory_info,
            'recommended_batch_size': recommended_batch_size,
            'can_run_training': can_run,
            'message': message,
            'estimated_memory_gb': self.estimate_model_memory(
                model_name, recommended_batch_size
            )
        }