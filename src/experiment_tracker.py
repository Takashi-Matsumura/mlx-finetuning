import os
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging


class ExperimentTracker:
    def __init__(self, experiments_dir: str = "./experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # メタデータファイル
        self.metadata_file = self.experiments_dir / "experiments_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """実験メタデータを読み込み"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                self.logger.warning(f"メタデータ読み込みエラー: {e}")
                self.metadata = {"experiments": {}}
        else:
            self.metadata = {"experiments": {}}
    
    def _save_metadata(self):
        """実験メタデータを保存"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"メタデータ保存エラー: {e}")
    
    def create_experiment(
        self,
        model_name: str,
        dataset_path: str,
        config: Dict[str, Any],
        description: str = ""
    ) -> str:
        """新しい実験を作成"""
        
        experiment_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        # 実験ディレクトリを作成
        experiment_dir = self.experiments_dir / experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        # 実験情報
        experiment_info = {
            "id": experiment_id,
            "model_name": model_name,
            "dataset_path": dataset_path,
            "description": description,
            "config": config,
            "status": "running",
            "created_at": timestamp,
            "started_at": timestamp,
            "completed_at": None,
            "duration_seconds": None,
            "output_dir": None,
            "metrics": [],
            "logs": [],
            "final_metrics": {},
            "error": None
        }
        
        # メタデータに追加
        self.metadata["experiments"][experiment_id] = experiment_info
        self._save_metadata()
        
        # 実験ファイルを保存
        experiment_file = experiment_dir / "experiment_info.json"
        with open(experiment_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_info, f, ensure_ascii=False, indent=2)
        
        # 設定ファイルを保存
        config_file = experiment_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"実験 {experiment_id} を作成しました")
        return experiment_id
    
    def log_metrics(
        self,
        experiment_id: str,
        step: int,
        metrics: Dict[str, float]
    ):
        """メトリクスをログに記録"""
        
        if experiment_id not in self.metadata["experiments"]:
            self.logger.error(f"実験 {experiment_id} が見つかりません")
            return
        
        timestamp = datetime.now().isoformat()
        
        # メトリクスエントリ
        metrics_entry = {
            "step": step,
            "timestamp": timestamp,
            "metrics": metrics
        }
        
        # メモリ上のメタデータを更新
        if "metrics" not in self.metadata["experiments"][experiment_id]:
            self.metadata["experiments"][experiment_id]["metrics"] = []
        
        self.metadata["experiments"][experiment_id]["metrics"].append(metrics_entry)
        
        # ファイルにも保存
        experiment_dir = self.experiments_dir / experiment_id
        metrics_file = experiment_dir / "metrics.jsonl"
        
        with open(metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics_entry, ensure_ascii=False) + '\n')
        
        # 定期的にメタデータを保存（100ステップごと）
        if step % 100 == 0:
            self._save_metadata()
    
    def log_message(
        self,
        experiment_id: str,
        level: str,
        message: str
    ):
        """ログメッセージを記録"""
        
        if experiment_id not in self.metadata["experiments"]:
            return
        
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message
        }
        
        # メタデータに追加
        if "logs" not in self.metadata["experiments"][experiment_id]:
            self.metadata["experiments"][experiment_id]["logs"] = []
        
        self.metadata["experiments"][experiment_id]["logs"].append(log_entry)
        
        # ログファイルに保存
        experiment_dir = self.experiments_dir / experiment_id
        log_file = experiment_dir / "logs.jsonl"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def complete_experiment(
        self,
        experiment_id: str,
        output_dir: str = None,
        metrics: Dict[str, float] = None
    ):
        """実験完了を記録"""
        
        if experiment_id not in self.metadata["experiments"]:
            self.logger.error(f"実験 {experiment_id} が見つかりません")
            return
        
        timestamp = datetime.now().isoformat()
        start_time = self.metadata["experiments"][experiment_id]["started_at"]
        
        # 実行時間を計算
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(timestamp)
        duration = (end_dt - start_dt).total_seconds()
        
        # 実験情報を更新
        self.metadata["experiments"][experiment_id].update({
            "status": "completed",
            "completed_at": timestamp,
            "duration_seconds": duration,
            "output_dir": output_dir,
            "final_metrics": metrics or {}
        })
        
        self._save_metadata()
        
        # 実験ファイルを更新
        experiment_dir = self.experiments_dir / experiment_id
        experiment_file = experiment_dir / "experiment_info.json"
        
        with open(experiment_file, 'w', encoding='utf-8') as f:
            json.dump(
                self.metadata["experiments"][experiment_id], 
                f, ensure_ascii=False, indent=2
            )
        
        self.logger.info(f"実験 {experiment_id} が完了しました（実行時間: {duration:.1f}秒）")
    
    def fail_experiment(
        self,
        experiment_id: str,
        error_message: str
    ):
        """実験失敗を記録"""
        
        if experiment_id not in self.metadata["experiments"]:
            return
        
        timestamp = datetime.now().isoformat()
        start_time = self.metadata["experiments"][experiment_id]["started_at"]
        
        # 実行時間を計算
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(timestamp)
        duration = (end_dt - start_dt).total_seconds()
        
        # 実験情報を更新
        self.metadata["experiments"][experiment_id].update({
            "status": "failed",
            "completed_at": timestamp,
            "duration_seconds": duration,
            "error": error_message
        })
        
        self._save_metadata()
        
        # ログにエラーを記録
        self.log_message(experiment_id, "ERROR", error_message)
        
        self.logger.error(f"実験 {experiment_id} が失敗しました: {error_message}")
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """実験情報を取得"""
        return self.metadata["experiments"].get(experiment_id)
    
    def list_experiments(
        self,
        status: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """実験リストを取得"""
        
        experiments = list(self.metadata["experiments"].values())
        
        # フィルタリング
        if status:
            experiments = [exp for exp in experiments if exp.get("status") == status]
        
        if model_name:
            experiments = [exp for exp in experiments if exp.get("model_name") == model_name]
        
        # 作成日時でソート（新しい順）
        experiments.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        # 制限
        if limit:
            experiments = experiments[:limit]
        
        return experiments
    
    def get_experiment_metrics(self, experiment_id: str) -> List[Dict[str, Any]]:
        """実験のメトリクス履歴を取得"""
        
        experiment_dir = self.experiments_dir / experiment_id
        metrics_file = experiment_dir / "metrics.jsonl"
        
        if not metrics_file.exists():
            return []
        
        metrics = []
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    metrics.append(json.loads(line.strip()))
        except Exception as e:
            self.logger.error(f"メトリクス読み込みエラー: {e}")
        
        return metrics
    
    def get_experiment_logs(self, experiment_id: str) -> List[Dict[str, Any]]:
        """実験のログを取得"""
        
        experiment_dir = self.experiments_dir / experiment_id
        log_file = experiment_dir / "logs.jsonl"
        
        if not log_file.exists():
            return []
        
        logs = []
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    logs.append(json.loads(line.strip()))
        except Exception as e:
            self.logger.error(f"ログ読み込みエラー: {e}")
        
        return logs
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """実験を削除"""
        
        if experiment_id not in self.metadata["experiments"]:
            return False
        
        try:
            # ディレクトリを削除
            experiment_dir = self.experiments_dir / experiment_id
            if experiment_dir.exists():
                import shutil
                shutil.rmtree(experiment_dir)
            
            # メタデータから削除
            del self.metadata["experiments"][experiment_id]
            self._save_metadata()
            
            self.logger.info(f"実験 {experiment_id} を削除しました")
            return True
            
        except Exception as e:
            self.logger.error(f"実験削除エラー: {e}")
            return False
    
    def export_experiment(
        self, 
        experiment_id: str, 
        output_path: str
    ) -> bool:
        """実験データをエクスポート"""
        
        if experiment_id not in self.metadata["experiments"]:
            return False
        
        try:
            experiment_info = self.metadata["experiments"][experiment_id]
            metrics = self.get_experiment_metrics(experiment_id)
            logs = self.get_experiment_logs(experiment_id)
            
            export_data = {
                "experiment_info": experiment_info,
                "metrics": metrics,
                "logs": logs,
                "exported_at": datetime.now().isoformat()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"実験 {experiment_id} をエクスポートしました: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"実験エクスポートエラー: {e}")
            return False
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """実験の統計情報を取得"""
        
        experiments = list(self.metadata["experiments"].values())
        
        total_experiments = len(experiments)
        completed = len([exp for exp in experiments if exp.get("status") == "completed"])
        failed = len([exp for exp in experiments if exp.get("status") == "failed"])
        running = len([exp for exp in experiments if exp.get("status") == "running"])
        
        # モデル別統計
        model_stats = {}
        for exp in experiments:
            model_name = exp.get("model_name", "unknown")
            if model_name not in model_stats:
                model_stats[model_name] = {"total": 0, "completed": 0, "failed": 0}
            
            model_stats[model_name]["total"] += 1
            if exp.get("status") == "completed":
                model_stats[model_name]["completed"] += 1
            elif exp.get("status") == "failed":
                model_stats[model_name]["failed"] += 1
        
        # 平均実行時間
        completed_experiments = [exp for exp in experiments if exp.get("status") == "completed"]
        avg_duration = 0
        if completed_experiments:
            durations = [exp.get("duration_seconds", 0) for exp in completed_experiments]
            avg_duration = sum(durations) / len(durations)
        
        return {
            "total_experiments": total_experiments,
            "completed": completed,
            "failed": failed,
            "running": running,
            "success_rate": completed / total_experiments if total_experiments > 0 else 0,
            "average_duration_seconds": avg_duration,
            "model_stats": model_stats
        }