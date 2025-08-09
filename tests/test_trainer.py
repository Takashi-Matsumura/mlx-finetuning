import pytest
import tempfile
import json
import os
from unittest.mock import Mock, patch

from src.trainer import TrainingManager
from src.experiment_tracker import ExperimentTracker
from src.utils.memory_monitor import MemoryMonitor


class TestTrainingManager:
    def setup_method(self):
        self.config = {
            'batch_size': 1,
            'learning_rate': 5e-5,
            'num_epochs': 2,
            'lora_rank': 16,
            'lora_alpha': 32
        }
        self.trainer = TrainingManager(self.config)
    
    def test_init(self):
        # デフォルト設定のテスト
        assert self.trainer.config['batch_size'] == 1
        assert self.trainer.config['learning_rate'] == 5e-5
        assert self.trainer.config['num_epochs'] == 2
        
        # 初期状態のテスト
        assert self.trainer.is_training == False
        assert self.trainer.current_experiment_id is None
    
    def test_validate_training_setup(self):
        # 存在しないデータセットファイル
        result = self.trainer.validate_training_setup(
            'test_model', 
            '/nonexistent/dataset.jsonl'
        )
        
        assert result['is_valid'] == False
        assert any('データセットファイルが存在しません' in error for error in result['errors'])
    
    def test_load_dataset(self):
        # テスト用JSONLファイルを作成
        test_data = [
            {'text': 'テストデータ1'},
            {'text': 'テストデータ2'},
            {'text': 'テストデータ3'}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # データセット読み込みテスト
            loaded_data = self.trainer.load_dataset(f.name)
            
            assert len(loaded_data) == 3
            assert loaded_data[0]['text'] == 'テストデータ1'
            assert loaded_data[1]['text'] == 'テストデータ2'
            assert loaded_data[2]['text'] == 'テストデータ3'
            
            # クリーンアップ
            os.unlink(f.name)
    
    def test_load_dataset_json(self):
        # テスト用JSONファイルを作成
        test_data = [
            {'text': 'テストデータ1'},
            {'text': 'テストデータ2'}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False)
            
            # データセット読み込みテスト
            loaded_data = self.trainer.load_dataset(f.name)
            
            assert len(loaded_data) == 2
            assert loaded_data[0]['text'] == 'テストデータ1'
            
            # クリーンアップ
            os.unlink(f.name)
    
    def test_estimate_training_time(self):
        # 推定時間計算のテスト
        estimate = self.trainer.estimate_training_time('google/gemma-2-2b-it', 1000)
        
        assert 'estimated_total_hours' in estimate
        assert 'estimated_per_epoch_hours' in estimate
        assert 'samples_per_batch' in estimate
        assert 'total_batches' in estimate
        
        # 時間が正の値であることを確認
        assert estimate['estimated_total_hours'] > 0
        assert estimate['estimated_per_epoch_hours'] > 0
    
    def test_get_training_status_idle(self):
        # 非実行時のステータス
        status = self.trainer.get_training_status()
        
        assert status['is_training'] == False
        assert status['experiment_id'] is None
        assert status['status'] == 'idle'
    
    @patch('src.trainer.MLX_AVAILABLE', False)
    def test_validate_training_setup_no_mlx(self):
        # MLXが利用できない場合
        trainer = TrainingManager(self.config)
        result = trainer.validate_training_setup('test_model', 'test_dataset.jsonl')
        
        assert result['is_valid'] == False
        assert any('MLXライブラリが利用できません' in error for error in result['errors'])


class TestMemoryMonitorIntegration:
    def test_memory_monitor_integration(self):
        memory_monitor = MemoryMonitor()
        
        # メモリ情報取得のテスト
        memory_info = memory_monitor.get_memory_info()
        
        assert 'total_gb' in memory_info
        assert 'available_gb' in memory_info
        assert 'used_gb' in memory_info
        assert 'percent' in memory_info
        
        # 値が正常範囲内であることを確認
        assert memory_info['total_gb'] > 0
        assert memory_info['available_gb'] >= 0
        assert memory_info['used_gb'] >= 0
        assert 0 <= memory_info['percent'] <= 100
    
    def test_estimate_model_memory(self):
        memory_monitor = MemoryMonitor()
        
        # 各モデルのメモリ推定
        models = [
            'google/gemma-2-2b-it',
            'elyza/Llama-3-ELYZA-JP-8B',
            'google/gemma-2-9b-it'
        ]
        
        for model in models:
            memory_estimate = memory_monitor.estimate_model_memory(model)
            assert memory_estimate > 0
            
            # バッチサイズが大きいほどメモリ使用量が増えることを確認
            memory_batch_1 = memory_monitor.estimate_model_memory(model, 1)
            memory_batch_2 = memory_monitor.estimate_model_memory(model, 2)
            assert memory_batch_2 > memory_batch_1
    
    def test_can_run_training(self):
        memory_monitor = MemoryMonitor(max_memory_gb=32)
        
        # 小さなモデルでのテスト
        can_run, message = memory_monitor.can_run_training('google/gemma-2-2b-it', 1)
        
        # メッセージが文字列であることを確認
        assert isinstance(message, str)
        assert isinstance(can_run, bool)
        
        # 非常に大きなバッチサイズでは実行不可能になることを確認
        can_run_large, _ = memory_monitor.can_run_training('google/gemma-2-9b-it', 32)
        assert can_run_large == False
    
    def test_suggest_batch_size(self):
        memory_monitor = MemoryMonitor()
        
        # 推奨バッチサイズの提案
        batch_size = memory_monitor.suggest_batch_size('google/gemma-2-2b-it')
        
        assert isinstance(batch_size, int)
        assert batch_size >= 1
        assert batch_size <= 8  # 最大8に制限されている
    
    def test_get_recommendation(self):
        memory_monitor = MemoryMonitor()
        
        recommendation = memory_monitor.get_recommendation('google/gemma-2-2b-it')
        
        assert 'current_memory' in recommendation
        assert 'recommended_batch_size' in recommendation
        assert 'can_run_training' in recommendation
        assert 'message' in recommendation
        assert 'estimated_memory_gb' in recommendation
        
        # 推奨バッチサイズが正の整数であることを確認
        assert isinstance(recommendation['recommended_batch_size'], int)
        assert recommendation['recommended_batch_size'] > 0


class TestExperimentTrackerIntegration:
    def setup_method(self):
        # 一時ディレクトリを実験ディレクトリとして使用
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ExperimentTracker(self.temp_dir)
    
    def teardown_method(self):
        # クリーンアップ
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_create_experiment(self):
        # 実験作成のテスト
        experiment_id = self.tracker.create_experiment(
            model_name='test_model',
            dataset_path='test_dataset.jsonl',
            config={'batch_size': 1, 'epochs': 3},
            description='テスト実験'
        )
        
        assert isinstance(experiment_id, str)
        assert len(experiment_id) == 8  # UUIDの最初8文字
        
        # 実験情報の取得
        experiment = self.tracker.get_experiment(experiment_id)
        
        assert experiment is not None
        assert experiment['model_name'] == 'test_model'
        assert experiment['dataset_path'] == 'test_dataset.jsonl'
        assert experiment['status'] == 'running'
        assert experiment['description'] == 'テスト実験'
    
    def test_log_metrics(self):
        # 実験作成
        experiment_id = self.tracker.create_experiment(
            model_name='test_model',
            dataset_path='test_dataset.jsonl',
            config={}
        )
        
        # メトリクスログ
        self.tracker.log_metrics(
            experiment_id,
            step=10,
            metrics={'loss': 0.5, 'accuracy': 0.8}
        )
        
        # メトリクス取得
        metrics = self.tracker.get_experiment_metrics(experiment_id)
        
        assert len(metrics) == 1
        assert metrics[0]['step'] == 10
        assert metrics[0]['metrics']['loss'] == 0.5
        assert metrics[0]['metrics']['accuracy'] == 0.8
    
    def test_complete_experiment(self):
        # 実験作成
        experiment_id = self.tracker.create_experiment(
            model_name='test_model',
            dataset_path='test_dataset.jsonl',
            config={}
        )
        
        # 実験完了
        self.tracker.complete_experiment(
            experiment_id,
            output_dir='/test/output',
            metrics={'final_loss': 0.3}
        )
        
        # 完了状態の確認
        experiment = self.tracker.get_experiment(experiment_id)
        
        assert experiment['status'] == 'completed'
        assert experiment['output_dir'] == '/test/output'
        assert experiment['final_metrics']['final_loss'] == 0.3
        assert experiment['duration_seconds'] is not None
    
    def test_fail_experiment(self):
        # 実験作成
        experiment_id = self.tracker.create_experiment(
            model_name='test_model',
            dataset_path='test_dataset.jsonl',
            config={}
        )
        
        # 実験失敗
        error_message = 'テストエラー'
        self.tracker.fail_experiment(experiment_id, error_message)
        
        # 失敗状態の確認
        experiment = self.tracker.get_experiment(experiment_id)
        
        assert experiment['status'] == 'failed'
        assert experiment['error'] == error_message
        assert experiment['duration_seconds'] is not None
    
    def test_list_experiments(self):
        # 複数の実験を作成
        exp1_id = self.tracker.create_experiment(
            model_name='model1',
            dataset_path='dataset1.jsonl',
            config={}
        )
        
        exp2_id = self.tracker.create_experiment(
            model_name='model2',
            dataset_path='dataset2.jsonl',
            config={}
        )
        
        # 一つを完了
        self.tracker.complete_experiment(exp1_id)
        
        # 全実験のリスト
        all_experiments = self.tracker.list_experiments()
        assert len(all_experiments) == 2
        
        # 完了した実験のみ
        completed_experiments = self.tracker.list_experiments(status='completed')
        assert len(completed_experiments) == 1
        assert completed_experiments[0]['id'] == exp1_id
        
        # モデル別フィルタ
        model1_experiments = self.tracker.list_experiments(model_name='model1')
        assert len(model1_experiments) == 1
        assert model1_experiments[0]['id'] == exp1_id
    
    def test_get_summary_stats(self):
        # 複数の実験を作成
        exp1_id = self.tracker.create_experiment(
            model_name='model1',
            dataset_path='dataset1.jsonl',
            config={}
        )
        
        exp2_id = self.tracker.create_experiment(
            model_name='model2',
            dataset_path='dataset2.jsonl',
            config={}
        )
        
        # 一つを完了、一つを失敗
        self.tracker.complete_experiment(exp1_id)
        self.tracker.fail_experiment(exp2_id, 'テストエラー')
        
        # 統計情報取得
        stats = self.tracker.get_summary_stats()
        
        assert stats['total_experiments'] == 2
        assert stats['completed'] == 1
        assert stats['failed'] == 1
        assert stats['running'] == 0
        assert stats['success_rate'] == 0.5


if __name__ == '__main__':
    pytest.main([__file__])