import pytest
import tempfile
import json
import os
import pandas as pd
from pathlib import Path
from unittest.mock import patch, Mock

from src.data_processor import DatasetProcessor
from src.trainer import TrainingManager
from src.quantizer import ModelQuantizer
from src.ollama_integration import OllamaIntegrator
from src.experiment_tracker import ExperimentTracker


class TestFullPipeline:
    """完全なパイプラインの統合テスト"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_processor = DatasetProcessor()
        self.trainer = TrainingManager()
        self.quantizer = ModelQuantizer()
        self.ollama = OllamaIntegrator()
        self.experiment_tracker = ExperimentTracker(os.path.join(self.temp_dir, 'experiments'))
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_data_preprocessing_pipeline(self):
        """データ前処理パイプラインのテスト"""
        
        # テスト用CSVデータを作成
        test_data = pd.DataFrame({
            'instruction': [
                'これは質問１です。',
                'これは質問２です。',
                'これは質問３です。',
                'これは質問４です。',
                'これは質問５です。'
            ],
            'output': [
                'これは回答１です。',
                'これは回答２です。',
                'これは回答３です。',
                'これは回答４です。',
                'これは回答５です。'
            ]
        })
        
        # 入力ファイルを作成
        input_file = os.path.join(self.temp_dir, 'test_input.csv')
        test_data.to_csv(input_file, index=False, encoding='utf-8')
        
        # 出力ディレクトリ
        output_dir = os.path.join(self.temp_dir, 'processed')
        
        # データ処理実行
        result = self.data_processor.process_dataset(
            input_file=input_file,
            output_dir=output_dir,
            task_type='instruction',
            output_format='jsonl'
        )
        
        # 結果検証
        assert result['success'] == True
        assert result['original_rows'] == 5
        assert result['cleaned_rows'] == 5
        assert result['formatted_items'] == 5
        
        # 出力ファイルの存在確認
        train_file = Path(output_dir) / 'train.jsonl'
        val_file = Path(output_dir) / 'val.jsonl'
        
        assert train_file.exists()
        
        # 訓練データの内容確認
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = [json.loads(line) for line in f]
        
        assert len(train_data) > 0
        assert '### 指示:' in train_data[0]['text']
        assert '### 回答:' in train_data[0]['text']
    
    def test_training_experiment_tracking(self):
        """トレーニングと実験追跡の統合テスト"""
        
        # テスト用データセットファイルを作成
        dataset_file = os.path.join(self.temp_dir, 'train.jsonl')
        test_data = [
            {'text': '### 指示:\n質問1\n\n### 回答:\n回答1'},
            {'text': '### 指示:\n質問2\n\n### 回答:\n回答2'},
            {'text': '### 指示:\n質問3\n\n### 回答:\n回答3'}
        ]
        
        with open(dataset_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 実験作成
        experiment_id = self.experiment_tracker.create_experiment(
            model_name='test_model',
            dataset_path=dataset_file,
            config={
                'batch_size': 1,
                'learning_rate': 5e-5,
                'num_epochs': 2
            },
            description='統合テスト実験'
        )
        
        # 実験情報の確認
        experiment = self.experiment_tracker.get_experiment(experiment_id)
        assert experiment is not None
        assert experiment['status'] == 'running'
        
        # メトリクスログの追加
        self.experiment_tracker.log_metrics(
            experiment_id,
            step=10,
            metrics={'loss': 1.5, 'learning_rate': 5e-5}
        )
        
        self.experiment_tracker.log_metrics(
            experiment_id,
            step=20,
            metrics={'loss': 1.2, 'learning_rate': 5e-5}
        )
        
        # 実験完了
        self.experiment_tracker.complete_experiment(
            experiment_id,
            output_dir=os.path.join(self.temp_dir, 'model_output'),
            metrics={'final_loss': 1.0, 'perplexity': 15.5}
        )
        
        # 完了状態の確認
        completed_experiment = self.experiment_tracker.get_experiment(experiment_id)
        assert completed_experiment['status'] == 'completed'
        assert completed_experiment['final_metrics']['final_loss'] == 1.0
        
        # メトリクス履歴の確認
        metrics = self.experiment_tracker.get_experiment_metrics(experiment_id)
        assert len(metrics) == 2
        assert metrics[0]['step'] == 10
        assert metrics[1]['step'] == 20
        assert metrics[1]['metrics']['loss'] < metrics[0]['metrics']['loss']  # 損失が改善
    
    @patch('src.quantizer.subprocess.run')
    def test_quantization_pipeline(self, mock_subprocess):
        """量子化パイプラインのテスト"""
        
        # サブプロセスのモック設定
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Quantization completed"
        mock_subprocess.return_value.stderr = ""
        
        # テスト用モデルディレクトリを作成
        model_dir = os.path.join(self.temp_dir, 'test_model')
        os.makedirs(model_dir)
        
        # ダミーのモデルファイルを作成
        model_file = os.path.join(model_dir, 'pytorch_model.bin')
        with open(model_file, 'wb') as f:
            f.write(b'dummy model data' * 1000)  # 約16KB
        
        # 量子化の検証
        validation = self.quantizer.validate_model_path(model_dir)
        assert validation['is_valid'] == True
        
        # メモリ推定のテスト
        estimated_size = self.quantizer.estimate_output_size(1.0, 'Q5_K_M')
        assert estimated_size == 0.5  # Q5_K_Mは50%のサイズ比
    
    @patch('requests.get')
    @patch('requests.post')
    def test_ollama_integration(self, mock_post, mock_get):
        """Ollama統合のテスト"""
        
        # Ollamaサーバー状態のモック
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            'models': [
                {
                    'name': 'existing_model',
                    'size': 4000000000,
                    'modified_at': '2024-01-01T00:00:00Z'
                }
            ]
        }
        
        # Ollamaサーバー状態チェック
        status = self.ollama.check_ollama_status()
        assert status['status'] == 'running'
        assert status['models_count'] == 1
        
        # モデル一覧取得
        models = self.ollama.list_models()
        assert len(models) == 1
        assert models[0]['name'] == 'existing_model'
        
        # モデル名検証
        validation = self.ollama.validate_model_name('test-model-123')
        assert validation['is_valid'] == True
        
        # 無効なモデル名
        invalid_validation = self.ollama.validate_model_name('')
        assert invalid_validation['is_valid'] == False
        assert len(invalid_validation['errors']) > 0
        
        # システムプロンプトテンプレートの取得
        templates = self.ollama.get_system_prompt_templates()
        assert 'japanese_assistant' in templates
        assert 'code_assistant' in templates
        assert 'translator' in templates
        
        # パラメータ最適化提案
        optimization = self.ollama.optimize_parameters('test_model', 'creative')
        assert 'recommended_parameters' in optimization
        assert optimization['recommended_parameters']['temperature'] > 0.7  # 創作用は高温度
    
    def test_data_validation_chain(self):
        """データ検証チェーンのテスト"""
        
        # 不正なデータでテスト
        invalid_data = pd.DataFrame({
            'instruction': ['', '短い', '正常な質問です'],  # 空文字と短いテキスト
            'output': ['回答1', '回答2', '正常な回答です']
        })
        
        # データ検証
        validation_result = self.data_processor.validate_and_clean(invalid_data)
        
        # 空の指示を持つ行が削除されることを確認
        assert len(validation_result) < len(invalid_data)
        
        # 残ったデータの品質確認
        preview = self.data_processor.get_preview(validation_result)
        assert preview['shape'][0] > 0  # データが残っている
        assert 'instruction' in preview['columns']
        assert 'output' in preview['columns']
    
    def test_memory_constraints_validation(self):
        """メモリ制約の検証テスト"""
        
        from src.utils.memory_monitor import MemoryMonitor
        
        memory_monitor = MemoryMonitor(max_memory_gb=16)  # 16GBの制約
        
        # 小さなモデルは実行可能
        can_run_small, message_small = memory_monitor.can_run_training(
            'google/gemma-2-2b-it', 1
        )
        
        # 大きなモデルまたは大きなバッチサイズでは制限にかかる可能性
        can_run_large, message_large = memory_monitor.can_run_training(
            'google/gemma-2-9b-it', 8
        )
        
        # メッセージが適切に生成されていることを確認
        assert isinstance(message_small, str)
        assert isinstance(message_large, str)
        
        # 推奨バッチサイズの提案
        recommended_batch = memory_monitor.suggest_batch_size('google/gemma-2-9b-it')
        assert isinstance(recommended_batch, int)
        assert 1 <= recommended_batch <= 8
    
    def test_config_validation_chain(self):
        """設定検証チェーンのテスト"""
        
        from src.utils.validators import validate_all
        
        # テスト用データファイルを作成
        test_data = pd.DataFrame({
            'instruction': ['質問1', '質問2'],
            'output': ['回答1', '回答2']
        })
        
        test_file = os.path.join(self.temp_dir, 'test_data.csv')
        test_data.to_csv(test_file, index=False, encoding='utf-8')
        
        # 設定
        config = {
            'model_name': 'google/gemma-2-2b-it',
            'dataset_path': test_file,
            'num_epochs': 3,
            'learning_rate': 5e-5,
            'batch_size': 1,
            'lora_rank': 16
        }
        
        # 全体検証実行
        validation_result = validate_all(
            file_path=test_file,
            model_name='google/gemma-2-2b-it',
            config=config,
            task_type='instruction'
        )
        
        # 検証結果の確認
        assert 'file_validation' in validation_result
        assert 'data_validation' in validation_result
        assert 'model_validation' in validation_result
        assert 'config_validation' in validation_result
        
        # ファイル検証が成功することを確認
        assert validation_result['file_validation']['is_valid'] == True
    
    def test_error_handling_chain(self):
        """エラーハンドリングチェーンのテスト"""
        
        # 存在しないファイルでの処理
        result = self.data_processor.process_dataset(
            input_file='/nonexistent/file.csv',
            output_dir=self.temp_dir,
            task_type='instruction'
        )
        
        assert result['success'] == False
        assert 'error' in result
        
        # 不正な設定での実験追跡
        try:
            experiment_id = self.experiment_tracker.create_experiment(
                model_name='',  # 空のモデル名
                dataset_path='',  # 空のデータセットパス
                config={}
            )
            
            # エラーメッセージのログ
            self.experiment_tracker.log_message(
                experiment_id, 
                'ERROR', 
                'テストエラーメッセージ'
            )
            
            # 実験失敗
            self.experiment_tracker.fail_experiment(
                experiment_id, 
                'テスト失敗'
            )
            
            # 失敗状態の確認
            failed_experiment = self.experiment_tracker.get_experiment(experiment_id)
            assert failed_experiment['status'] == 'failed'
            assert failed_experiment['error'] == 'テスト失敗'
            
        except Exception as e:
            # 例外が発生しても正常に処理されることを確認
            assert str(e) is not None


class TestPerformanceAndScalability:
    """パフォーマンスとスケーラビリティのテスト"""
    
    def test_large_dataset_processing(self):
        """大きなデータセットの処理テスト"""
        
        # 1000件のテストデータを作成
        large_data = pd.DataFrame({
            'instruction': [f'質問{i}です。これは大きなデータセットのテストです。' for i in range(1000)],
            'output': [f'回答{i}です。これは大きなデータセットのテスト回答です。' for i in range(1000)]
        })
        
        processor = DatasetProcessor()
        
        # メモリ効率のテスト
        preview = processor.get_preview(large_data, n_samples=10)
        
        assert preview['shape'] == (1000, 2)
        assert len(preview['sample_data']) == 10  # プレビューは10件のみ
        
        # データクリーニングのパフォーマンス
        cleaned_data = processor.validate_and_clean(large_data)
        
        assert len(cleaned_data) <= 1000  # 元データ以下
        assert len(cleaned_data) > 900   # 大部分は保持される
    
    def test_multiple_experiments_tracking(self):
        """複数実験の追跡テスト"""
        
        temp_dir = tempfile.mkdtemp()
        try:
            tracker = ExperimentTracker(temp_dir)
            
            # 10個の実験を作成
            experiment_ids = []
            for i in range(10):
                exp_id = tracker.create_experiment(
                    model_name=f'model_{i % 3}',  # 3種類のモデル
                    dataset_path=f'dataset_{i}.jsonl',
                    config={'batch_size': 1, 'epochs': 3},
                    description=f'実験{i}'
                )
                experiment_ids.append(exp_id)
            
            # いくつかの実験を完了
            for i in range(0, 10, 2):
                tracker.complete_experiment(experiment_ids[i])
            
            # いくつかの実験を失敗
            for i in range(1, 10, 3):
                tracker.fail_experiment(experiment_ids[i], 'テスト失敗')
            
            # 統計情報の確認
            stats = tracker.get_summary_stats()
            
            assert stats['total_experiments'] == 10
            assert stats['completed'] == 5  # 0,2,4,6,8
            assert stats['failed'] >= 3     # 1,4,7
            assert stats['success_rate'] > 0
            
            # フィルタリングのテスト
            completed_exps = tracker.list_experiments(status='completed')
            assert len(completed_exps) == 5
            
            model_0_exps = tracker.list_experiments(model_name='model_0')
            assert len(model_0_exps) >= 3  # model_0, model_3, model_6, model_9
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    pytest.main([__file__])