import re
import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path


class DataValidator:
    def __init__(self):
        self.supported_formats = ['csv', 'json', 'jsonl', 'txt']
        self.required_columns = {
            'instruction': ['instruction', 'output'],  # inputは任意
            'chat': ['user', 'assistant'],
            'qa': ['question', 'answer'],
        }
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """ファイルの基本検証"""
        if not os.path.exists(file_path):
            return {
                'is_valid': False,
                'errors': ['ファイルが存在しません'],
                'warnings': []
            }
        
        # ファイルサイズチェック
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        # 拡張子チェック
        file_ext = Path(file_path).suffix.lower()[1:]  # .を除く
        
        errors = []
        warnings = []
        
        if file_ext not in self.supported_formats:
            errors.append(f'サポートされていないファイル形式: {file_ext}')
        
        if file_size_mb > 500:  # 500MB制限
            errors.append(f'ファイルサイズが大きすぎます: {file_size_mb:.1f}MB')
        elif file_size_mb > 100:
            warnings.append(f'ファイルサイズが大きいです: {file_size_mb:.1f}MB')
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'file_info': {
                'size_mb': file_size_mb,
                'format': file_ext
            }
        }
    
    def validate_dataset(
        self, 
        data: pd.DataFrame, 
        task_type: str = 'instruction'
    ) -> Dict[str, Any]:
        """データセットの内容検証"""
        errors = []
        warnings = []
        
        # 空データチェック
        if data.empty:
            return {
                'is_valid': False,
                'errors': ['データが空です'],
                'warnings': []
            }
        
        # 必要カラムチェック
        required_cols = self.required_columns.get(task_type, [])
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            errors.append(f'必要なカラムが不足: {missing_cols}')
        
        # データ品質チェック
        total_rows = len(data)
        
        # 空行チェック
        empty_rows = data.isnull().all(axis=1).sum()
        if empty_rows > 0:
            warnings.append(f'空行が {empty_rows} 行あります')
        
        # 重複チェック
        duplicate_rows = data.duplicated().sum()
        if duplicate_rows > 0:
            warnings.append(f'重複行が {duplicate_rows} 行あります')
        
        # テキスト長チェック
        text_columns = [col for col in data.columns if data[col].dtype == 'object']
        text_stats = {}
        
        for col in text_columns:
            if col in data.columns:
                lengths = data[col].astype(str).str.len()
                text_stats[col] = {
                    'min': lengths.min(),
                    'max': lengths.max(),
                    'mean': lengths.mean(),
                    'median': lengths.median()
                }
                
                # 短すぎるテキスト
                short_texts = (lengths < 10).sum()
                if short_texts > total_rows * 0.1:  # 10%以上
                    warnings.append(
                        f'{col}: 短いテキストが多いです ({short_texts}/{total_rows})'
                    )
                
                # 長すぎるテキスト
                long_texts = (lengths > 4000).sum()
                if long_texts > 0:
                    warnings.append(
                        f'{col}: 長いテキストがあります ({long_texts}件, >4000文字)'
                    )
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'stats': {
                'total_rows': total_rows,
                'empty_rows': empty_rows,
                'duplicate_rows': duplicate_rows,
                'text_stats': text_stats
            }
        }
    
    def validate_training_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """トレーニング設定の検証"""
        errors = []
        warnings = []
        
        # 必須パラメータ
        required_params = ['model_name', 'dataset_path', 'num_epochs']
        missing_params = [p for p in required_params if p not in config]
        
        if missing_params:
            errors.append(f'必須パラメータが不足: {missing_params}')
        
        # パラメータ範囲チェック
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if not (1e-6 <= lr <= 1e-2):
                warnings.append(f'学習率が推奨範囲外: {lr} (推奨: 1e-6 ~ 1e-2)')
        
        if 'batch_size' in config:
            bs = config['batch_size']
            if not (1 <= bs <= 32):
                warnings.append(f'バッチサイズが推奨範囲外: {bs} (推奨: 1~32)')
        
        if 'num_epochs' in config:
            epochs = config['num_epochs']
            if not (1 <= epochs <= 20):
                warnings.append(f'エポック数が推奨範囲外: {epochs} (推奨: 1~20)')
        
        if 'lora_rank' in config:
            rank = config['lora_rank']
            if not (4 <= rank <= 128):
                warnings.append(f'LoRA rankが推奨範囲外: {rank} (推奨: 4~128)')
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }


class ModelValidator:
    def __init__(self):
        self.supported_models = [
            'elyza/Llama-3-ELYZA-JP-8B',
            'google/gemma-2-9b-it',
            'google/gemma-2-2b-it',
            'meta-llama/Llama-3.1-8B-Instruct'
        ]
    
    def validate_model_name(self, model_name: str) -> Dict[str, Any]:
        """モデル名の検証"""
        errors = []
        warnings = []
        
        if not model_name:
            errors.append('モデル名が指定されていません')
        elif model_name not in self.supported_models:
            warnings.append(f'未テストのモデル: {model_name}')
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'supported_models': self.supported_models
        }
    
    def validate_model_path(self, model_path: str) -> Dict[str, Any]:
        """モデルパスの検証"""
        errors = []
        warnings = []
        
        if not os.path.exists(model_path):
            errors.append(f'モデルパスが存在しません: {model_path}')
        else:
            # ファイルサイズチェック
            size_gb = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(model_path)
                for filename in filenames
            ) / (1024**3)
            
            if size_gb > 50:  # 50GB制限
                warnings.append(f'モデルサイズが大きいです: {size_gb:.1f}GB')
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }


class ConfigValidator:
    def __init__(self):
        pass
    
    def validate_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """YAML設定ファイルの検証"""
        errors = []
        warnings = []
        
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 基本構造チェック
            required_sections = ['training', 'lora', 'data']
            missing_sections = [s for s in required_sections if s not in config]
            
            if missing_sections:
                errors.append(f'設定セクションが不足: {missing_sections}')
            
        except FileNotFoundError:
            errors.append(f'設定ファイルが見つかりません: {config_path}')
        except yaml.YAMLError as e:
            errors.append(f'YAML解析エラー: {e}')
        except Exception as e:
            errors.append(f'設定ファイル読み込みエラー: {e}')
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }


def validate_all(
    file_path: str,
    model_name: str,
    config: Dict[str, Any],
    task_type: str = 'instruction'
) -> Dict[str, Any]:
    """全体的な検証"""
    data_validator = DataValidator()
    model_validator = ModelValidator()
    
    # ファイル検証
    file_result = data_validator.validate_file(file_path)
    
    # データ読み込みと検証
    data_result = {'is_valid': False, 'errors': ['データ読み込み失敗']}
    if file_result['is_valid']:
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                data = pd.read_json(file_path)
            else:
                raise ValueError(f'未対応形式: {file_path}')
            
            data_result = data_validator.validate_dataset(data, task_type)
        except Exception as e:
            data_result = {
                'is_valid': False,
                'errors': [f'データ読み込みエラー: {e}'],
                'warnings': []
            }
    
    # モデル検証
    model_result = model_validator.validate_model_name(model_name)
    
    # 設定検証
    config_result = data_validator.validate_training_config(config)
    
    # 全体結果
    all_valid = all([
        file_result['is_valid'],
        data_result['is_valid'],
        model_result['is_valid'],
        config_result['is_valid']
    ])
    
    return {
        'is_valid': all_valid,
        'file_validation': file_result,
        'data_validation': data_result,
        'model_validation': model_result,
        'config_validation': config_result
    }