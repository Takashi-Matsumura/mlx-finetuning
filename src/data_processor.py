import pandas as pd
import json
import os
import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

from .utils.japanese_utils import JapaneseTextProcessor
from .utils.validators import DataValidator


class DatasetProcessor:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.japanese_processor = JapaneseTextProcessor()
        self.validator = DataValidator()
        self.logger = logging.getLogger(__name__)
        
        # デフォルト設定
        self.default_config = {
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1,
            'min_text_length': 10,
            'max_text_length': 4096,
            'remove_duplicates': True,
            'normalize_japanese': True,
            'random_state': 42
        }
        
        # 設定をマージ
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """データセットを読み込み"""
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8')
            elif file_ext == '.json':
                df = pd.read_json(file_path, encoding='utf-8')
            elif file_ext == '.jsonl':
                lines = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        lines.append(json.loads(line.strip()))
                df = pd.DataFrame(lines)
            elif file_ext == '.txt':
                # プレーンテキストファイルの場合
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 行ごとに分割してデータフレームに
                lines = content.strip().split('\n')
                df = pd.DataFrame({'text': lines})
            else:
                raise ValueError(f'サポートされていないファイル形式: {file_ext}')
            
            self.logger.info(f'データセット読み込み完了: {len(df)} 行')
            return df
            
        except Exception as e:
            self.logger.error(f'データセット読み込みエラー: {e}')
            raise
    
    def preprocess_text(self, text: str) -> str:
        """テキストの前処理"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        if self.config.get('normalize_japanese', True):
            text = self.japanese_processor.normalize_text(text)
        
        # 長さ制限
        max_length = self.config.get('max_text_length', 4096)
        if len(text) > max_length:
            text = text[:max_length]
        
        return text.strip()
    
    def validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """データの検証とクリーニング"""
        original_count = len(df)
        
        # 空行を削除
        df = df.dropna(how='all')
        
        # テキストカラムの前処理
        text_columns = df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            df[col] = df[col].apply(self.preprocess_text)
        
        # 短すぎるテキストを削除（空文字列や非常に短いもののみ）
        min_length = self.config.get('min_text_length', 3)  # さらに短い閾値に変更
        if 'text' in df.columns:
            df = df[df['text'].str.len() >= min_length]
        elif 'instruction' in df.columns:
            # instructionとoutputの両方をチェック（より緩い条件）
            df = df[(df['instruction'].str.len() >= 3) & 
                   (df['output'].str.len() >= 3)]
        
        # 重複削除
        if self.config.get('remove_duplicates', True):
            df = df.drop_duplicates()
        
        cleaned_count = len(df)
        removed_count = original_count - cleaned_count
        
        self.logger.info(
            f'データクリーニング完了: {original_count} → {cleaned_count} 行 '
            f'({removed_count} 行削除)'
        )
        
        return df
    
    def format_for_training(
        self, 
        df: pd.DataFrame, 
        task_type: str = 'instruction',
        template: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """トレーニング用フォーマットに変換"""
        formatted_data = []
        
        if task_type == 'instruction':
            if 'instruction' in df.columns and 'output' in df.columns:
                for _, row in df.iterrows():
                    instruction = row['instruction']
                    input_text = row.get('input', '')
                    output = row['output']
                    
                    if input_text:
                        prompt = f"### 指示:\n{instruction}\n\n### 入力:\n{input_text}\n\n### 回答:\n"
                    else:
                        prompt = f"### 指示:\n{instruction}\n\n### 回答:\n"
                    
                    formatted_data.append({
                        'text': f"{prompt}{output}"
                    })
            
        elif task_type == 'chat':
            if 'user' in df.columns and 'assistant' in df.columns:
                for _, row in df.iterrows():
                    user_msg = row['user']
                    assistant_msg = row['assistant']
                    
                    text = f"### ユーザー:\n{user_msg}\n\n### アシスタント:\n{assistant_msg}"
                    formatted_data.append({'text': text})
        
        elif task_type == 'qa':
            if 'question' in df.columns and 'answer' in df.columns:
                for _, row in df.iterrows():
                    question = row['question']
                    answer = row['answer']
                    
                    text = f"### 質問:\n{question}\n\n### 回答:\n{answer}"
                    formatted_data.append({'text': text})
        
        elif task_type == 'custom' and template:
            # カスタムテンプレートを使用
            for _, row in df.iterrows():
                try:
                    text = template.format(**row.to_dict())
                    formatted_data.append({'text': text})
                except KeyError as e:
                    self.logger.warning(f'テンプレート変数不足: {e}')
                    continue
        
        else:
            # プレーンテキストとして扱う
            if 'text' in df.columns:
                formatted_data = [{'text': text} for text in df['text']]
            else:
                # 最初のテキストカラムを使用
                text_col = df.select_dtypes(include=['object']).columns[0]
                formatted_data = [{'text': text} for text in df[text_col]]
        
        self.logger.info(f'フォーマット変換完了: {len(formatted_data)} 件')
        return formatted_data
    
    def split_dataset(
        self, 
        data: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
        """データセットを訓練/検証/テスト用に分割"""
        
        if len(data) == 0:
            self.logger.warning("データが空のため分割できません")
            return [], [], []
        
        # データが少ない場合は全て訓練データとして使用
        if len(data) < 3:
            self.logger.info(f"データ数が少ないため({len(data)}件)、全て訓練データとして使用")
            return data, [], []
        
        train_ratio = self.config.get('train_split', 0.8)
        val_ratio = self.config.get('val_split', 0.1) 
        test_ratio = self.config.get('test_split', 0.1)
        
        # データ数に応じて分割方法を調整
        if len(data) < 10:
            # 少ないデータの場合は検証とテストを統合
            train_data, temp_data = train_test_split(
                data,
                train_size=max(0.6, train_ratio),  # 最低60%は訓練データに
                random_state=self.config.get('random_state', 42)
            )
            val_data = temp_data
            test_data = []
        else:
            # 通常の分割
            train_data, temp_data = train_test_split(
                data,
                train_size=train_ratio,
                random_state=self.config.get('random_state', 42)
            )
            
            # 残りを検証とテストに分割
            if len(temp_data) > 1 and val_ratio > 0 and test_ratio > 0:
                val_size = val_ratio / (val_ratio + test_ratio)
                val_data, test_data = train_test_split(
                    temp_data,
                    train_size=val_size,
                    random_state=self.config.get('random_state', 42)
                )
            elif val_ratio > 0:
                val_data = temp_data
                test_data = []
            elif test_ratio > 0:
                val_data = []
                test_data = temp_data
            else:
                val_data = []
                test_data = []
        
        self.logger.info(
            f'データ分割完了: 訓練 {len(train_data)}, '
            f'検証 {len(val_data)}, テスト {len(test_data)}'
        )
        
        return train_data, val_data, test_data
    
    def save_dataset(
        self, 
        data: List[Dict[str, str]], 
        output_path: str,
        format_type: str = 'jsonl'
    ) -> None:
        """データセットを保存"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format_type == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        elif format_type == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        elif format_type == 'csv':
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False, encoding='utf-8')
        
        else:
            raise ValueError(f'サポートされていない出力形式: {format_type}')
        
        self.logger.info(f'データセット保存完了: {output_path} ({len(data)} 件)')
    
    def process_dataset(
        self,
        input_file: str,
        output_dir: str,
        task_type: str = 'instruction',
        template: Optional[str] = None,
        output_format: str = 'jsonl'
    ) -> Dict[str, Any]:
        """データセット処理のメイン関数"""
        
        try:
            # 1. ファイル検証
            file_validation = self.validator.validate_file(input_file)
            if not file_validation['is_valid']:
                return {
                    'success': False,
                    'error': 'ファイル検証失敗',
                    'details': file_validation
                }
            
            # 2. データ読み込み
            df = self.load_dataset(input_file)
            
            # 3. データ検証
            data_validation = self.validator.validate_dataset(df, task_type)
            
            # 4. データクリーニング
            df_clean = self.validate_and_clean(df)
            
            # 5. フォーマット変換
            formatted_data = self.format_for_training(df_clean, task_type, template)
            
            # 6. データ分割
            train_data, val_data, test_data = self.split_dataset(formatted_data)
            
            # 7. 保存
            os.makedirs(output_dir, exist_ok=True)
            
            self.save_dataset(
                train_data, 
                os.path.join(output_dir, f'train.{output_format}'),
                output_format
            )
            
            if val_data:
                self.save_dataset(
                    val_data, 
                    os.path.join(output_dir, f'val.{output_format}'),
                    output_format
                )
            
            if test_data:
                self.save_dataset(
                    test_data, 
                    os.path.join(output_dir, f'test.{output_format}'),
                    output_format
                )
            
            # 処理レポート生成（JSON互換型に変換）
            report = {
                'success': True,
                'input_file': input_file,
                'output_dir': output_dir,
                'task_type': task_type,
                'original_rows': int(len(df)),
                'cleaned_rows': int(len(df_clean)),
                'formatted_items': int(len(formatted_data)),
                'splits': {
                    'train': int(len(train_data)),
                    'val': int(len(val_data)),
                    'test': int(len(test_data))
                },
                'validation': self._convert_to_json_serializable(data_validation),
                'config': self._convert_to_json_serializable(self.config)
            }
            
            # レポート保存
            with open(os.path.join(output_dir, 'processing_report.json'), 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            return report
            
        except Exception as e:
            self.logger.error(f'データセット処理エラー: {e}')
            return {
                'success': False,
                'error': str(e),
                'details': None
            }
    
    def get_preview(self, df: pd.DataFrame, n_samples: int = 5) -> Dict[str, Any]:
        """データセットのプレビューを生成"""
        return {
            'shape': (int(df.shape[0]), int(df.shape[1])),
            'columns': list(df.columns),
            'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            'sample_data': self._convert_to_json_serializable(df.head(n_samples).to_dict('records')),
            'null_counts': {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
            'text_stats': self._get_text_stats(df)
        }
    
    def _get_text_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """テキスト統計情報を取得"""
        text_columns = df.select_dtypes(include=['object']).columns
        stats = {}
        
        for col in text_columns:
            lengths = df[col].astype(str).str.len()
            stats[col] = {
                'min_length': int(lengths.min()),
                'max_length': int(lengths.max()),
                'mean_length': float(lengths.mean()),
                'median_length': float(lengths.median())
            }
        
        return stats
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """オブジェクトをJSON互換の形式に変換"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_json_serializable(item) for item in obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj