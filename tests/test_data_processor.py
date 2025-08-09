import pytest
import pandas as pd
import json
import tempfile
import os
from pathlib import Path

from src.data_processor import DatasetProcessor
from src.utils.japanese_utils import JapaneseTextProcessor


class TestJapaneseTextProcessor:
    def setup_method(self):
        self.processor = JapaneseTextProcessor()
    
    def test_normalize_text(self):
        # 全角数字の変換
        result = self.processor.normalize_text("０１２３４５６７８９")
        assert result == "0123456789"
        
        # 全角英字の変換
        result = self.processor.normalize_text("ＡＢＣＤＥＦＧＨ")
        assert result == "ABCDEFGH"
        
        # 連続スペースの正規化
        result = self.processor.normalize_text("テスト   データ")
        assert result == "テスト データ"
    
    def test_clean_text(self):
        # 制御文字の削除
        dirty_text = "こんにちは\x00世界\x0B"
        clean_text = self.processor.clean_text(dirty_text)
        assert clean_text == "こんにちは世界"
    
    def test_validate_japanese_text(self):
        # 正常なテキスト
        result = self.processor.validate_japanese_text("これは日本語のテストです。")
        assert result['is_valid'] == True
        assert len(result['errors']) == 0
        
        # 空のテキスト
        result = self.processor.validate_japanese_text("")
        assert result['is_valid'] == False
        assert "テキストが空です" in result['errors']
        
        # 短いテキスト
        result = self.processor.validate_japanese_text("短い")
        assert "テキストが短すぎます" in result['warnings']


class TestDatasetProcessor:
    def setup_method(self):
        self.processor = DatasetProcessor()
    
    def test_load_dataset_csv(self):
        # テスト用CSVファイルを作成
        test_data = pd.DataFrame({
            'instruction': ['質問1', '質問2'],
            'output': ['回答1', '回答2']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            test_data.to_csv(f.name, index=False)
            
            # データセット読み込みテスト
            df = self.processor.load_dataset(f.name)
            
            assert len(df) == 2
            assert 'instruction' in df.columns
            assert 'output' in df.columns
            
            # クリーンアップ
            os.unlink(f.name)
    
    def test_load_dataset_json(self):
        # テスト用JSONファイルを作成
        test_data = [
            {'instruction': '質問1', 'output': '回答1'},
            {'instruction': '質問2', 'output': '回答2'}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False)
            
            # データセット読み込みテスト
            df = self.processor.load_dataset(f.name)
            
            assert len(df) == 2
            assert 'instruction' in df.columns
            assert 'output' in df.columns
            
            # クリーンアップ
            os.unlink(f.name)
    
    def test_preprocess_text(self):
        # 正常なテキスト
        result = self.processor.preprocess_text("これはテストです。")
        assert result == "これはテストです。"
        
        # 空のテキスト
        result = self.processor.preprocess_text(None)
        assert result == ""
        
        # 長すぎるテキスト
        long_text = "あ" * 5000
        result = self.processor.preprocess_text(long_text)
        assert len(result) <= 4096
    
    def test_validate_and_clean(self):
        # テストデータ作成
        df = pd.DataFrame({
            'instruction': ['質問1', '質問2', '', '質問4'],
            'output': ['回答1', '回答2', '回答3', '回答4']
        })
        
        cleaned_df = self.processor.validate_and_clean(df)
        
        # 空行が削除されているかチェック
        assert len(cleaned_df) == 3  # 空のinstructionを持つ行が削除される
    
    def test_format_for_training_instruction(self):
        df = pd.DataFrame({
            'instruction': ['質問1', '質問2'],
            'output': ['回答1', '回答2']
        })
        
        formatted = self.processor.format_for_training(df, 'instruction')
        
        assert len(formatted) == 2
        assert '### 指示:' in formatted[0]['text']
        assert '### 回答:' in formatted[0]['text']
        assert '質問1' in formatted[0]['text']
        assert '回答1' in formatted[0]['text']
    
    def test_format_for_training_chat(self):
        df = pd.DataFrame({
            'user': ['ユーザー1', 'ユーザー2'],
            'assistant': ['アシスタント1', 'アシスタント2']
        })
        
        formatted = self.processor.format_for_training(df, 'chat')
        
        assert len(formatted) == 2
        assert '### ユーザー:' in formatted[0]['text']
        assert '### アシスタント:' in formatted[0]['text']
        assert 'ユーザー1' in formatted[0]['text']
        assert 'アシスタント1' in formatted[0]['text']
    
    def test_split_dataset(self):
        # テストデータ作成
        data = [{'text': f'テキスト{i}'} for i in range(100)]
        
        train, val, test = self.processor.split_dataset(data)
        
        # 分割比率のチェック
        assert len(train) == 80  # 80%
        assert len(val) == 10    # 10%
        assert len(test) == 10   # 10%
        
        # 重複チェック
        all_texts = [item['text'] for item in train + val + test]
        assert len(all_texts) == len(set(all_texts))  # 重複なし
    
    def test_get_preview(self):
        df = pd.DataFrame({
            'instruction': ['質問1', '質問2'],
            'output': ['回答1', '回答2']
        })
        
        preview = self.processor.get_preview(df)
        
        assert preview['shape'] == (2, 2)
        assert 'instruction' in preview['columns']
        assert 'output' in preview['columns']
        assert len(preview['sample_data']) == 2
    
    def test_process_dataset_full_pipeline(self):
        # テスト用データファイルを作成
        test_data = pd.DataFrame({
            'instruction': ['質問1です', '質問2です', '質問3です', '質問4です', '質問5です'],
            'output': ['回答1です', '回答2です', '回答3です', '回答4です', '回答5です']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as input_file:
            test_data.to_csv(input_file.name, index=False)
            
            with tempfile.TemporaryDirectory() as output_dir:
                # 処理実行
                result = self.processor.process_dataset(
                    input_file.name,
                    output_dir,
                    'instruction',
                    None,
                    'jsonl'
                )
                
                assert result['success'] == True
                assert result['original_rows'] == 5
                assert result['cleaned_rows'] == 5
                assert result['formatted_items'] == 5
                
                # 出力ファイルの存在確認
                train_file = Path(output_dir) / 'train.jsonl'
                assert train_file.exists()
                
                # 処理レポートの存在確認
                report_file = Path(output_dir) / 'processing_report.json'
                assert report_file.exists()
            
            # クリーンアップ
            os.unlink(input_file.name)


if __name__ == '__main__':
    pytest.main([__file__])