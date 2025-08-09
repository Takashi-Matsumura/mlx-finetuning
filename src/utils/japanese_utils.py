import re
import unicodedata
from typing import Dict, Any


class JapaneseTextProcessor:
    def __init__(self):
        self.normalize_patterns = {
            # 全角英数字を半角に変換
            'fullwidth_alpha': (r'[Ａ-Ｚａ-ｚ０-９]', self._normalize_fullwidth_alpha),
            # 全角記号を半角に変換
            'fullwidth_symbols': (r'[！-～]', self._normalize_fullwidth_symbols),
            # 連続する空白を単一スペースに
            'multiple_spaces': (r'\s+', ' '),
            # 不要な改行を削除
            'extra_newlines': (r'\n\s*\n', '\n'),
        }
    
    def normalize_text(self, text: str) -> str:
        """日本語テキストを正規化"""
        if not text:
            return ""
        
        # Unicode正規化（NFKC）
        text = unicodedata.normalize('NFKC', text)
        
        # 各パターンを適用
        for pattern_name, (pattern, func) in self.normalize_patterns.items():
            if callable(func):
                text = func(text)
            else:
                text = re.sub(pattern, func, text)
        
        # 前後の空白を削除
        return text.strip()
    
    def _normalize_fullwidth_alpha(self, text: str) -> str:
        """全角英数字を半角に変換"""
        result = ""
        for char in text:
            code = ord(char)
            if 0xFF01 <= code <= 0xFF5E:  # 全角ASCII範囲
                result += chr(code - 0xFEE0)
            else:
                result += char
        return result
    
    def _normalize_fullwidth_symbols(self, text: str) -> str:
        """全角記号を半角に変換（一部のみ）"""
        symbol_map = {
            '！': '!', '？': '?', '，': ',', '．': '.', 
            '：': ':', '；': ';', '（': '(', '）': ')',
            '［': '[', '］': ']', '｛': '{', '｝': '}',
            '＜': '<', '＞': '>', '＋': '+', '－': '-',
            '＝': '=', '＊': '*', '／': '/', '％': '%'
        }
        
        for full, half in symbol_map.items():
            text = text.replace(full, half)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """テキストのクリーニング"""
        if not text:
            return ""
        
        # 正規化
        text = self.normalize_text(text)
        
        # 制御文字を削除
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # 絵文字を削除（オプション）
        # text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
        
        return text
    
    def validate_japanese_text(self, text: str) -> dict:
        """日本語テキストの品質を検証"""
        if not text:
            return {
                'is_valid': False,
                'errors': ['テキストが空です'],
                'warnings': []
            }
        
        errors = []
        warnings = []
        
        # 長さチェック
        if len(text) < 10:
            warnings.append('テキストが短すぎます（10文字未満）')
        
        if len(text) > 4096:
            warnings.append('テキストが長すぎます（4096文字超）')
        
        # 日本語文字の割合チェック
        japanese_chars = len(re.findall(r'[ひらがなカタカナ漢字]', text))
        total_chars = len(text)
        japanese_ratio = japanese_chars / total_chars if total_chars > 0 else 0
        
        if japanese_ratio < 0.1:
            warnings.append(f'日本語文字の割合が低いです（{japanese_ratio:.1%}）')
        
        # 制御文字のチェック
        control_chars = re.findall(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', text)
        if control_chars:
            errors.append(f'制御文字が含まれています: {len(control_chars)}文字')
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'stats': {
                'length': len(text),
                'japanese_ratio': japanese_ratio,
                'lines': len(text.split('\n'))
            }
        }
    
    def format_for_training(self, text: str, format_type: str = 'instruction') -> str:
        """トレーニング用フォーマットに変換"""
        cleaned_text = self.clean_text(text)
        
        if format_type == 'instruction':
            return f"### 指示:\n{cleaned_text}\n\n### 回答:\n"
        elif format_type == 'chat':
            return f"### ユーザー:\n{cleaned_text}\n\n### アシスタント:\n"
        elif format_type == 'qa':
            return f"### 質問:\n{cleaned_text}\n\n### 回答:\n"
        else:
            return cleaned_text