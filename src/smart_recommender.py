"""
Smart Parameter Recommender
データセットの特性を分析して最適なパラメータを推奨するモジュール
"""

import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import re


class DatasetAnalyzer:
    """データセット分析クラス"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """データセットを詳細に分析"""
        try:
            dataset_stats = {
                'total_samples': 0,
                'avg_text_length': 0,
                'max_text_length': 0,
                'min_text_length': float('inf'),
                'complexity_score': 0,
                'content_diversity': 0,
                'instruction_types': [],
                'has_specific_knowledge': False,
                'knowledge_concentration': 0
            }
            
            texts = []
            instruction_patterns = []
            
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            text = data.get('text', '')
                            texts.append(text)
                            
                            # 指示パターンを抽出
                            if '### 指示:' in text:
                                instruction = text.split('### 指示:')[1].split('### 回答:')[0].strip()
                                instruction_patterns.append(instruction)
                                
                        except json.JSONDecodeError:
                            continue
            
            if not texts:
                return dataset_stats
            
            dataset_stats['total_samples'] = len(texts)
            
            # テキスト長分析
            text_lengths = [len(text) for text in texts]
            dataset_stats['avg_text_length'] = sum(text_lengths) / len(text_lengths)
            dataset_stats['max_text_length'] = max(text_lengths)
            dataset_stats['min_text_length'] = min(text_lengths)
            
            # 複雑度スコア計算
            dataset_stats['complexity_score'] = self._calculate_complexity(texts)
            
            # 多様性スコア計算
            dataset_stats['content_diversity'] = self._calculate_diversity(instruction_patterns)
            
            # 特定知識の検出
            dataset_stats['has_specific_knowledge'] = self._detect_specific_knowledge(texts)
            dataset_stats['knowledge_concentration'] = self._calculate_knowledge_concentration(texts)
            
            # 指示タイプ分類
            dataset_stats['instruction_types'] = self._classify_instructions(instruction_patterns)
            
            return dataset_stats
            
        except Exception as e:
            self.logger.error(f"データセット分析エラー: {e}")
            return {}
    
    def _calculate_complexity(self, texts: List[str]) -> float:
        """テキストの複雑度を計算（0-1の値）"""
        if not texts:
            return 0.0
        
        total_score = 0
        for text in texts:
            # 文字種の多様性
            char_diversity = len(set(text)) / len(text) if text else 0
            
            # 文の長さの標準偏差
            sentences = text.split('。')
            if len(sentences) > 1:
                sentence_lengths = [len(s) for s in sentences if s.strip()]
                avg_len = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
                variance = sum((l - avg_len) ** 2 for l in sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
                length_complexity = min(variance / 100, 1.0)  # 正規化
            else:
                length_complexity = 0.1
            
            total_score += (char_diversity + length_complexity) / 2
        
        return min(total_score / len(texts), 1.0)
    
    def _calculate_diversity(self, instructions: List[str]) -> float:
        """指示の多様性を計算（0-1の値）"""
        if not instructions:
            return 0.0
        
        unique_instructions = set(instructions)
        return len(unique_instructions) / len(instructions)
    
    def _detect_specific_knowledge(self, texts: List[str]) -> bool:
        """特定の知識（固有名詞、数値など）があるかを検出"""
        all_text = ' '.join(texts)
        
        # 固有名詞パターン
        proper_nouns = re.findall(r'[A-Z][a-zA-Z]+|[ァ-ヶー]+|株式会社|有限会社', all_text)
        
        # 具体的な数値パターン
        numbers = re.findall(r'\d{4}年|\d+名|\d+件|\d+円', all_text)
        
        return len(proper_nouns) > 3 or len(numbers) > 2
    
    def _calculate_knowledge_concentration(self, texts: List[str]) -> float:
        """知識の集中度を計算（0-1の値、高いほど特定分野に集中）"""
        all_text = ' '.join(texts)
        
        # キーワードの出現頻度を分析
        word_freq = {}
        important_words = re.findall(r'[ァ-ヶー]{2,}|[A-Za-z]{3,}|[一-龯]{2,}', all_text)
        
        for word in important_words:
            if len(word) >= 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        if not word_freq:
            return 0.0
        
        # 上位10%の単語の出現率を計算
        sorted_words = sorted(word_freq.values(), reverse=True)
        top_count = max(1, len(sorted_words) // 10)
        top_words_freq = sum(sorted_words[:top_count])
        total_freq = sum(sorted_words)
        
        return top_words_freq / total_freq if total_freq > 0 else 0.0
    
    def _classify_instructions(self, instructions: List[str]) -> List[str]:
        """指示のタイプを分類"""
        types = []
        
        question_patterns = ['何', 'どこ', 'いつ', 'どのよう', 'なぜ', 'どちら', '?', '？']
        factual_patterns = ['について教えて', 'とは', 'の説明', 'について']
        comparison_patterns = ['比較', '違い', 'どちら', 'メリット', 'デメリット']
        
        for instruction in instructions:
            if any(pattern in instruction for pattern in question_patterns):
                types.append('質問応答')
            elif any(pattern in instruction for pattern in factual_patterns):
                types.append('事実説明')
            elif any(pattern in instruction for pattern in comparison_patterns):
                types.append('比較分析')
            else:
                types.append('一般対話')
        
        return list(set(types))


class SmartParameterRecommender:
    """スマートパラメータ推奨エンジン"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analyzer = DatasetAnalyzer()
    
    def recommend_training_parameters(self, dataset_path: str) -> Dict[str, Any]:
        """データセット分析に基づくファインチューニングパラメータ推奨"""
        
        # データセット分析実行
        stats = self.analyzer.analyze_dataset(dataset_path)
        
        if not stats:
            return self._get_default_training_params()
        
        total_samples = stats['total_samples']
        complexity = stats['complexity_score']
        diversity = stats['content_diversity']
        has_knowledge = stats['has_specific_knowledge']
        concentration = stats['knowledge_concentration']
        
        # 基本パラメータ決定ロジック
        params = {}
        
        # データサイズ別基本設定
        if total_samples <= 10:
            # 小規模：深い学習が必要
            base_epochs = 8
            base_lr = 1e-4
            base_batch = 1
            iters_multiplier = 25  # 最低200イテレーション
        elif total_samples <= 30:
            # 中小規模：バランス重視
            base_epochs = 6
            base_lr = 8e-5
            base_batch = 1
            iters_multiplier = 12
        elif total_samples <= 100:
            # 中規模：効率重視
            base_epochs = 4
            base_lr = 5e-5
            base_batch = 2
            iters_multiplier = 8
        else:
            # 大規模：保守的設定
            base_epochs = 3
            base_lr = 2e-5
            base_batch = 2
            iters_multiplier = 6
        
        # 複雑度による調整
        if complexity > 0.7:
            # 高複雑度：より慎重に
            base_epochs = max(base_epochs - 1, 3)
            base_lr *= 0.8
        elif complexity < 0.3:
            # 低複雑度：より積極的に
            base_epochs += 2
            base_lr *= 1.2
        
        # 特定知識の有無による調整
        if has_knowledge and concentration > 0.6:
            # 特定分野の知識が集中：深い学習
            base_epochs += 3
            base_lr *= 1.1
            iters_multiplier *= 1.5
        
        # 多様性による調整
        if diversity < 0.3:
            # 低多様性：過学習リスク対策
            params['lora_dropout'] = 0.15
            params['weight_decay'] = 0.02
        else:
            params['lora_dropout'] = 0.1
            params['weight_decay'] = 0.01
        
        # 最終パラメータ設定
        params.update({
            'batch_size': base_batch,
            'learning_rate': min(base_lr, 2e-4),  # 上限制限
            'num_epochs': min(base_epochs, 15),   # 上限制限
            'lora_rank': 32 if has_knowledge else 16,
            'lora_alpha': 128 if has_knowledge else 64,
            'gradient_accumulation_steps': 8,
            'warmup_steps': max(10, total_samples // 2),
            'save_steps': max(50, total_samples * 2),
            'eval_steps': max(50, total_samples * 2),
            'early_stopping_patience': 5 if total_samples > 50 else 3,
            'max_seq_length': 2048,
            'logging_steps': 10,
            'fp16': True
        })
        
        # 予想値計算
        estimated_iters = int(total_samples * base_epochs * iters_multiplier / 10)
        estimated_time = estimated_iters * 0.5  # 秒
        
        recommendation = {
            'parameters': params,
            'rationale': self._generate_rationale(stats, params),
            'estimated_iterations': estimated_iters,
            'estimated_time_minutes': estimated_time / 60,
            'confidence_level': self._calculate_confidence(stats),
            'dataset_analysis': stats
        }
        
        return recommendation
    
    def recommend_quantization_parameters(self, model_size_gb: float, use_case: str) -> Dict[str, Any]:
        """モデルサイズと用途に基づく量子化パラメータ推奨"""
        
        params = {}
        
        if use_case == "高精度重視":
            if model_size_gb < 3:
                params['method'] = 'Q5_K_M'
                params['reason'] = '小型モデルでも高精度を維持'
            else:
                params['method'] = 'Q8_0'
                params['reason'] = '大型モデルの高精度保持'
        elif use_case == "速度重視":
            params['method'] = 'Q4_K_M'
            params['reason'] = '高速推論を優先'
        elif use_case == "メモリ効率重視":
            if model_size_gb > 5:
                params['method'] = 'Q2_K'
                params['reason'] = '大幅なサイズ削減'
            else:
                params['method'] = 'Q4_K_M'
                params['reason'] = 'バランス重視の圧縮'
        else:
            # バランス重視
            params['method'] = 'Q5_K_M'
            params['reason'] = 'サイズと精度のバランス最適'
        
        return params
    
    def recommend_ollama_parameters(self, model_type: str, dataset_stats: Dict[str, Any]) -> Dict[str, Any]:
        """モデルタイプとデータセット特性に基づくOllamaパラメータ推奨"""
        
        has_knowledge = dataset_stats.get('has_specific_knowledge', False)
        concentration = dataset_stats.get('knowledge_concentration', 0)
        total_samples = dataset_stats.get('total_samples', 0)
        
        params = {
            'num_ctx': 4096,
            'repeat_penalty': 1.1
        }
        
        if has_knowledge and concentration > 0.6:
            # 特定知識重視：より決定的
            params['temperature'] = 0.3
            params['top_p'] = 0.8
            params['top_k'] = 20
        elif total_samples <= 10:
            # 小規模データセット：精密制御
            params['temperature'] = 0.5
            params['top_p'] = 0.85
            params['top_k'] = 30
        else:
            # 標準設定
            params['temperature'] = 0.7
            params['top_p'] = 0.9
            params['top_k'] = 40
        
        return params
    
    def _get_default_training_params(self) -> Dict[str, Any]:
        """デフォルト推奨パラメータ"""
        return {
            'parameters': {
                'batch_size': 1,
                'learning_rate': 5e-5,
                'num_epochs': 3,
                'lora_rank': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.1,
                'gradient_accumulation_steps': 8,
                'warmup_steps': 100,
                'weight_decay': 0.01,
                'save_steps': 500,
                'eval_steps': 500,
                'early_stopping_patience': 3,
                'max_seq_length': 2048,
                'logging_steps': 10,
                'fp16': True
            },
            'rationale': 'デフォルト安全設定',
            'estimated_iterations': 100,
            'estimated_time_minutes': 2,
            'confidence_level': 'medium'
        }
    
    def _generate_rationale(self, stats: Dict[str, Any], params: Dict[str, Any]) -> str:
        """推奨理由を生成"""
        reasons = []
        
        total_samples = stats['total_samples']
        has_knowledge = stats['has_specific_knowledge']
        complexity = stats['complexity_score']
        
        if total_samples <= 10:
            reasons.append(f"小規模データセット({total_samples}件)のため深い学習設定を採用")
        elif total_samples >= 100:
            reasons.append(f"大規模データセット({total_samples}件)のため効率重視設定を採用")
        
        if has_knowledge:
            reasons.append("特定知識を含むため、高ランクLoRA設定で記憶能力を強化")
        
        if complexity > 0.6:
            reasons.append("高複雑度データのため慎重な学習率を設定")
        elif complexity < 0.4:
            reasons.append("シンプルなデータのため積極的な学習設定を採用")
        
        return " / ".join(reasons) if reasons else "標準的なバランス型設定"
    
    def _calculate_confidence(self, stats: Dict[str, Any]) -> str:
        """推奨の信頼度を計算"""
        total_samples = stats['total_samples']
        diversity = stats['content_diversity']
        
        if total_samples >= 20 and diversity > 0.3:
            return 'high'
        elif total_samples >= 5 and diversity > 0.1:
            return 'medium'
        else:
            return 'low'