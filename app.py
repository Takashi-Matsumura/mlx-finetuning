import streamlit as st
import yaml
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
from datetime import datetime

# プロジェクトのソースコードをインポート
from src.data_processor import DatasetProcessor
from src.trainer import TrainingManager
from src.quantizer import ModelQuantizer
from src.ollama_integration import OllamaIntegrator
from src.experiment_tracker import ExperimentTracker
from src.utils.memory_monitor import MemoryMonitor
from src.utils.validators import validate_all

# ページ設定
st.set_page_config(
    page_title="LLM ファインチューニング アプリ",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# セッション状態の初期化
if 'training_manager' not in st.session_state:
    st.session_state['training_manager'] = None
if 'current_experiment_id' not in st.session_state:
    st.session_state['current_experiment_id'] = None


def load_config():
    """設定ファイルを読み込み"""
    try:
        with open('./config/default_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        with open('./config/models.yaml', 'r', encoding='utf-8') as f:
            models_config = yaml.safe_load(f)
        
        return config, models_config
    except Exception as e:
        st.error(f"設定ファイル読み込みエラー: {e}")
        return {}, {}


def sidebar_menu():
    """サイドバーメニュー"""
    with st.sidebar:
        st.title("🚀 LLM ファインチューニング")
        st.markdown("---")
        
        menu_options = [
            ("🏠 ホーム", "home"),
            ("📊 データセット準備", "dataset"),
            ("🚀 ファインチューニング", "training"),
            ("📦 量子化", "quantization"),
            ("🤖 Ollama統合", "ollama"),
            ("📈 実験履歴", "experiments"),
            ("⚙️ 設定", "settings")
        ]
        
        selected_page = st.radio(
            "メニュー",
            options=[option[1] for option in menu_options],
            format_func=lambda x: next(option[0] for option in menu_options if option[1] == x),
            key="sidebar_menu"
        )
        
        st.markdown("---")
        
        # システム情報
        memory_monitor = MemoryMonitor()
        memory_info = memory_monitor.get_memory_info()
        
        st.subheader("💻 システム情報")
        st.metric("使用可能メモリ", f"{memory_info['available_gb']:.1f} GB")
        st.metric("メモリ使用率", f"{memory_info['percent']:.1f}%")
        
        return selected_page


def home_page():
    """ホームページ"""
    st.title("🚀 LLM ファインチューニング アプリ")
    st.markdown("MacBook Air M4用の日本語LLMファインチューニング自動化ツール")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 データセット処理", "自動化", help="CSV/JSONファイルの自動前処理")
        
    with col2:
        st.metric("🚀 ファインチューニング", "MLX対応", help="Apple Silicon最適化")
        
    with col3:
        st.metric("📦 量子化", "GGUF変換", help="Ollama統合対応")
    
    st.markdown("---")
    
    # クイックスタートガイド
    st.subheader("📋 クイックスタート")
    
    with st.expander("1️⃣ データセット準備", expanded=True):
        st.markdown("""
        - CSV/JSONファイルをアップロード
        - 自動的な日本語正規化
        - 品質検証とデータ分割
        """)
    
    with st.expander("2️⃣ ファインチューニング実行"):
        st.markdown("""
        - モデル選択（ELYZA/Gemma2）
        - LoRAパラメータの設定
        - リアルタイム進捗監視
        """)
    
    with st.expander("3️⃣ 量子化と統合"):
        st.markdown("""
        - GGUF形式への変換
        - 量子化処理（Q4/Q5/Q8）
        - Ollamaへの自動登録
        """)
    
    # 最近の実験
    st.subheader("📈 最近の実験")
    experiment_tracker = ExperimentTracker()
    recent_experiments = experiment_tracker.list_experiments(limit=5)
    
    if recent_experiments:
        df = pd.DataFrame([
            {
                'ID': exp['id'][:8],
                'モデル': exp['model_name'].split('/')[-1],
                'ステータス': exp['status'],
                '作成日時': exp['created_at'][:19]
            }
            for exp in recent_experiments
        ])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("まだ実験が実行されていません")


def dataset_page():
    """データセット準備ページ"""
    st.title("📊 データセット準備")
    
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "データファイルをアップロード",
        type=['csv', 'json', 'jsonl', 'txt'],
        help="CSV、JSON、JSONL、TXTファイルに対応"
    )
    
    if uploaded_file is not None:
        # 一時ファイルに保存
        temp_path = f"./data/raw/{uploaded_file.name}"
        os.makedirs("./data/raw", exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"ファイルアップロード完了: {uploaded_file.name}")
        
        # データプレビュー
        try:
            processor = DatasetProcessor()
            df = processor.load_dataset(temp_path)
            
            st.subheader("📋 データプレビュー")
            preview_info = processor.get_preview(df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("行数", preview_info['shape'][0])
                st.metric("列数", preview_info['shape'][1])
            
            with col2:
                st.write("**カラム情報**")
                for col, dtype in preview_info['dtypes'].items():
                    st.write(f"• {col}: {dtype}")
            
            # サンプルデータ表示
            st.write("**サンプルデータ**")
            st.dataframe(df.head(), use_container_width=True)
            
            # 処理オプション
            st.subheader("⚙️ 処理オプション")
            
            col1, col2 = st.columns(2)
            
            with col1:
                normalize_text = st.checkbox("日本語正規化", value=True)
                remove_duplicates = st.checkbox("重複削除", value=True)
                task_type = st.selectbox(
                    "タスクタイプ",
                    options=['instruction', 'chat', 'qa', 'custom'],
                    format_func=lambda x: {
                        'instruction': '指示実行',
                        'chat': 'チャット',
                        'qa': '質問応答',
                        'custom': 'カスタム'
                    }[x]
                )
            
            with col2:
                train_ratio = st.slider("訓練データ比率", 0.5, 0.9, 0.8)
                min_length = st.number_input("最小文字数", 10, 1000, 50)
                output_format = st.selectbox("出力形式", ['jsonl', 'json', 'csv'])
            
            # カスタムテンプレート
            custom_template = None
            if task_type == 'custom':
                st.subheader("📝 カスタムテンプレート")
                custom_template = st.text_area(
                    "テンプレート（{column_name}で列を参照）",
                    value="### 指示:\n{instruction}\n\n### 回答:\n{output}",
                    height=100
                )
            
            # 処理実行ボタン
            if st.button("🚀 データセット処理開始", type="primary"):
                with st.spinner("データセット処理中..."):
                    config = {
                        'train_split': train_ratio,
                        'val_split': (1 - train_ratio) * 0.5,
                        'test_split': (1 - train_ratio) * 0.5,
                        'min_text_length': min_length,
                        'remove_duplicates': remove_duplicates,
                        'normalize_japanese': normalize_text
                    }
                    
                    processor_with_config = DatasetProcessor(config)
                    
                    output_dir = f"./data/processed/{uploaded_file.name.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    result = processor_with_config.process_dataset(
                        temp_path,
                        output_dir,
                        task_type,
                        custom_template,
                        output_format
                    )
                    
                    if result['success']:
                        st.success("✅ データセット処理完了！")
                        
                        # 結果表示
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("元データ", f"{result['original_rows']} 行")
                        with col2:
                            st.metric("処理後", f"{result['cleaned_rows']} 行")
                        with col3:
                            st.metric("フォーマット後", f"{result['formatted_items']} 件")
                        
                        # 分割結果
                        st.write("**データ分割結果**")
                        splits_df = pd.DataFrame([
                            {'分割': '訓練', '件数': result['splits']['train']},
                            {'分割': '検証', '件数': result['splits']['val']},
                            {'分割': 'テスト', '件数': result['splits']['test']}
                        ])
                        st.dataframe(splits_df, use_container_width=True)
                        
                        # 出力パス表示
                        st.info(f"📁 出力ディレクトリ: {output_dir}")
                        
                    else:
                        st.error(f"❌ 処理エラー: {result['error']}")
                        
        except Exception as e:
            st.error(f"データ読み込みエラー: {e}")


def training_page():
    """ファインチューニングページ"""
    st.title("🚀 ファインチューニング")
    
    # 設定読み込み
    config, models_config = load_config()
    
    # モデル選択
    st.subheader("🤖 モデル選択")
    
    model_options = {}
    if 'base_models' in models_config:
        for key, model_info in models_config['base_models'].items():
            model_options[model_info['name']] = model_info['display_name']
    
    selected_model = st.selectbox(
        "ベースモデル",
        options=list(model_options.keys()),
        format_func=lambda x: model_options.get(x, x)
    )
    
    if selected_model and selected_model in [info['name'] for info in models_config.get('base_models', {}).values()]:
        model_info = next(info for info in models_config['base_models'].values() if info['name'] == selected_model)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("モデルサイズ", f"{model_info['size_gb']} GB")
        with col2:
            st.metric("コンテキスト長", f"{model_info['context_length']:,}")
        with col3:
            st.metric("推奨バッチサイズ", model_info['recommended_batch_size'])
        
        st.info(f"📝 {model_info['description']}")
    
    # データセット選択
    st.subheader("📊 データセット選択")
    
    processed_data_dir = Path("./data/processed")
    if processed_data_dir.exists():
        dataset_dirs = [d for d in processed_data_dir.iterdir() if d.is_dir()]
        
        if dataset_dirs:
            selected_dataset_dir = st.selectbox(
                "処理済みデータセット",
                options=dataset_dirs,
                format_func=lambda x: x.name
            )
            
            # データセット情報表示
            if selected_dataset_dir:
                train_file = selected_dataset_dir / "train.jsonl"
                if train_file.exists():
                    with open(train_file, 'r') as f:
                        train_count = sum(1 for _ in f)
                    st.info(f"📊 訓練データ: {train_count:,} 件")
        else:
            st.warning("処理済みデータセットがありません。先にデータセット準備を行ってください。")
            selected_dataset_dir = None
    else:
        st.warning("データディレクトリが存在しません。")
        selected_dataset_dir = None
    
    # ハイパーパラメータ設定
    st.subheader("⚙️ ハイパーパラメータ設定")
    
    with st.expander("基本設定", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            batch_size = st.selectbox("バッチサイズ", [1, 2, 4], index=0)
            learning_rate = st.select_slider(
                "学習率",
                options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
                value=5e-5,
                format_func=lambda x: f"{x:.0e}"
            )
            num_epochs = st.slider("エポック数", 1, 10, 3)
        
        with col2:
            lora_rank = st.slider("LoRA Rank", 4, 64, 16)
            lora_alpha = st.slider("LoRA Alpha", 8, 128, 32)
            lora_dropout = st.slider("LoRA Dropout", 0.0, 0.3, 0.1)
    
    with st.expander("詳細設定"):
        col1, col2 = st.columns(2)
        
        with col1:
            gradient_accumulation_steps = st.number_input("勾配蓄積ステップ", 1, 32, 8)
            warmup_steps = st.number_input("ウォームアップステップ", 0, 1000, 100)
            weight_decay = st.number_input("重み減衰", 0.0, 0.1, 0.01, format="%.3f")
        
        with col2:
            save_steps = st.number_input("保存間隔", 100, 2000, 500)
            eval_steps = st.number_input("評価間隔", 100, 2000, 500)
            early_stopping_patience = st.number_input("早期停止待機", 1, 10, 3)
    
    # メモリチェック
    if selected_model:
        memory_monitor = MemoryMonitor()
        can_run, message = memory_monitor.can_run_training(selected_model, batch_size)
        
        if can_run:
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")
            suggested_batch_size = memory_monitor.suggest_batch_size(selected_model)
            st.info(f"💡 推奨バッチサイズ: {suggested_batch_size}")
    
    # トレーニング実行
    st.subheader("🚀 トレーニング実行")
    
    if selected_dataset_dir and selected_model:
        train_file_path = str(selected_dataset_dir / "train.jsonl")
        
        if st.button("🚀 ファインチューニング開始", type="primary"):
            # トレーニング設定
            training_config = {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'lora_rank': lora_rank,
                'lora_alpha': lora_alpha,
                'lora_dropout': lora_dropout,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'warmup_steps': warmup_steps,
                'weight_decay': weight_decay,
                'save_steps': save_steps,
                'eval_steps': eval_steps,
                'early_stopping_patience': early_stopping_patience
            }
            
            # トレーニングマネージャーを初期化
            trainer = TrainingManager(training_config)
            st.session_state['training_manager'] = trainer
            
            # セッション状態でトレーニング状況を管理
            if 'training_progress' not in st.session_state:
                st.session_state['training_progress'] = 0
            if 'training_status' not in st.session_state:
                st.session_state['training_status'] = "準備中..."
            
            # プログレスバーとステータス表示
            progress_bar = st.progress(st.session_state['training_progress'])
            status_text = st.empty()
            status_text.text(st.session_state['training_status'])
            
            # セッション状態を更新する関数（スレッドセーフ）
            def update_progress(progress):
                st.session_state['training_progress'] = progress
            
            def update_status(status):
                st.session_state['training_status'] = status
            
            # トレーニング開始（コールバック無しで実行）
            try:
                experiment_id = trainer.start_training(
                    selected_model,
                    train_file_path,
                    progress_callback=None,  # 無効化
                    status_callback=None     # 無効化
                )
                
                st.session_state['current_experiment_id'] = experiment_id
                st.success(f"✅ トレーニングを開始しました（実験ID: {experiment_id}）")
                
            except Exception as e:
                st.error(f"❌ トレーニング開始エラー: {e}")
    
    # 現在のトレーニング状況
    if st.session_state.get('training_manager'):
        status = st.session_state['training_manager'].get_training_status()
        
        if status['is_training']:
            # 実験の状態をチェック
            experiment_id = status['experiment_id']
            if experiment_id:
                experiment_info = ExperimentTracker().get_experiment(experiment_id)
                
                # 実験が完了している場合はトレーニング状態をリセット
                if experiment_info and experiment_info.get('status') == 'completed':
                    st.session_state['training_manager'].is_training = False
                    st.session_state['training_manager'].current_experiment_id = None
                    st.success(f"🎉 トレーニングが完了しました！実験ID: {experiment_id}")
                    st.info("📈 実験履歴ページで詳細な結果を確認できます")
                    st.rerun()
                
                elif experiment_info and experiment_info.get('status') == 'failed':
                    st.session_state['training_manager'].is_training = False
                    st.session_state['training_manager'].current_experiment_id = None
                    st.error(f"❌ トレーニングが失敗しました。実験ID: {experiment_id}")
                    st.rerun()
                
                else:
                    # 実行中表示
                    with st.container():
                        st.info(f"🔄 トレーニング実行中... (実験ID: {experiment_id})")
                        
                        # 実験の進捗をリアルタイム表示
                        if experiment_info:
                            col1, col2 = st.columns(2)
                            with col1:
                                duration = experiment_info.get('duration_seconds', 0)
                                duration_str = f"{duration:.1f}秒" if duration is not None else "実行中"
                                st.metric("実行時間", duration_str)
                            with col2:
                                exp_status = experiment_info.get('status', 'running')
                                status_emoji = {"running": "🔄", "completed": "✅", "failed": "❌"}
                                st.metric("ステータス", f"{status_emoji.get(exp_status, '❓')} {exp_status}")
                        
                        # 停止ボタン
                        if st.button("⏹️ トレーニング停止"):
                            st.session_state['training_manager'].stop_current_training()
                            st.warning("⏹️ トレーニング停止要求を送信しました")
                        
                        # 自動リフレッシュ
                        if st.button("🔄 状況更新"):
                            st.rerun()
        else:
            # トレーニング非実行時の表示
            if st.session_state.get('current_experiment_id'):
                last_experiment_id = st.session_state['current_experiment_id']
                experiment_info = ExperimentTracker().get_experiment(last_experiment_id)
                
                if experiment_info:
                    if experiment_info.get('status') == 'completed':
                        st.success(f"✅ 最新のトレーニングが完了しました！")
                        st.info(f"実験ID: {last_experiment_id}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("📈 実験詳細を見る"):
                                st.switch_page("experiments")
                        with col2:
                            if st.button("📦 量子化に進む"):
                                st.switch_page("quantization")


def quantization_page():
    """量子化ページ"""
    st.title("📦 量子化")
    
    quantizer = ModelQuantizer()
    
    # llama.cpp状態チェック
    st.subheader("🔧 llama.cpp 状態")
    
    if quantizer.check_llama_cpp():
        st.success("✅ llama.cpp が利用可能です")
    else:
        st.error("❌ llama.cpp が見つかりません")
        st.markdown("""
        **セットアップが必要です:**
        ```bash
        ./setup.sh
        ```
        """)
        return
    
    # 量子化方法の説明
    st.subheader("📊 量子化方法")
    
    quant_info = quantizer.get_quantization_info()
    methods_df = pd.DataFrame([
        {
            '方法': method,
            '説明': info['description'],
            'サイズ比': f"{info['size_ratio']*100:.0f}%",
            '品質': info['quality']
        }
        for method, info in quant_info['available_methods'].items()
    ])
    
    st.dataframe(methods_df, use_container_width=True)
    
    # モデル選択
    st.subheader("🤖 モデル選択")
    
    # ファインチューニング済みモデルを検索
    finetuned_dir = Path("./models/finetuned")
    available_models = []
    
    if finetuned_dir.exists():
        for model_dir in finetuned_dir.iterdir():
            if model_dir.is_dir():
                # MLXファインチューニング結果を検索
                adapters_file = model_dir / "adapters.safetensors"
                mlx_model_dirs = list(model_dir.glob("mlx_model_*"))
                
                if adapters_file.exists() and mlx_model_dirs:
                    # MLXモデルとLoRAアダプターの組み合わせ
                    mlx_model_path = mlx_model_dirs[0]  # 最初のMLXモデルディレクトリを使用
                    model_info = {
                        'path': str(mlx_model_path),
                        'adapters': str(adapters_file),
                        'experiment_id': model_dir.name,
                        'display_name': f'Gemma-2-2b-it + LoRA ({model_dir.name[:8]})'
                    }
                    available_models.append(model_info)
                
                # 従来のfinal_modelディレクトリもサポート
                final_model_path = model_dir / "final_model"
                if final_model_path.exists():
                    model_info = {
                        'path': str(final_model_path),
                        'adapters': None,
                        'experiment_id': model_dir.name,
                        'display_name': f'Traditional Model ({model_dir.name[:8]})'
                    }
                    available_models.append(model_info)
    
    if available_models:
        selected_model_info = st.selectbox(
            "ファインチューニング済みモデル",
            options=available_models,
            format_func=lambda x: x['display_name']
        )
        selected_model_path = selected_model_info['path']
        
        # モデル情報を表示
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**ベースモデル**: {selected_model_info['path']}")
        with col2:
            if selected_model_info['adapters']:
                st.info(f"**LoRAアダプター**: {selected_model_info['adapters']}")
            else:
                st.info("**モデル**: 従来形式")
    else:
        st.warning("ファインチューニング済みモデルが見つかりません")
        st.info("代わりにモデルパスを直接指定することもできます")
        selected_model_path = st.text_input("モデルパス", placeholder="/path/to/model")
    
    if selected_model_path:
        # モデル検証
        validation_result = quantizer.validate_model_path(selected_model_path)
        
        if validation_result['is_valid']:
            st.success(f"✅ モデル検証OK (サイズ: {validation_result['model_size_gb']:.1f} GB)")
        else:
            for error in validation_result['errors']:
                st.error(f"❌ {error}")
            for warning in validation_result['warnings']:
                st.warning(f"⚠️ {warning}")
    
    # 量子化設定
    st.subheader("⚙️ 量子化設定")
    
    col1, col2 = st.columns(2)
    
    with col1:
        quantization_method = st.selectbox(
            "量子化方法",
            options=list(quant_info['available_methods'].keys()),
            index=1  # Q5_K_M をデフォルト
        )
        
        output_name = st.text_input(
            "出力ファイル名",
            value=f"model-{quantization_method.lower()}",
            help="拡張子(.gguf)は自動で追加されます"
        )
    
    with col2:
        if selected_model_path and validation_result.get('is_valid'):
            input_size = validation_result['model_size_gb']
            estimated_size = quantizer.estimate_output_size(input_size, quantization_method)
            
            st.metric("入力サイズ", f"{input_size:.1f} GB")
            st.metric("推定出力サイズ", f"{estimated_size:.1f} GB")
            st.metric("圧縮率", f"{(estimated_size/input_size)*100:.0f}%")
    
    # 量子化実行
    if selected_model_path and st.button("🚀 量子化開始", type="primary"):
        output_dir = "./models/quantized"
        os.makedirs(output_dir, exist_ok=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress):
            progress_bar.progress(progress)
        
        def update_status(status):
            status_text.text(status)
        
        with st.spinner("量子化処理中..."):
            result = quantizer.full_quantization_pipeline(
                selected_model_path,
                output_dir,
                quantization_method,
                progress_callback=update_progress,
                status_callback=update_status
            )
            
            if result['success']:
                st.success("✅ 量子化完了！")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("GGUF サイズ", f"{result.get('gguf_size_mb', 0)/1024:.1f} GB")
                with col2:
                    st.metric("量子化後サイズ", f"{result.get('quantized_size_mb', 0)/1024:.1f} GB")
                with col3:
                    st.metric("圧縮率", f"{result.get('compression_ratio', 1)*100:.0f}%")
                
                st.info(f"📁 出力ファイル: {result['quantized_path']}")
                
                # Ollama統合の提案
                if st.button("🤖 Ollamaに登録"):
                    st.session_state['quantized_model_path'] = result['quantized_path']
                    st.switch_page("Ollama統合")
                    
            else:
                st.error(f"❌ 量子化エラー: {result['error']}")


def ollama_page():
    """Ollama統合ページ"""
    st.title("🤖 Ollama 統合")
    
    integrator = OllamaIntegrator()
    
    # Ollama状態チェック
    st.subheader("🔧 Ollama サーバー状態")
    
    status = integrator.check_ollama_status()
    
    if status['status'] == 'running':
        st.success(f"✅ Ollama サーバー稼働中 ({status['url']})")
        st.info(f"📊 登録済みモデル: {status['models_count']} 個")
    elif status['status'] == 'not_running':
        st.error("❌ Ollama サーバーが起動していません")
        if st.button("🚀 Ollama サーバー起動"):
            with st.spinner("サーバー起動中..."):
                if integrator.start_ollama_server():
                    st.success("✅ サーバー起動完了！")
                    st.rerun()
                else:
                    st.error("❌ サーバー起動に失敗しました")
        return
    else:
        st.error(f"❌ Ollama エラー: {status['error']}")
        return
    
    # 既存モデル一覧
    st.subheader("📋 登録済みモデル")
    
    models = integrator.list_models()
    if models:
        models_df = pd.DataFrame([
            {
                'モデル名': model['name'],
                'サイズ': f"{model.get('size', 0) / (1024**3):.1f} GB",
                '更新日': model.get('modified_at', 'N/A')[:19] if model.get('modified_at') else 'N/A'
            }
            for model in models
        ])
        st.dataframe(models_df, use_container_width=True)
    else:
        st.info("登録済みモデルがありません")
    
    # 新しいモデルの登録
    st.subheader("➕ 新しいモデルを登録")
    
    # 量子化済みモデルを検索
    quantized_dir = Path("./models/quantized")
    available_gguf = []
    
    if quantized_dir.exists():
        available_gguf = list(quantized_dir.glob("*.gguf"))
    
    # セッション状態から量子化済みモデルパスを取得
    if 'quantized_model_path' in st.session_state:
        available_gguf.insert(0, Path(st.session_state['quantized_model_path']))
    
    if available_gguf:
        selected_gguf = st.selectbox(
            "GGUFファイル",
            options=available_gguf,
            format_func=lambda x: x.name
        )
    else:
        st.warning("量子化済みGGUFファイルが見つかりません")
        selected_gguf = st.text_input("GGUFファイルパス", placeholder="/path/to/model.gguf")
        if selected_gguf:
            selected_gguf = Path(selected_gguf)
    
    if selected_gguf:
        # モデル名設定
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.text_input(
                "モデル名",
                value=selected_gguf.stem if hasattr(selected_gguf, 'stem') else "",
                help="英数字、ハイフン、アンダースコアのみ使用可能"
            )
            
            # モデル名検証
            if model_name:
                validation = integrator.validate_model_name(model_name)
                if not validation['is_valid']:
                    for error in validation['errors']:
                        st.error(f"❌ {error}")
                for warning in validation['warnings']:
                    st.warning(f"⚠️ {warning}")
        
        with col2:
            # システムプロンプトテンプレート選択
            templates = integrator.get_system_prompt_templates()
            template_names = list(templates.keys())
            
            selected_template = st.selectbox(
                "システムプロンプト",
                options=template_names,
                format_func=lambda x: templates[x]['name']
            )
        
        # システムプロンプト編集
        if selected_template == 'custom':
            system_prompt = st.text_area(
                "カスタムシステムプロンプト",
                height=150,
                placeholder="システムプロンプトを入力してください..."
            )
        else:
            system_prompt = st.text_area(
                "システムプロンプト",
                value=templates[selected_template]['prompt'],
                height=150
            )
        
        # パラメータ設定
        st.subheader("⚙️ パラメータ設定")
        
        # 用途別プリセット
        use_cases = ['general', 'creative', 'precise', 'translation', 'coding']
        use_case = st.selectbox(
            "用途",
            options=use_cases,
            format_func=lambda x: {
                'general': '一般用途',
                'creative': '創作',
                'precise': '正確性重視',
                'translation': '翻訳',
                'coding': 'コーディング'
            }[x]
        )
        
        # パラメータ最適化提案を取得
        optimization = integrator.optimize_parameters(model_name, use_case)
        recommended = optimization['recommended_parameters']
        
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider(
                "Temperature",
                0.0, 2.0, recommended['temperature'],
                help="応答のランダム性"
            )
            top_p = st.slider(
                "Top P",
                0.0, 1.0, recommended['top_p'],
                help="核サンプリング"
            )
        
        with col2:
            repeat_penalty = st.slider(
                "Repeat Penalty",
                1.0, 2.0, recommended['repeat_penalty'],
                help="繰り返し抑制"
            )
            num_ctx = st.selectbox(
                "Context Length",
                options=[2048, 4096, 8192, 16384],
                index=1,
                help="コンテキスト長"
            )
        
        # モデル登録実行
        if st.button("🤖 モデル登録", type="primary", disabled=not model_name):
            parameters = {
                'temperature': temperature,
                'top_p': top_p,
                'repeat_penalty': repeat_penalty,
                'num_ctx': num_ctx
            }
            
            with st.spinner("モデル登録中..."):
                result = integrator.create_model_from_gguf(
                    str(selected_gguf),
                    model_name,
                    system_prompt,
                    parameters
                )
                
                if result['success']:
                    st.success("✅ モデル登録完了！")
                    
                    # テスト実行
                    st.subheader("🧪 モデルテスト")
                    test_result = integrator.test_model(model_name)
                    
                    if test_result['success']:
                        st.success("✅ テスト成功")
                        st.write("**テストプロンプト:**", test_result['prompt'])
                        st.write("**応答:**", test_result['response'])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("生成トークン数", test_result.get('eval_count', 0))
                        with col2:
                            duration_ms = test_result.get('eval_duration', 0) / 1_000_000
                            st.metric("応答時間", f"{duration_ms:.0f} ms")
                    else:
                        st.error(f"❌ テスト失敗: {test_result['error']}")
                    
                    # セッション状態をクリア
                    if 'quantized_model_path' in st.session_state:
                        del st.session_state['quantized_model_path']
                        
                else:
                    st.error(f"❌ 登録エラー: {result['error']}")
    
    # モデル管理
    if models:
        st.subheader("🗑️ モデル管理")
        
        model_to_delete = st.selectbox(
            "削除するモデル",
            options=[''] + [model['name'] for model in models],
            format_func=lambda x: "選択してください" if x == '' else x
        )
        
        if model_to_delete and st.button("🗑️ モデル削除", type="secondary"):
            if st.checkbox("削除を確認します"):
                result = integrator.delete_model(model_to_delete)
                
                if result['success']:
                    st.success(f"✅ {result['message']}")
                    st.rerun()
                else:
                    st.error(f"❌ {result['error']}")


def experiments_page():
    """実験履歴ページ"""
    st.title("📈 実験履歴")
    
    experiment_tracker = ExperimentTracker()
    
    # 統計情報
    stats = experiment_tracker.get_summary_stats()
    
    st.subheader("📊 統計情報")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("総実験数", stats['total_experiments'])
    with col2:
        st.metric("完了", stats['completed'])
    with col3:
        st.metric("失敗", stats['failed'])
    with col4:
        st.metric("実行中", stats['running'])
    
    if stats['total_experiments'] > 0:
        success_rate = stats['success_rate'] * 100
        avg_duration_hours = stats['average_duration_seconds'] / 3600
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("成功率", f"{success_rate:.1f}%")
        with col2:
            st.metric("平均実行時間", f"{avg_duration_hours:.1f} 時間")
    
    # 実験フィルタ
    st.subheader("🔍 実験検索")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox(
            "ステータス",
            options=['すべて', 'completed', 'failed', 'running'],
            format_func=lambda x: {
                'すべて': 'すべて',
                'completed': '完了',
                'failed': '失敗',
                'running': '実行中'
            }[x]
        )
    
    with col2:
        # モデル一覧を取得
        all_experiments = experiment_tracker.list_experiments()
        model_names = list(set([exp.get('model_name', '') for exp in all_experiments]))
        
        model_filter = st.selectbox(
            "モデル",
            options=['すべて'] + model_names,
            format_func=lambda x: x.split('/')[-1] if x != 'すべて' else x
        )
    
    with col3:
        limit = st.number_input("表示件数", 5, 100, 20)
    
    # 実験リスト取得
    experiments = experiment_tracker.list_experiments(
        status=None if status_filter == 'すべて' else status_filter,
        model_name=None if model_filter == 'すべて' else model_filter,
        limit=limit
    )
    
    if experiments:
        st.subheader("📋 実験一覧")
        
        # 実験データフレーム作成
        exp_data = []
        for exp in experiments:
            duration_hours = (exp.get('duration_seconds', 0) / 3600) if exp.get('duration_seconds') else None
            
            exp_data.append({
                'ID': exp['id'][:8],
                'モデル': exp['model_name'].split('/')[-1],
                'ステータス': {
                    'completed': '✅ 完了',
                    'failed': '❌ 失敗',
                    'running': '🔄 実行中'
                }.get(exp['status'], exp['status']),
                '作成日時': exp['created_at'][:19],
                '実行時間': f"{duration_hours:.1f}h" if duration_hours else "N/A",
                '最終損失': exp.get('final_metrics', {}).get('final_loss', 'N/A')
            })
        
        df = pd.DataFrame(exp_data)
        
        # クリック可能な実験選択
        selected_exp_id = st.selectbox(
            "詳細を表示する実験",
            options=[''] + [exp['id'] for exp in experiments],
            format_func=lambda x: "選択してください" if x == '' else f"{x[:8]} - {next(e['model_name'].split('/')[-1] for e in experiments if e['id'] == x)}"
        )
        
        st.dataframe(df, use_container_width=True)
        
        # 実験詳細表示
        if selected_exp_id:
            exp_detail = experiment_tracker.get_experiment(selected_exp_id)
            
            if exp_detail:
                st.subheader(f"📄 実験詳細: {selected_exp_id[:8]}")
                
                # 基本情報
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**基本情報**")
                    st.write(f"• モデル: {exp_detail['model_name']}")
                    st.write(f"• データセット: {Path(exp_detail['dataset_path']).name}")
                    st.write(f"• ステータス: {exp_detail['status']}")
                    st.write(f"• 作成日時: {exp_detail['created_at'][:19]}")
                
                with col2:
                    st.write("**実行情報**")
                    if exp_detail.get('duration_seconds'):
                        st.write(f"• 実行時間: {exp_detail['duration_seconds']/3600:.1f} 時間")
                    if exp_detail.get('output_dir'):
                        st.write(f"• 出力: {exp_detail['output_dir']}")
                    if exp_detail.get('error'):
                        st.error(f"エラー: {exp_detail['error']}")
                
                # 設定表示
                with st.expander("⚙️ 設定"):
                    st.json(exp_detail.get('config', {}))
                
                # メトリクス表示
                metrics = experiment_tracker.get_experiment_metrics(selected_exp_id)
                
                if metrics:
                    st.subheader("📊 学習曲線")
                    
                    # 損失グラフ
                    metrics_df = pd.DataFrame([
                        {
                            'step': m['step'],
                            'loss': m['metrics'].get('loss', 0),
                            'avg_loss': m['metrics'].get('avg_loss', 0)
                        }
                        for m in metrics if 'loss' in m['metrics']
                    ])
                    
                    if not metrics_df.empty:
                        fig = px.line(
                            metrics_df, 
                            x='step', 
                            y=['loss', 'avg_loss'],
                            title='Training Loss'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # ログ表示
                logs = experiment_tracker.get_experiment_logs(selected_exp_id)
                
                if logs:
                    with st.expander("📜 ログ"):
                        for log in logs[-10:]:  # 最新10件
                            timestamp = log['timestamp'][:19]
                            level = log['level']
                            message = log['message']
                            st.text(f"[{timestamp}] {level}: {message}")
                
                # 実験操作
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📁 実験エクスポート"):
                        export_path = f"./experiments/{selected_exp_id}_export.json"
                        if experiment_tracker.export_experiment(selected_exp_id, export_path):
                            st.success(f"✅ エクスポート完了: {export_path}")
                        else:
                            st.error("❌ エクスポート失敗")
                
                with col2:
                    if st.button("🗑️ 実験削除", type="secondary"):
                        if st.checkbox("削除を確認します", key=f"delete_{selected_exp_id}"):
                            if experiment_tracker.delete_experiment(selected_exp_id):
                                st.success("✅ 実験を削除しました")
                                st.rerun()
                            else:
                                st.error("❌ 削除に失敗しました")
    else:
        st.info("実験データがありません")


def settings_page():
    """設定ページ"""
    st.title("⚙️ 設定")
    
    # 設定ファイルの編集
    st.subheader("📝 設定ファイル")
    
    tab1, tab2 = st.tabs(["基本設定", "モデル設定"])
    
    with tab1:
        try:
            with open('./config/default_config.yaml', 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            edited_config = st.text_area(
                "default_config.yaml",
                value=config_content,
                height=400,
                help="YAML形式で設定を編集してください"
            )
            
            if st.button("💾 基本設定を保存"):
                try:
                    # YAML検証
                    yaml.safe_load(edited_config)
                    
                    with open('./config/default_config.yaml', 'w', encoding='utf-8') as f:
                        f.write(edited_config)
                    
                    st.success("✅ 設定を保存しました")
                    
                except yaml.YAMLError as e:
                    st.error(f"❌ YAML形式エラー: {e}")
                except Exception as e:
                    st.error(f"❌ 保存エラー: {e}")
                    
        except Exception as e:
            st.error(f"設定ファイル読み込みエラー: {e}")
    
    with tab2:
        try:
            with open('./config/models.yaml', 'r', encoding='utf-8') as f:
                models_content = f.read()
            
            edited_models = st.text_area(
                "models.yaml",
                value=models_content,
                height=400,
                help="利用可能なモデルとテンプレートの設定"
            )
            
            if st.button("💾 モデル設定を保存"):
                try:
                    # YAML検証
                    yaml.safe_load(edited_models)
                    
                    with open('./config/models.yaml', 'w', encoding='utf-8') as f:
                        f.write(edited_models)
                    
                    st.success("✅ 設定を保存しました")
                    
                except yaml.YAMLError as e:
                    st.error(f"❌ YAML形式エラー: {e}")
                except Exception as e:
                    st.error(f"❌ 保存エラー: {e}")
                    
        except Exception as e:
            st.error(f"設定ファイル読み込みエラー: {e}")
    
    # システム情報
    st.subheader("💻 システム情報")
    
    memory_monitor = MemoryMonitor()
    memory_info = memory_monitor.get_memory_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("総メモリ", f"{memory_info['total_gb']:.1f} GB")
        st.metric("使用メモリ", f"{memory_info['used_gb']:.1f} GB")
        st.metric("利用可能メモリ", f"{memory_info['available_gb']:.1f} GB")
    
    with col2:
        st.metric("使用率", f"{memory_info['percent']:.1f}%")
        st.metric("空きメモリ", f"{memory_info['free_gb']:.1f} GB")
    
    # ディスク使用量
    st.subheader("💾 ディスク使用量")
    
    directories = {
        "データ": "./data",
        "モデル": "./models",
        "実験": "./experiments",
        "ログ": "./logs"
    }
    
    for name, path in directories.items():
        if os.path.exists(path):
            size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(path)
                for filename in filenames
            ) / (1024**3)  # GB
            
            st.metric(name, f"{size:.2f} GB")
        else:
            st.metric(name, "0 GB")


# メイン関数
def main():
    try:
        # サイドバーメニュー
        selected_page = sidebar_menu()
        
        # ページ表示
        if selected_page == "home":
            home_page()
        elif selected_page == "dataset":
            dataset_page()
        elif selected_page == "training":
            training_page()
        elif selected_page == "quantization":
            quantization_page()
        elif selected_page == "ollama":
            ollama_page()
        elif selected_page == "experiments":
            experiments_page()
        elif selected_page == "settings":
            settings_page()
            
    except Exception as e:
        st.error(f"アプリケーションエラー: {e}")
        logger.exception("Application error")


if __name__ == "__main__":
    main()