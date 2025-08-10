import streamlit as st
import yaml
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
import tarfile
import shutil
from datetime import datetime

# プロジェクトのソースコードをインポート
from src.data_processor import DatasetProcessor
from src.trainer import TrainingManager
from src.quantizer import ModelQuantizer
from src.ollama_integration import OllamaIntegrator
from src.experiment_tracker import ExperimentTracker
from src.utils.memory_monitor import MemoryMonitor
from src.utils.validators import validate_all
from src.smart_recommender import SmartParameterRecommender
from typing import Dict, Any, List


def create_transfer_archive(experiment_id: str, output_name: str) -> Dict[str, Any]:
    """転送用アーカイブを作成"""
    try:
        # パス設定
        finetuned_dir = Path(f"./models/finetuned/{experiment_id}")
        quantized_dir = Path("./models/quantized")
        experiments_dir = Path(f"./experiments/{experiment_id}")
        output_path = Path(f"./{output_name}.tar.gz")
        
        # 必要なファイルを確認
        required_files = []
        
        # 1. LoRAアダプター
        adapters_file = finetuned_dir / "adapters.safetensors"
        if adapters_file.exists():
            required_files.append(("models/finetuned", adapters_file))
        
        # 2. アダプター設定
        adapter_config = finetuned_dir / "adapter_config.json"
        if adapter_config.exists():
            required_files.append(("models/finetuned", adapter_config))
        
        # 3. 量子化ファイルを探す
        gguf_file = None
        
        # まず、MLXモデルディレクトリからIDを取得
        mlx_model_dir = None
        for mlx_dir in finetuned_dir.glob("mlx_model_*"):
            mlx_model_dir = mlx_dir
            break
        
        if mlx_model_dir and mlx_model_dir.exists():
            # MLXモデルIDを抽出
            mlx_id = mlx_model_dir.name.replace("mlx_model_", "")
            
            # Q5_K_M量子化ファイルを優先的に探す
            for priority_suffix in ["-Q5_K_M.gguf", "-Q4_K_M.gguf", ".gguf"]:
                candidate_file = quantized_dir / f"mlx_model_{mlx_id}{priority_suffix}"
                if candidate_file.exists():
                    gguf_file = candidate_file
                    break
            
            # それでも見つからない場合は、glob検索
            if not gguf_file:
                for gguf_path in quantized_dir.glob(f"*{mlx_id}*.gguf"):
                    gguf_file = gguf_path
                    break
        
        if gguf_file and gguf_file.exists():
            required_files.append(("models/quantized", gguf_file))
        
        # 4. 実験設定
        exp_info = experiments_dir / "experiment_info.json"
        if exp_info.exists():
            required_files.append(("experiments", exp_info))
        
        if not required_files:
            return {
                'success': False,
                'error': f'実験 {experiment_id} の必要ファイルが見つかりません'
            }
        
        # アーカイブ作成
        with tarfile.open(output_path, 'w:gz') as tar:
            for category, file_path in required_files:
                # アーカイブ内での相対パス
                if category == "models/finetuned":
                    arcname = f"models/finetuned/{experiment_id}/{file_path.name}"
                elif category == "models/quantized":
                    arcname = f"models/quantized/{file_path.name}"
                elif category == "experiments":
                    arcname = f"experiments/{experiment_id}/{file_path.name}"
                
                tar.add(file_path, arcname=arcname)
        
        # ファイルサイズ
        size_mb = output_path.stat().st_size / (1024**2)
        
        return {
            'success': True,
            'archive_path': str(output_path.absolute()),
            'filename': output_path.name,
            'size_mb': size_mb,
            'gguf_filename': gguf_file.name if gguf_file else 'model.gguf',
            'files_included': len(required_files)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def get_cleanup_info() -> Dict[str, float]:
    """クリーンアップ対象のストレージ情報を取得"""
    
    def get_dir_size(path: Path) -> float:
        """ディレクトリサイズをGB単位で取得"""
        if not path.exists():
            return 0.0
        
        total_size = 0
        try:
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except (OSError, PermissionError):
            pass
        
        return total_size / (1024**3)
    
    return {
        'finetuned_size_gb': get_dir_size(Path("./models/finetuned")),
        'quantized_size_gb': get_dir_size(Path("./models/quantized")),
        'experiments_size_gb': get_dir_size(Path("./experiments")),
        'mlx_cache_size_gb': get_dir_size(Path("./models/cache")),
        'gguf_cache_size_gb': get_dir_size(Path("./models/gguf_cache"))
    }


def perform_cleanup(cleanup_options: List[str]) -> Dict[str, Any]:
    """クリーンアップを実行"""
    try:
        total_freed = 0.0
        
        # クリーンアップ前のサイズ記録
        cleanup_info_before = get_cleanup_info()
        
        for option in cleanup_options:
            if "ファインチューニング結果" in option:
                target_dir = Path("./models/finetuned")
                if target_dir.exists():
                    total_freed += cleanup_info_before['finetuned_size_gb']
                    shutil.rmtree(target_dir)
                    target_dir.mkdir(exist_ok=True)
            
            elif "量子化ファイル" in option:
                target_dir = Path("./models/quantized")
                if target_dir.exists():
                    total_freed += cleanup_info_before['quantized_size_gb']
                    shutil.rmtree(target_dir)
                    target_dir.mkdir(exist_ok=True)
            
            elif "実験データ" in option:
                target_dir = Path("./experiments")
                if target_dir.exists():
                    total_freed += cleanup_info_before['experiments_size_gb']
                    # metadata ファイルは保持
                    for item in target_dir.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                        elif item.name != "experiments_metadata.json":
                            item.unlink()
            
            elif "MLXキャッシュ" in option:
                target_dir = Path("./models/cache")
                if target_dir.exists():
                    total_freed += cleanup_info_before['mlx_cache_size_gb']
                    shutil.rmtree(target_dir)
                    target_dir.mkdir(exist_ok=True)
            
            elif "GGUFキャッシュ" in option:
                target_dir = Path("./models/gguf_cache")
                if target_dir.exists():
                    total_freed += cleanup_info_before['gguf_cache_size_gb']
                    shutil.rmtree(target_dir)
                    target_dir.mkdir(exist_ok=True)
        
        return {
            'success': True,
            'freed_gb': total_freed
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


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


def get_disk_usage():
    """ディスク使用量を取得"""
    import shutil
    
    try:
        total, used, free = shutil.disk_usage(".")
        
        total_gb = total / (1024**3)
        used_gb = used / (1024**3) 
        free_gb = free / (1024**3)
        usage_percent = (used / total) * 100
        
        return {
            'total_gb': total_gb,
            'used_gb': used_gb,
            'free_gb': free_gb,
            'usage_percent': usage_percent
        }
    except Exception as e:
        logger.error(f"ディスク使用量取得エラー: {e}")
        return {
            'total_gb': 0,
            'used_gb': 0,
            'free_gb': 0,
            'usage_percent': 0
        }


def cleanup_docker():
    """Dockerのクリーンアップを実行"""
    import subprocess
    
    try:
        with st.spinner("🐳 Dockerクリーンアップ実行中..."):
            results = []
            
            # 未使用イメージを削除
            result = subprocess.run(
                ["docker", "image", "prune", "-a", "-f"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                results.append(f"Images: {result.stdout.split('Total reclaimed space: ')[-1].strip()}")
            
            # 未使用コンテナを削除
            result = subprocess.run(
                ["docker", "container", "prune", "-f"],
                capture_output=True, text=True
            )
            if result.returncode == 0 and "Total reclaimed space: " in result.stdout:
                results.append(f"Containers: {result.stdout.split('Total reclaimed space: ')[-1].strip()}")
            
            # 未使用ボリュームを削除
            result = subprocess.run(
                ["docker", "volume", "prune", "-f"],
                capture_output=True, text=True
            )
            if result.returncode == 0 and "Total reclaimed space: " in result.stdout:
                results.append(f"Volumes: {result.stdout.split('Total reclaimed space: ')[-1].strip()}")
            
            # ビルドキャッシュを削除
            result = subprocess.run(
                ["docker", "builder", "prune", "-a", "-f"],
                capture_output=True, text=True
            )
            if result.returncode == 0 and "Total:" in result.stdout:
                cache_size = result.stdout.split("Total:")[-1].strip()
                results.append(f"Build Cache: {cache_size}")
            
            if results:
                st.success(f"✅ Dockerクリーンアップ完了!\n\n" + "\n".join(results))
            else:
                st.info("ℹ️ クリーンアップ対象のDockerリソースはありませんでした")
                
    except FileNotFoundError:
        st.warning("⚠️ Dockerが見つかりません。Dockerがインストールされているか確認してください。")
    except Exception as e:
        st.error(f"❌ Dockerクリーンアップエラー: {e}")


def cleanup_temp_files():
    """一時ファイルのクリーンアップを実行"""
    import subprocess
    
    try:
        with st.spinner("🗂️ 一時ファイル削除中..."):
            deleted_files = 0
            
            # プロジェクト内の一時ファイルを削除
            temp_patterns = [
                "*.log", "*.tmp", "__pycache__", ".DS_Store", 
                "*.pyc", ".pytest_cache", ".coverage"
            ]
            
            for pattern in temp_patterns:
                result = subprocess.run(
                    ["find", ".", "-name", pattern, "-type", "f", "-delete"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    # ファイル数をカウント（概算）
                    count_result = subprocess.run(
                        ["find", ".", "-name", pattern, "-type", "f"],
                        capture_output=True, text=True
                    )
                    deleted_files += len([l for l in count_result.stdout.split('\n') if l.strip()])
            
            # システムの一時ディレクトリもクリーンアップ（安全な範囲で）
            subprocess.run(["rm", "-rf", "/tmp/streamlit-*"], capture_output=True)
            subprocess.run(["rm", "-rf", "/tmp/mlx_*"], capture_output=True)
            
            st.success(f"✅ 一時ファイル削除完了! 約{deleted_files}個のファイルを処理しました")
            
    except Exception as e:
        st.error(f"❌ 一時ファイル削除エラー: {e}")


def cleanup_quantization_files():
    """量子化関連の不要ファイルをクリーンアップ"""
    try:
        with st.spinner("📦 量子化ファイル整理中..."):
            cleaned_size = 0
            
            # models/quantizedディレクトリの破損ファイルを確認・削除
            quantized_dir = Path("./models/quantized")
            if quantized_dir.exists():
                for file in quantized_dir.glob("*.gguf"):
                    try:
                        # GGUFファイルの簡易チェック（最初の4バイトがGGUF）
                        with open(file, 'rb') as f:
                            header = f.read(4)
                            if header != b'GGUF':
                                file_size = file.stat().st_size / (1024**3)  # GB
                                file.unlink()
                                cleaned_size += file_size
                                st.warning(f"破損ファイルを削除: {file.name}")
                    except Exception:
                        # 読み取りできないファイルも削除
                        try:
                            file_size = file.stat().st_size / (1024**3)
                            file.unlink()
                            cleaned_size += file_size
                            st.warning(f"アクセス不可ファイルを削除: {file.name}")
                        except:
                            pass
            
            # MLXのキャッシュディレクトリをクリーンアップ
            mlx_cache_dirs = [
                Path("./models/cache"),
                Path("./models/.cache"),
                Path("~/.cache/mlx").expanduser(),
                Path("~/.cache/huggingface").expanduser()
            ]
            
            for cache_dir in mlx_cache_dirs:
                if cache_dir.exists():
                    try:
                        import shutil
                        dir_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) / (1024**3)
                        shutil.rmtree(cache_dir, ignore_errors=True)
                        cleaned_size += dir_size
                    except:
                        pass
            
            if cleaned_size > 0:
                st.success(f"✅ 量子化ファイル整理完了! {cleaned_size:.2f}GB削除しました")
            else:
                st.info("ℹ️ クリーンアップ対象の量子化ファイルはありませんでした")
                
    except Exception as e:
        st.error(f"❌ 量子化ファイル整理エラー: {e}")


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
    
    # 既存ディレクトリ管理セクション
    st.subheader("🗂️ データセットディレクトリ管理")
    
    processed_data_dir = Path("./data/processed")
    
    if processed_data_dir.exists():
        # 既存ディレクトリの一覧取得
        dataset_dirs = [d for d in processed_data_dir.iterdir() if d.is_dir()]
        
        if dataset_dirs:
            # ディレクトリ情報を表示
            dir_info = []
            for dir_path in dataset_dirs:
                # ファイル数とサイズを計算
                file_count = sum(1 for f in dir_path.rglob('*') if f.is_file())
                total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                total_size_mb = total_size / (1024 * 1024)
                
                # 作成日時を取得
                creation_time = datetime.fromtimestamp(dir_path.stat().st_ctime)
                
                dir_info.append({
                    'ディレクトリ名': dir_path.name,
                    'ファイル数': file_count,
                    'サイズ(MB)': f"{total_size_mb:.1f}",
                    '作成日時': creation_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'パス': str(dir_path)
                })
            
            # データフレームで表示
            df = pd.DataFrame(dir_info)
            st.dataframe(df.drop('パス', axis=1), use_container_width=True)
            
            # 削除機能
            with st.expander("🗑️ ディレクトリ削除", expanded=False):
                selected_dirs = st.multiselect(
                    "削除するディレクトリを選択",
                    options=[info['ディレクトリ名'] for info in dir_info],
                    help="複数選択可能です。削除したディレクトリは復元できません。"
                )
                
                if selected_dirs:
                    # 削除対象の詳細情報
                    total_size_to_delete = sum(
                        float(info['サイズ(MB)']) for info in dir_info 
                        if info['ディレクトリ名'] in selected_dirs
                    )
                    total_files_to_delete = sum(
                        info['ファイル数'] for info in dir_info 
                        if info['ディレクトリ名'] in selected_dirs
                    )
                    
                    st.warning(f"⚠️ 削除予定: {len(selected_dirs)}個のディレクトリ、{total_files_to_delete}個のファイル、{total_size_to_delete:.1f}MB")
                    
                    # 確認チェックボックス
                    confirm_delete = st.checkbox(
                        "上記のディレクトリを完全に削除することを確認します",
                        key="confirm_dataset_delete"
                    )
                    
                    if confirm_delete:
                        if st.button("🗑️ 選択したディレクトリを削除", type="secondary"):
                            deleted_count = 0
                            deleted_size = 0
                            
                            with st.spinner("ディレクトリを削除中..."):
                                for dir_name in selected_dirs:
                                    dir_path = processed_data_dir / dir_name
                                    if dir_path.exists():
                                        try:
                                            # サイズを記録してから削除
                                            size_mb = float(next(
                                                info['サイズ(MB)'] for info in dir_info 
                                                if info['ディレクトリ名'] == dir_name
                                            ))
                                            
                                            shutil.rmtree(dir_path)
                                            deleted_count += 1
                                            deleted_size += size_mb
                                            
                                        except Exception as e:
                                            st.error(f"❌ {dir_name} の削除に失敗: {e}")
                            
                            if deleted_count > 0:
                                st.success(f"✅ {deleted_count}個のディレクトリを削除しました（{deleted_size:.1f}MB削除）")
                                st.rerun()
                            else:
                                st.error("❌ ディレクトリの削除に失敗しました")
        else:
            st.info("📁 処理済みデータセットディレクトリはありません")
    else:
        st.info("📁 data/processedディレクトリが存在しません")
    
    st.divider()
    
    # サンプルファイル選択オプション
    st.subheader("📁 ファイル選択")
    
    # data/templatesディレクトリのサンプルファイル一覧を取得
    templates_dir = Path("./data/templates")
    sample_files = []
    if templates_dir.exists():
        for ext in ['*.csv', '*.json', '*.jsonl', '*.txt']:
            sample_files.extend(templates_dir.glob(ext))
    
    tab1, tab2 = st.tabs(["📂 サンプルファイル", "📤 ファイルアップロード"])
    
    selected_file_path = None
    
    with tab1:
        if sample_files:
            st.write("data/templatesディレクトリのサンプルファイル:")
            sample_file_names = [str(f.name) for f in sample_files]
            selected_sample = st.selectbox(
                "サンプルファイルを選択:",
                options=[""] + sample_file_names,
                help="data/templatesディレクトリにあるサンプルファイルから選択"
            )
            
            if selected_sample:
                selected_file_path = str(templates_dir / selected_sample)
                st.success(f"選択されたファイル: {selected_sample}")
        else:
            st.info("data/templatesディレクトリにサンプルファイルが見つかりません")
    
    with tab2:
        # ファイルアップロード
        uploaded_file = st.file_uploader(
            "データファイルをアップロード",
            type=['csv', 'json', 'jsonl', 'txt'],
            help="CSV、JSON、JSONL、TXTファイルに対応"
        )
    
    # ファイル処理（アップロードまたはサンプルファイル選択）
    processed_file_path = None
    
    if uploaded_file is not None:
        # 一時ファイルに保存
        temp_path = f"./data/raw/{uploaded_file.name}"
        os.makedirs("./data/raw", exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"ファイルアップロード完了: {uploaded_file.name}")
        processed_file_path = temp_path
        
    elif selected_file_path:
        processed_file_path = selected_file_path
    
    if processed_file_path:
        # データプレビュー
        try:
            processor = DatasetProcessor()
            df = processor.load_dataset(processed_file_path)
            
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
                    
                    # ファイル名から拡張子を除いた部分を取得
                    base_filename = Path(processed_file_path).stem
                    output_dir = f"./data/processed/{base_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    result = processor_with_config.process_dataset(
                        processed_file_path,
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
                        
                        # データセットサイズの推奨表示
                        train_count = result['splits']['train']
                        if train_count > 20:
                            st.warning("⚠️ **データセット最適化推奨**")
                            st.markdown("""
                            **ファインチューニング効率のための推奨事項:**
                            - 📊 **最適サイズ**: 10-20件が効果的
                            - 🎯 **現在のサイズ**: {}件（やや多め）
                            - 💡 **推奨**: 最も重要な10-15件に絞り込むと学習効率が向上します
                            - ✨ **利点**: 学習が深く、特定情報への回答精度が向上
                            """.format(train_count))
                        elif train_count >= 10:
                            st.info("✅ **データセットサイズ良好**")
                            st.markdown("""
                            **現在のデータセットサイズ: {}件**
                            - 🎯 ファインチューニングに最適なサイズです
                            - 📈 効果的な学習が期待できます
                            """.format(train_count))
                        elif train_count >= 5:
                            st.info("ℹ️ **小規模データセット**")
                            st.markdown("""
                            **現在のデータセットサイズ: {}件**
                            - 📝 少数精鋭のデータセットです
                            - 🚀 高い学習率と多くのイテレーションで効果的
                            """.format(train_count))
                        else:
                            st.warning("⚠️ **データセット不足**")
                            st.markdown("""
                            **現在のデータセットサイズ: {}件（少なすぎます）**
                            - 📈 **推奨**: 最低5-10件のデータを準備してください
                            - 🎯 **品質重視**: 少数でも正確なデータが重要です
                            """.format(train_count))
                        
                        # 分割結果
                        st.write("**データ分割結果**")
                        splits_df = pd.DataFrame([
                            {'分割': '訓練', '件数': result['splits']['train']},
                            {'分割': '検証', '件数': result['splits']['val']},
                            {'分割': 'テスト', '件数': result['splits']['test']}
                        ])
                        st.dataframe(splits_df, use_container_width=True)
                        
                        # ファインチューニング推奨パラメータ
                        if train_count <= 10:
                            st.info("🎯 **推奨ファインチューニング設定（小規模データセット用）**")
                            recommended_col1, recommended_col2 = st.columns(2)
                            with recommended_col1:
                                st.markdown("""
                                **基本設定:**
                                - エポック数: 5-8
                                - 学習率: 1e-4 ～ 2e-4
                                - バッチサイズ: 1-2
                                """)
                            with recommended_col2:
                                st.markdown("""
                                **期待される効果:**
                                - より深い学習
                                - 特定情報への正確な回答
                                - 過学習リスクの軽減
                                """)
                        
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
    default_model = None
    if 'base_models' in models_config:
        for key, model_info in models_config['base_models'].items():
            model_options[model_info['name']] = model_info['display_name']
            # Gemma2:2bをデフォルトに設定
            if key == 'gemma2-2b':
                default_model = model_info['name']
    
    # デフォルトモデルのインデックスを取得
    default_index = 0
    if default_model and default_model in model_options:
        default_index = list(model_options.keys()).index(default_model)
    
    selected_model = st.selectbox(
        "ベースモデル",
        options=list(model_options.keys()),
        index=default_index,
        format_func=lambda x: model_options.get(x, x),
        help="推奨: Gemma 2 2B Instruct (軽量で高速)"
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
    st.info(f"📁 処理済みデータセットの保存場所: `{processed_data_dir.resolve()}`")
    
    if processed_data_dir.exists():
        dataset_dirs = [d for d in processed_data_dir.iterdir() if d.is_dir()]
        
        if dataset_dirs:
            selected_dataset_dir = st.selectbox(
                "処理済みデータセット",
                options=dataset_dirs,
                format_func=lambda x: x.name,
                help="データセット準備で作成された処理済みデータセットから選択"
            )
            
            # データセット情報表示
            if selected_dataset_dir:
                st.write(f"**選択されたディレクトリ**: `{selected_dataset_dir}`")
                
                train_file = selected_dataset_dir / "train.jsonl"
                val_file = selected_dataset_dir / "val.jsonl"
                test_file = selected_dataset_dir / "test.jsonl"
                
                col1, col2, col3 = st.columns(3)
                
                if train_file.exists():
                    with open(train_file, 'r') as f:
                        train_count = sum(1 for _ in f)
                    with col1:
                        st.metric("📊 訓練データ", f"{train_count:,} 件")
                
                if val_file.exists():
                    with open(val_file, 'r') as f:
                        val_count = sum(1 for _ in f)
                    with col2:
                        st.metric("📊 検証データ", f"{val_count:,} 件")
                
                if test_file.exists():
                    with open(test_file, 'r') as f:
                        test_count = sum(1 for _ in f)
                    with col3:
                        st.metric("📊 テストデータ", f"{test_count:,} 件")
        else:
            st.warning("処理済みデータセットがありません。先にデータセット準備を行ってください。")
            selected_dataset_dir = None
    else:
        st.warning("データディレクトリが存在しません。")
        selected_dataset_dir = None
    
    # モデルファイル配置手順
    st.subheader("📁 モデルファイル配置")
    
    with st.expander("モデルファイルの準備手順", expanded=True):
        st.info("""
        **このアプリはローカルのモデルファイルを使用します。以下の手順でファイルを配置してください：**
        """)
        
        # モデルファイル配置状況の確認
        gemma_path = Path("./models/gemma-2-2b-it")
        elyza_path = Path("./models/Llama-3-ELYZA-JP-8B")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🤖 Gemma 2 2B Instruct**")
            if gemma_path.exists():
                required_files = ["config.json", "model.safetensors.index.json"]
                safetensors_files = list(gemma_path.glob("model-*.safetensors"))
                tokenizer_files = [f for f in ["tokenizer.json", "tokenizer.model"] if (gemma_path / f).exists()]
                
                all_required_exist = all((gemma_path / f).exists() for f in required_files)
                has_tokenizer = len(tokenizer_files) > 0
                has_model_files = len(safetensors_files) > 0
                
                if all_required_exist and has_tokenizer and has_model_files:
                    st.success("✅ ファイル配置完了")
                    st.write(f"📊 モデルファイル数: {len(safetensors_files)}")
                else:
                    st.error("❌ ファイル不足")
                    missing = []
                    if not all_required_exist:
                        missing.extend([f for f in required_files if not (gemma_path / f).exists()])
                    if not has_tokenizer:
                        missing.append("tokenizer files")
                    if not has_model_files:
                        missing.append("model-*.safetensors")
                    st.write(f"不足ファイル: {', '.join(missing)}")
            else:
                st.warning("⚠️ ディレクトリなし")
                st.code(f"配置先: {gemma_path.absolute()}")
        
        with col2:
            st.write("**🗾 ELYZA Japanese 8B**")
            if elyza_path.exists():
                st.success("✅ ファイル配置完了")
            else:
                st.warning("⚠️ ディレクトリなし") 
                st.code(f"配置先: {elyza_path.absolute()}")
        
        st.markdown("""
        ### 📥 ファイル入手方法:
        
        #### **Gemma 2 2B Instruct**:
        1. [HuggingFace](https://huggingface.co/google/gemma-2-2b-it) からダウンロード
        2. 以下のディレクトリに配置: `./models/gemma-2-2b-it/`
        3. 必要ファイル:
           - `config.json`
           - `model-*.safetensors` (複数ファイル)
           - `model.safetensors.index.json`  
           - `tokenizer.json` または `tokenizer.model`
        
        #### **ELYZA Japanese 8B**:
        1. [HuggingFace](https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B) からダウンロード
        2. 以下のディレクトリに配置: `./models/Llama-3-ELYZA-JP-8B/`
        """)
    
    # ハイパーパラメータ設定
    st.subheader("⚙️ ハイパーパラメータ設定")
    
    # スマート推奨機能
    if selected_dataset_dir:
        with st.expander("🧠 AI推奨設定", expanded=True):
            try:
                from src.smart_recommender import SmartParameterRecommender
                recommender = SmartParameterRecommender()
                
                # データセットパスを取得
                train_file_path = str(selected_dataset_dir / "train.jsonl")
                
                # 推奨パラメータを取得
                recommendation = recommender.recommend_training_parameters(train_file_path)
                
                col_rec1, col_rec2 = st.columns([2, 1])
                with col_rec1:
                    st.info(f"💡 **推奨理由**: {recommendation['rationale']}")
                    st.write(f"⏱️ **予想実行時間**: {recommendation['estimated_time_minutes']:.1f}分")
                    st.write(f"🔄 **予想イテレーション数**: {recommendation['estimated_iterations']:,}回")
                with col_rec2:
                    apply_ai_recommendations = st.checkbox("AI推奨設定を適用", value=True, key="training_ai_rec")
                
                # 信頼度表示
                confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                st.write(f"**信頼度**: {confidence_color.get(recommendation['confidence_level'], '❓')} {recommendation['confidence_level']}")
                
            except Exception as e:
                st.warning(f"AI推奨機能の読み込みに失敗: {e}")
                apply_ai_recommendations = False
                recommendation = None
    else:
        apply_ai_recommendations = False
        recommendation = None
    
    with st.expander("基本設定", expanded=True):
        # デフォルト値を設定（AI推奨が有効な場合はそれを使用）
        default_batch_size = 1
        default_learning_rate = 5e-5
        default_epochs = 3
        default_lora_rank = 16
        default_lora_alpha = 32
        default_lora_dropout = 0.1
        
        if apply_ai_recommendations and recommendation:
            params = recommendation.get('parameters', {})
            default_batch_size = params.get('batch_size', 1)
            default_learning_rate = params.get('learning_rate', 5e-5)
            default_epochs = params.get('num_epochs', 3)
            default_lora_rank = params.get('lora_rank', 16)
            default_lora_alpha = params.get('lora_alpha', 32)
            default_lora_dropout = params.get('lora_dropout', 0.1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            batch_size = st.selectbox("バッチサイズ", [1, 2, 4], 
                                      index=[1, 2, 4].index(default_batch_size) if default_batch_size in [1, 2, 4] else 0)
            
            lr_options = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
            lr_index = 2  # デフォルトで5e-5
            if default_learning_rate in lr_options:
                lr_index = lr_options.index(default_learning_rate)
            elif default_learning_rate > 1e-4:
                lr_index = 4  # 2e-4を選択
            elif default_learning_rate > 5e-5:
                lr_index = 3  # 1e-4を選択
                
            learning_rate = st.select_slider(
                "学習率",
                options=lr_options,
                value=lr_options[lr_index],
                format_func=lambda x: f"{x:.0e}"
            )
            num_epochs = st.slider("エポック数", 1, 15, min(default_epochs, 15))
        
        with col2:
            lora_rank = st.slider("LoRA Rank", 4, 64, min(default_lora_rank, 64))
            lora_alpha = st.slider("LoRA Alpha", 8, 256, min(default_lora_alpha, 256))
            lora_dropout = st.slider("LoRA Dropout", 0.0, 0.3, default_lora_dropout)
    
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
            progress_container = st.container()
            with progress_container:
                st.markdown("### 🚀 トレーニング進行状況")
                progress_bar = st.progress(st.session_state['training_progress'])
                progress_col1, progress_col2 = st.columns(2)
                with progress_col1:
                    st.metric("🔄 進行状況", f"{st.session_state['training_progress']*100:.1f}%")
                with progress_col2:
                    if 'training_current_loss' not in st.session_state:
                        st.session_state['training_current_loss'] = 'N/A'
                    st.metric("📊 現在のLoss", st.session_state['training_current_loss'])
                
                status_text = st.text_area("📝 ステータス", st.session_state['training_status'], height=100, disabled=True)
            
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
    
    # 現在のトレーニング状況（改善版）
    st.subheader("🔍 トレーニング状況確認")
    
    # 最新の実験を自動取得
    tracker = ExperimentTracker()
    experiments = tracker.list_experiments()
    
    if experiments:
        latest_experiment = experiments[0]
        experiment_id = latest_experiment['id']
        
        # 実験状況表示
        col1, col2, col3 = st.columns(3)
        with col1:
            status = latest_experiment.get('status', 'unknown')
            status_emoji = {"running": "🔄", "completed": "✅", "failed": "❌"}
            st.metric("ステータス", f"{status_emoji.get(status, '❓')} {status}")
            
        with col2:
            duration = latest_experiment.get('duration_seconds')
            if duration:
                duration_str = f"{duration:.1f}秒"
            elif status == 'running':
                import time
                start_time = latest_experiment.get('started_at')
                if start_time:
                    from datetime import datetime
                    start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    now = datetime.now(start.tzinfo)
                    duration = (now - start).total_seconds()
                    duration_str = f"{duration:.0f}秒経過"
                else:
                    duration_str = "実行中"
            else:
                duration_str = "N/A"
            st.metric("実行時間", duration_str)
            
        with col3:
            st.metric("実験ID", experiment_id[:8])
        
        # 進行状況バー（推定）
        if status == 'running':
            # 実行時間ベースで進行率を推定
            estimated_total_time = 120  # 2分と推定
            if duration:
                progress = min(duration / estimated_total_time, 0.95)  # 最大95%まで
            else:
                progress = 0.1
            
            st.progress(progress, text=f"進行中... ({progress*100:.0f}%)")
            
            # リアルタイム更新ボタン
            col_refresh1, col_refresh2 = st.columns(2)
            with col_refresh1:
                if st.button("🔄 状況更新", type="secondary"):
                    st.rerun()
            with col_refresh2:
                # 自動リフレッシュ（5秒間隔）
                if st.button("⏸️ 自動更新停止", help="5秒ごとの自動更新を停止"):
                    st.session_state['auto_refresh'] = False
                    
            # 自動リフレッシュ機能
            if st.session_state.get('auto_refresh', True):
                import time
                time.sleep(5)
                st.rerun()
                
        elif status == 'completed':
            st.progress(1.0, text="完了 (100%)")
            
            # 完了時のアクション
            col_action1, col_action2, col_action3 = st.columns(3)
            with col_action1:
                if st.button("📈 実験詳細"):
                    st.switch_page("experiments")
            with col_action2:
                if st.button("📦 量子化へ"):
                    st.switch_page("quantization")  
            with col_action3:
                if st.button("🤖 Ollama統合"):
                    st.switch_page("Ollama統合")
                    
            # 最終メトリクス表示
            final_metrics = latest_experiment.get('final_metrics', {})
            if final_metrics:
                st.subheader("📊 最終結果")
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    if 'final_loss' in final_metrics:
                        st.metric("最終Loss", f"{final_metrics['final_loss']:.3f}")
                with metric_col2:
                    if 'perplexity' in final_metrics:
                        st.metric("Perplexity", f"{final_metrics['perplexity']:.1f}")
                        
        elif status == 'failed':
            st.progress(0.0, text="失敗")
            st.error(f"❌ トレーニングが失敗しました: {latest_experiment.get('error', '不明なエラー')}")
            
    else:
        st.info("📝 まだトレーニングが実行されていません")
        
    # トレーニング管理の状態リセット
    if st.session_state.get('training_manager'):
        status = st.session_state['training_manager'].get_training_status()
        if not status['is_training'] and experiments and experiments[0].get('status') == 'completed':
            # 完了状態を正しく認識させる
            st.session_state['training_manager'].is_training = False
            st.session_state['training_manager'].current_experiment_id = None


def quantization_page():
    """量子化ページ"""
    st.title("📦 量子化")
    
    quantizer = ModelQuantizer()
    
    # ディスク容量管理セクション
    st.subheader("💾 ディスク容量管理")
    
    # ディスク容量表示
    disk_info = get_disk_usage()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("使用容量", f"{disk_info['used_gb']:.1f} GB")
    with col2:
        st.metric("空き容量", f"{disk_info['free_gb']:.1f} GB")
    with col3:
        capacity_color = "🔴" if disk_info['usage_percent'] > 95 else "🟡" if disk_info['usage_percent'] > 85 else "🟢"
        st.metric("使用率", f"{capacity_color} {disk_info['usage_percent']:.1f}%")
    
    # 容量不足警告
    if disk_info['free_gb'] < 5:
        st.warning(f"⚠️ 空き容量が{disk_info['free_gb']:.1f}GBです。量子化には最低5GB必要です。")
        
        # クリーンアップ機能
        with st.expander("🧹 ディスククリーンアップ", expanded=True):
            st.markdown("""
            **以下のクリーンアップを実行してディスク容量を確保できます：**
            """)
            
            cleanup_col1, cleanup_col2 = st.columns(2)
            
            with cleanup_col1:
                if st.button("🐳 Dockerクリーンアップ", help="未使用のDockerイメージ・コンテナ・ボリュームを削除"):
                    cleanup_docker()
                
                if st.button("🗂️ 一時ファイル削除", help="ログファイル、キャッシュファイルなどを削除"):
                    cleanup_temp_files()
            
            with cleanup_col2:
                if st.button("📦 量子化ファイル整理", help="古い量子化ファイルや失敗したファイルを削除"):
                    cleanup_quantization_files()
                
                if st.button("🔄 容量再確認", help="ディスク使用量を最新状態に更新"):
                    st.rerun()
    
    st.divider()
    
    # llama.cpp状態チェック
    st.subheader("🔧 llama.cpp 状態")
    
    if quantizer.check_llama_cpp():
        st.success("✅ llama.cpp が利用可能です")
    else:
        st.error("❌ llama.cpp が見つかりません")
        
        with st.expander("📋 解決方法（初心者向け）", expanded=True):
            st.warning("**量子化にはllama.cppのセットアップが必要です**")
            
            st.markdown("""
            ### 🛠️ 自動セットアップ手順:
            
            #### **方法1: 自動セットアップスクリプト（推奨）**
            以下のボタンを押すと自動でllama.cppをセットアップします：
            """)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("🔧 自動セットアップ実行", type="primary"):
                    with st.spinner("llama.cppをセットアップ中..."):
                        try:
                            # 自動セットアップを実行
                            import subprocess
                            result = subprocess.run(
                                ["bash", "./setup.sh"], 
                                capture_output=True, 
                                text=True,
                                timeout=600  # 10分タイムアウト
                            )
                            
                            if result.returncode == 0:
                                st.success("✅ セットアップ完了！ページを更新してください")
                                st.rerun()
                            else:
                                st.error(f"❌ セットアップ失敗: {result.stderr}")
                                
                        except subprocess.TimeoutExpired:
                            st.error("❌ セットアップがタイムアウトしました（10分）")
                        except Exception as e:
                            st.error(f"❌ セットアップエラー: {e}")
            
            with col2:
                if st.button("🔄 状態を再確認"):
                    st.rerun()
            
            st.markdown("""
            #### **方法2: 手動実行（上級者向け）**
            ターミナルで以下のコマンドを実行してください：
            ```bash
            cd /Users/matsbaccano/Projects/clone/mlx-finetuning
            ./setup.sh
            ```
            
            #### **📋 setup.shの処理内容:**
            - Homebrewの確認・インストール
            - Minicondaの確認・インストール  
            - llama.cppのクローンとビルド
            - Ollamaのインストール
            - 必要な依存関係の設定
            
            #### **⏱️ 所要時間:**
            初回セットアップ: 約5-15分（ネットワーク速度による）
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
    
    # スマート推奨機能
    if selected_model_path and validation_result.get('is_valid'):
        with st.expander("🧠 AI推奨設定", expanded=True):
            try:
                from src.smart_recommender import SmartParameterRecommender
                recommender = SmartParameterRecommender()
                
                # モデルサイズ取得
                model_size_gb = validation_result.get('model_size_gb', 2.0)
                
                # 用途選択UI
                use_case = st.selectbox(
                    "用途を選択（推奨設定の参考にします）",
                    options=["高精度重視", "速度重視", "メモリ効率重視", "バランス重視"],
                    index=3,  # バランス重視をデフォルト
                    help="選択した用途に応じて最適な量子化方法を推奨します"
                )
                
                # 推奨パラメータを取得
                recommendation = recommender.recommend_quantization_parameters(model_size_gb, use_case)
                
                col_rec1, col_rec2 = st.columns([2, 1])
                with col_rec1:
                    st.info(f"💡 **推奨**: {recommendation['method']} - {recommendation['reason']}")
                with col_rec2:
                    apply_recommendations = st.checkbox("推奨設定を適用", value=True)
                
            except Exception as e:
                st.warning(f"推奨機能の読み込みに失敗: {e}")
                apply_recommendations = False
    else:
        apply_recommendations = False
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 推奨設定が利用可能で適用される場合
        if apply_recommendations and 'recommendation' in locals():
            default_method = recommendation['method']
            # デフォルトインデックスを計算
            method_list = list(quant_info['available_methods'].keys())
            try:
                default_index = method_list.index(default_method)
            except ValueError:
                default_index = 1  # Q5_K_Mをデフォルト
        else:
            default_index = 1  # Q5_K_Mをデフォルト
        
        quantization_method = st.selectbox(
            "量子化方法",
            options=list(quant_info['available_methods'].keys()),
            index=default_index
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
        
        # スマート推奨機能
        with st.expander("🧠 AI推奨設定", expanded=True):
            try:
                from src.smart_recommender import SmartParameterRecommender
                recommender = SmartParameterRecommender()
                
                # ファインチューニング結果からデータセット情報を取得を試行
                dataset_stats = {'total_samples': 10, 'has_specific_knowledge': True}  # デフォルト値
                
                # 実験IDからデータセット分析結果を取得する試行
                if selected_gguf and hasattr(selected_gguf, 'name'):
                    model_filename = selected_gguf.name
                    # ファイル名から実験IDを抽出（例: mlx_model_1754767357-Q5_K_M.gguf -> 1754767357）
                    import re
                    experiment_match = re.search(r'mlx_model_(\d+)', model_filename)
                    if experiment_match:
                        experiment_timestamp = experiment_match.group(1)
                        # 実験情報から元のデータセットパスを取得する試行
                        from src.experiment_tracker import ExperimentTracker
                        tracker = ExperimentTracker()
                        try:
                            # 最近の実験からデータセット統計情報を取得
                            experiments = tracker.list_experiments()
                            if experiments:
                                latest_exp = experiments[0]  # 最新実験
                                if 'dataset_analysis' in latest_exp.get('final_metrics', {}):
                                    dataset_stats = latest_exp['final_metrics']['dataset_analysis']
                        except:
                            pass  # デフォルト値を使用
                
                # モデルタイプ推定（Gemmaの場合）
                model_type = "gemma2"  
                
                # 推奨パラメータを取得
                ollama_recommendation = recommender.recommend_ollama_parameters(model_type, dataset_stats)
                
                st.info(f"💡 **推奨設定**: データセット特性（{dataset_stats.get('total_samples', 'N/A')}件、特定知識{'あり' if dataset_stats.get('has_specific_knowledge') else 'なし'}）に基づく最適化")
                
                col_rec1, col_rec2 = st.columns([2, 1])
                with col_rec1:
                    temp_rec = ollama_recommendation.get('temperature', 0.7)
                    top_p_rec = ollama_recommendation.get('top_p', 0.9)
                    st.write(f"Temperature: {temp_rec}, Top-P: {top_p_rec}")
                with col_rec2:
                    apply_ollama_recommendations = st.checkbox("推奨設定を適用", value=True, key="ollama_rec")
                
            except Exception as e:
                st.warning(f"推奨機能の読み込みに失敗: {e}")
                apply_ollama_recommendations = False
                ollama_recommendation = {'temperature': 0.7, 'top_p': 0.9, 'num_ctx': 4096, 'repeat_penalty': 1.1}
        
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
        
        # パラメータ最適化提案を取得（従来の方法も併用）
        optimization = integrator.optimize_parameters(model_name, use_case)
        recommended = optimization['recommended_parameters']
        
        # AI推奨が有効な場合はそちらを優先
        if apply_ollama_recommendations and 'ollama_recommendation' in locals():
            final_recommendations = ollama_recommendation
        else:
            final_recommendations = recommended
        
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider(
                "Temperature",
                0.0, 2.0, final_recommendations.get('temperature', 0.7),
                help="応答のランダム性"
            )
            top_p = st.slider(
                "Top P",
                0.0, 1.0, final_recommendations.get('top_p', 0.9),
                help="核サンプリング"
            )
        
        with col2:
            repeat_penalty = st.slider(
                "Repeat Penalty",
                1.0, 2.0, final_recommendations.get('repeat_penalty', 1.1),
                help="繰り返し抑制"
            )
            num_ctx = st.selectbox(
                "Context Length",
                options=[2048, 4096, 8192, 16384],
                index=1 if final_recommendations.get('num_ctx', 4096) == 4096 else 0,
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
        
        if model_to_delete:
            # 削除確認チェックボックスを先に表示
            confirm_delete = st.checkbox(f"「{model_to_delete}」の削除を確認します")
            
            if confirm_delete and st.button("🗑️ モデル削除", type="secondary"):
                with st.spinner("モデル削除中..."):
                    result = integrator.delete_model(model_to_delete)
                    
                    if result['success']:
                        st.success(f"✅ {result['message']}")
                        st.rerun()
                    else:
                        st.error(f"❌ {result['error']}")
    
    # ===============================
    # ✅ モデル転送用tarボール作成
    # ===============================
    st.header("📦 モデル転送")
    st.write("ファインチューニング済みモデルを他のPCに転送するためのアーカイブを作成します。")
    
    # 利用可能な実験を取得
    experiment_tracker = ExperimentTracker()
    experiments = experiment_tracker.list_experiments()
    completed_experiments = [exp for exp in experiments if exp.get('status') == 'completed']
    
    if completed_experiments:
        col1, col2 = st.columns(2)
        
        with col1:
            selected_exp = st.selectbox(
                "転送する実験を選択",
                options=completed_experiments,
                format_func=lambda x: f"{x['id'][:8]} - {x.get('model_name', 'unknown')}"
            )
        
        with col2:
            output_name = st.text_input(
                "アーカイブ名", 
                value=f"finetuned-model-{selected_exp['id'][:8]}"
            )
        
        if st.button("📦 転送用アーカイブ作成", type="primary"):
            with st.spinner("アーカイブ作成中..."):
                archive_result = create_transfer_archive(selected_exp['id'], output_name)
                
                if archive_result['success']:
                    st.success(f"✅ アーカイブ作成完了: {archive_result['archive_path']}")
                    st.info(f"📏 ファイルサイズ: {archive_result['size_mb']:.1f} MB")
                    
                    # 転送手順を表示
                    st.subheader("📋 転送手順")
                    transfer_commands = f"""# 1. 転送先PCにファイルをコピー
scp {archive_result['archive_path']} user@target-pc:/path/to/destination/

# 2. 転送先PCでアーカイブを展開
tar -xzf {archive_result['filename']}

# 3. Ollamaモデルを作成
ollama create my-finetuned-model -f <(cat <<EOF
FROM ./models/quantized/{archive_result['gguf_filename']}

SYSTEM "あなたは親切で知識豊富な日本語アシスタントです。ユーザーの質問に対して、正確で有益な回答を日本語で提供してください。回答は分かりやすく簡潔にまとめ、必要に応じて具体例を示してください。"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.1
EOF
)"""
                    st.code(transfer_commands, language="bash")
                else:
                    st.error(f"❌ アーカイブ作成エラー: {archive_result['error']}")
    else:
        st.info("📋 転送可能な完了済み実験がありません")
    
    # ===============================
    # 🧹 ストレージクリーンアップ
    # ===============================
    st.header("🧹 ストレージクリーンアップ")
    st.write("ファインチューニング結果ファイルを削除してストレージ容量を確保します。")
    
    # ストレージ使用量を取得
    cleanup_info = get_cleanup_info()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ファインチューニング結果", f"{cleanup_info['finetuned_size_gb']:.2f} GB")
    with col2:
        st.metric("量子化ファイル", f"{cleanup_info['quantized_size_gb']:.2f} GB")
    with col3:
        st.metric("実験データ", f"{cleanup_info['experiments_size_gb']:.2f} GB")
    
    st.warning("⚠️ HuggingFaceベースモデル（gemma-2-2b-it等）は削除されません。")
    
    # クリーンアップオプション
    cleanup_options = st.multiselect(
        "削除対象を選択",
        [
            "ファインチューニング結果 (models/finetuned/)",
            "量子化ファイル (models/quantized/)",
            "実験データ (experiments/)",
            "MLXキャッシュ (models/cache/)",
            "GGUFキャッシュ (models/gguf_cache/)"
        ],
        default=["ファインチューニング結果 (models/finetuned/)", "量子化ファイル (models/quantized/)"]
    )
    
    if cleanup_options:
        confirm_cleanup = st.checkbox("⚠️ 削除を確認します（この操作は取り消せません）")
        
        if confirm_cleanup and st.button("🗑️ クリーンアップ実行", type="secondary"):
            with st.spinner("クリーンアップ実行中..."):
                cleanup_result = perform_cleanup(cleanup_options)
                
                if cleanup_result['success']:
                    st.success(f"✅ クリーンアップ完了: {cleanup_result['freed_gb']:.2f} GB 削除")
                    st.rerun()
                else:
                    st.error(f"❌ クリーンアップエラー: {cleanup_result['error']}")


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