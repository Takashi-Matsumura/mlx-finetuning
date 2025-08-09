# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a complete **REAL MLX fine-tuning** application that performs actual neural network weight updates on Apple Silicon Macs. Unlike simulation-based approaches, this app generates actual LoRA adapter files (.safetensors) with trained weights using the MLX framework.

**GitHub Repository**: mlx-finetuning - A public repository for educational and research purposes in MLX-based fine-tuning.

**Key Achievement**: Successfully implemented and tested real MLX fine-tuning with Google Gemma-2-2b-it, achieving loss reduction from 2.785 to 0.324 in 26.2 seconds, with successful knowledge injection (company-specific information).

## Development Commands

### Setup and Installation

#### Standard Environment (Conda)
- Initial setup: `./setup.sh` (installs dependencies, builds llama.cpp, sets up environment)
- Install dependencies: `pip install -r requirements.txt`
- Activate environment: `conda activate llm-finetuning`

#### MLX Environment (Real Fine-tuning)
For actual neural network weight updates with MLX:
```bash
# Create virtual environment
python3 -m venv mlx_env
source mlx_env/bin/activate

# Install MLX dependencies (Apple Silicon only)
pip install mlx mlx-lm

# Install app dependencies
pip install streamlit pandas numpy transformers
pip install scikit-learn psutil plotly torch pyyaml jinja2
```

### Running the Application

#### Standard Mode
- Start web app: `streamlit run app.py`
- Development mode: `streamlit run app.py --server.runOnSave true`
- Access at: `http://localhost:8501`

#### MLX Mode (Real Fine-tuning)
```bash
source mlx_env/bin/activate
streamlit run app.py --server.port 8506 --server.address 0.0.0.0
# Access at: http://localhost:8506
```

#### Background Mode (for connection issues)
```bash
nohup streamlit run app.py --server.port 8506 --server.address 0.0.0.0 > streamlit.log 2>&1 &
```

### Testing
- Run all tests: `pytest`
- Run specific tests: `pytest tests/test_data_processor.py`
- Run with coverage: `pytest --cov=src tests/`

### Development Tools
- Code formatting: `black .` and `isort .`
- Type checking: `mypy .`
- Linting: `flake8`

## Architecture Overview

### Core Components
- **src/data_processor.py**: Dataset loading, preprocessing, validation, and formatting
- **src/trainer.py**: Training orchestration and interface (delegates to MLX trainer)
- **src/mlx_trainer.py**: **CORE** - Real MLX fine-tuning implementation using `python -m mlx_lm lora`
- **src/quantizer.py**: GGUF quantization with llama.cpp integration and PyTorch fallbacks
- **src/ollama_integration.py**: Ollama server integration with Modelfile generation
- **src/experiment_tracker.py**: Comprehensive experiment logging and metrics tracking

### Utilities
- **src/utils/japanese_utils.py**: Japanese text normalization and validation
- **src/utils/memory_monitor.py**: Memory usage monitoring and optimization recommendations
- **src/utils/validators.py**: Data validation, model validation, and configuration validation

### Configuration
- **config/default_config.yaml**: Training hyperparameters, LoRA settings, system configuration
- **config/models.yaml**: Available models, task types, and dataset templates

### Data Flow

#### Standard Mode (Simulation)
1. **Data Preparation**: CSV/JSON → preprocessed JSONL → train/val/test splits
2. **Fine-tuning**: Base model + dataset → simulated training metrics
3. **Quantization**: PyTorch model → GGUF → quantized GGUF
4. **Deployment**: Quantized model → Ollama registration → ready for inference

#### MLX Mode (Real Fine-tuning) ✅ IMPLEMENTED & TESTED
1. **Data Preparation**: CSV/JSON → preprocessed JSONL → MLX training format
2. **Model Download**: HuggingFace gated model → MLX format conversion with authentication
3. **Real Fine-tuning**: `python -m mlx_lm lora` → actual gradient computation → .safetensors LoRA adapters
4. **Quantization**: MLX model → GGUF conversion → quantized formats
5. **Deployment**: LoRA + base model → Ollama Modelfile → local inference ready

**実績**: 
- Google Gemma-2-2b-it fine-tuning: 48 iterations, 26.2s, loss 2.785→0.324
- Company information injection: 33.5s, loss 0.5, successful knowledge transfer
- Both models successfully integrated into Ollama and tested

## Key Features
- Streamlit web interface with 7 main pages (Home, Dataset, Training, Quantization, Ollama, Experiments, Settings)
- **Real MLX fine-tuning**: Actual neural network weight updates with Apple Silicon optimization
- Real-time memory monitoring and batch size recommendations
- Comprehensive experiment tracking with metrics visualization
- Support for multiple model formats and quantization levels
- Japanese text processing and validation
- Error handling and progress monitoring throughout the pipeline
- **LoRA fine-tuning**: Low-Rank Adaptation for efficient training
- **Weight file generation**: .safetensors files with actual trained weights

## Supported Models
- ELYZA Japanese models (Llama-3-ELYZA-JP-8B)
- Google Gemma 2 (2B, 9B variants)
- Meta Llama 3.1 (8B Instruct)

## System Requirements

### Standard Environment
- macOS 13.0以上
- Apple Silicon Mac (M1/M2/M3/M4)
- メモリ 16GB以上推奨
- Python 3.11以上

### MLX Environment (Real Fine-tuning)
- **Apple Silicon Mac 必須** (MLXはApple Silicon専用)
- **MLX Library**: `pip install mlx mlx-lm`
- **Additional dependencies**: `pip install scikit-learn psutil plotly`
- **Memory**: 8GB以上（実際の重み更新のため）

### Troubleshooting Dependencies
Common missing modules when setting up MLX environment:
```bash
# If missing scikit-learn
pip install scikit-learn

# If missing psutil  
pip install psutil

# If missing plotly
pip install plotly

# If Streamlit connection issues
pkill -f streamlit
nohup streamlit run app.py --server.port 8506 --server.address 0.0.0.0 > streamlit.log 2>&1 &
```

## Technical Implementation Details

### MLX Fine-tuning Implementation (src/mlx_trainer.py)
**Core Method**: `_run_mlx_training()`
- Uses command-line approach: `python -m mlx_lm lora` for stability
- Implements timestamp-based unique paths to avoid cache conflicts
- Handles HuggingFace authentication for gated models
- Generates real .safetensors LoRA adapter files

### Successful Troubleshooting Resolved
1. **MLX model cache conflicts**: Fixed with timestamp-based unique directories
2. **HuggingFace 401 errors**: Resolved with proper token authentication  
3. **Disk space issues**: Multiple 26GB+ cleanups of Docker containers
4. **Import errors**: Fixed missing dependencies (scikit-learn, psutil, plotly)
5. **Quantization limitations**: Implemented PyTorch fallbacks for Gemma2 models
6. **Ollama integration**: Fixed path resolution and Modelfile generation

## GitHub Repository Management

### Files Included in Repository
**Core Application**:
- `app.py`, `src/`, `config/`, `tests/`, `requirements.txt`, `setup.sh`
- `sample_data.csv`, `sample_data_extended.csv` (sample datasets)
- `data/templates/` (data format templates)
- `CLAUDE.md`, `README.md` (documentation)
- `.gitignore` (repository management)

### Files Excluded (.gitignore)
**Large/Generated Files**:
- `llama.cpp/` (35GB+ external library)
- `mlx_env/` (virtual environment)
- `models/` (multi-GB model files and cache)
- `experiments/`, `logs/` (runtime data)

**Security**: All sensitive information (API keys, tokens) are retrieved from environment variables, not hardcoded.

### Quantization Workflow (src/quantizer.py)
- Primary: llama.cpp GGUF conversion
- Fallback: PyTorch model download when MLX conversion unsupported
- Formats: Q4_K_M, Q5_K_M, Q8_0 with automatic optimization
- Integration: Automatic Ollama model creation with custom parameters

### Experiment Tracking (src/experiment_tracker.py)
- Unique experiment IDs with timestamp tracking
- Real-time metrics logging (loss, perplexity, duration)
- JSON-based experiment persistence
- Complete experiment history and comparison

## Language Preferences
- All responses and UI text should be in Japanese
- Code comments and documentation can be in English  
- Error messages and user feedback should be in Japanese

## Critical Files and Locations
- **Models**: `./models/finetuned/{experiment_id}/` - Contains LoRA adapters
- **Experiments**: `./experiments/{experiment_id}/` - Contains training logs and metrics
- **Ollama Models**: Created in Ollama with names like `gemma2-finetuned-{experiment_id}`
- **Quantized**: Automatically moved to Ollama model storage