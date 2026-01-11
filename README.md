# Parakeet TDT Transcription with ONNX Runtime

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Parakeet TDT** is a high-performance implementation of NVIDIA's [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model using [ONNX Runtime](https://onnxruntime.ai/), specifically optimized for ultra-fast, local transcription on consumer CPUs.

This version introduces **Batch Processing** capabilities and dynamic **Priority Modes** to balance performance and system responsiveness.

> [!NOTE]
> This project is a refined fork and continuation of the original work by [groxaxo](https://github.com/groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai).

## 🚀 Key Features

- **Extreme CPU Efficiency**: Optimized for modern CPUs using ONNX Runtime with INT8 quantization.
- **Batch Processing**: Queue multiple files for background transcription with real-time progress tracking and ETA calculation.
- **Priority Modes**: Switch between "High Priority" (full CPU utilization) and "Low Priority" (background mode) via configuration.
- **OpenAI Compatible**: Drop-in replacement for OpenAI's transcription API.
- **Multilingual**: Automatic language detection supporting 25 European languages.
- **Web UI**: Built-in dashboard for monitoring progress, adjusting settings, and easy drag-and-drop transcription.

## 🌍 Multilingual Support

The model automatically identifies and transcribes speech in any of the **25 supported languages**:

English, Spanish, French, Russian, German, Italian, Polish, Ukrainian, Romanian, Dutch, Hungarian, Greek, Swedish, Czech, Bulgarian, Portuguese, Slovak, Croatian, Danish, Finnish, Lithuanian, Slovenian, Latvian, Estonian, Maltese.

## 📊 Performance Benchmark

Parakeet TDT (CPU) outperforms standard Whisper implementations and competes with GPU-accelerated versions.

| Implementation | Hardware | Model | Precision | Speedup |
| --- | --- | --- | --- | --- |
| **Parakeet TDT** (Ours) | **CPU** (i7-12700KF) | **TDT 0.6B v3** | **int8** | **~29.7x** |
| **Parakeet TDT** (Ours) | **CPU** (i7-4790) | **TDT 0.6B v3** | **int8** | **~17.0x** |
| faster-whisper | GPU (RTX 3070 Ti) | Large-v2 | int8 | 13.2x |
| faster-whisper | CPU (i7-12700K) | Small | int8 | 7.6x |

*   **Speedup Factor**: Audio Duration / Processing Time. Higher is better.
*   **Real Time Factor (RTF)**: ~0.033 on modern hardware.

## ⚙️ Installation & Setup

### Requirements
- **Python 3.10+**
- **FFmpeg** (installed and in system PATH)

### Quick Start (Conda Recommended)

```bash
# Clone the repository
git clone https://github.com/oooskarrr/parakeet-fastapi
cd parakeet-tdt-0.6b-v3-fastapi-openai

# Create and activate environment
conda create -n parakeet-onnx python=3.10
conda activate parakeet-onnx

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

The server will be available at `http://localhost:5092`.

## 🛠️ Configuration & Priority Modes

You can customize the application behavior in `config.yaml`.

### CPU Priority
The app supports two priority modes to manage CPU thread allocation:
- **High Priority**: Uses maximum available cores for fastest transcription.
- **Low Priority**: Uses a reduced number of threads, suitable for background tasks without slowing down your system.

```yaml
cpu:
  priority_mode: "high"  # Choices: "high", "low"
  high_priority_threads: 6
  low_priority_threads: 3
```

## 📦 Batch Processing

The server includes a robust background task system. You can upload multiple files at once via the Web UI or the `/v1/audio/transcriptions/batch` endpoint.
- **Persistence**: Jobs are saved to `jobs.json` and will resume (as pending) if the server restarts.
- **Observability**: Real-time progress, status updates, and estimated time of completion (ETA) for each file and the entire batch.

## 🔌 API Usage

### OpenAI-Compatible Client (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:5092/v1",
    api_key="sk-no-key-required"
)

audio_file = open("audio.mp3", "rb")
transcript = client.audio.transcriptions.create(
  model="parakeet-tdt-0.6b-v3",
  file=audio_file,
  response_format="text"
)

print(transcript)
```

## 🖥️ Open WebUI Integration

Use Parakeet TDT as a local backend for [Open WebUI](https://openwebui.com/):

1. Go to **Settings -> Audio** in Open WebUI.
2. Set **STT Engine** to `OpenAI`.
3. Set **OpenAI Base URL** to `http://YOUR_IP:5092/v1`.
4. Set **OpenAI API Key** to `sk-no-key-required`.
5. Set **STT Model** to `parakeet-tdt-0.6b-v3`.

## 🙏 Acknowledgments

- **[Original Author & Repository](https://github.com/groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai)**
- **[Shadowfita](https://github.com/Shadowfita/parakeet-tdt-0.6b-v2-fastapi)** - For the foundation FastAPI implementation.
- **[NVIDIA](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)** - For the open-source Parakeet TDT model family.
