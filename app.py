import sys

sys.stdout = sys.stderr

import os, sys, json, math, re, threading
import logging
import time
import shutil
import uuid
import subprocess
import datetime
import psutil
from typing import List, Tuple, Optional
from werkzeug.utils import secure_filename

import flask
from flask import Flask, request, jsonify, render_template, Response
from waitress import serve
from pathlib import Path
from huggingface_hub import snapshot_download

# Import configuration
from config import config
from job_queue import JobQueue, JobStatus

ROOT_DIR = Path(os.getcwd()).as_posix()
# Configure environment
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
if sys.platform == "win32":
    os.environ["PATH"] = ROOT_DIR + f";{ROOT_DIR}/ffmpeg;" + os.environ["PATH"]

# Model storage optimization
MODEL_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
LOCAL_MODEL_DIR = os.path.join(ROOT_DIR, "models", "parakeet-tdt")
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# Configure logging
log_level = config.log_level.upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(config.log_file, encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# Server configuration from config
host = config.server_host
port = config.server_port
threads = config.waitress_threads

# Audio processing configuration from config
CHUNK_MINUTE = config.chunk_duration_minutes
SILENCE_THRESHOLD = config.silence_threshold
SILENCE_MIN_DURATION = config.silence_min_duration
SILENCE_SEARCH_WINDOW = config.silence_search_window
SILENCE_DETECT_TIMEOUT = config.silence_detect_timeout
MIN_SPLIT_GAP = config.silence_min_split_gap


try:
    logger.info("Loading Parakeet TDT 0.6B V3 ONNX model with INT8 quantization...")
    import onnx_asr

    import onnxruntime as ort
    
    # Get CPU thread count from configuration
    cpu_threads = config.cpu_threads
    logger.info(f"CPU Priority Mode: {config.cpu_priority_mode}")
    logger.info(f"Using {cpu_threads} CPU threads for inference")
    
    # Configure session options for optimal CPU performance
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = cpu_threads
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load model with configuration settings
    
    # Check if model files exist locally in the short folder
    if not os.path.exists(LOCAL_MODEL_DIR) or not os.listdir(LOCAL_MODEL_DIR):
        logger.info(f"Downloading model components to {LOCAL_MODEL_DIR}...")
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=LOCAL_MODEL_DIR,
            local_dir_use_symlinks=False,
            allow_patterns=["config.json", "vocab.txt", "*.int8.onnx*"]
        )
    
    # Load model from local path
    asr_model = onnx_asr.load_model(
        MODEL_REPO,
        path=LOCAL_MODEL_DIR,
        quantization=config.model_quantization,
        providers=[config.model_provider],
        sess_options=sess_options,
    ).with_timestamps()
    
    logger.info(f"Model loaded successfully: {config.model_name} ({config.model_quantization})")
    logger.info(f"Provider: {config.model_provider}")
    logger.info(f"Language auto-detect: {'enabled' if config.language_auto_detect else 'disabled'}")
    logger.info(f"Default language: {config.default_language}")
except Exception as e:
    logger.error(f"Model loading failed: {e}", exc_info=True)
    sys.exit()

logger.info("=" * 50)


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = config.upload_folder
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = config.max_file_size_mb * 1024 * 1024
os.makedirs(config.transcription_folder, exist_ok=True)

# Progress tracking
progress_tracker = {}
# Initialize Job Queue
job_queue = JobQueue()


def get_transcription_path(original_filename: str, extension: str) -> Path:
    """
    Generate safe transcription file path, handling collisions.
    
    Args:
        original_filename: Original audio filename
        extension: File extension to use (e.g., '.txt', '.srt')
        
    Returns:
        Path object for the transcription file
    """
    base_name = Path(original_filename).stem
    safe_name = secure_filename(base_name)
    output_path = Path(config.transcription_folder) / f"{safe_name}{extension}"
    
    # Handle collisions by appending counter
    counter = 1
    while output_path.exists():
        output_path = Path(config.transcription_folder) / f"{safe_name}_{counter}{extension}"
        counter += 1
    
    return output_path


def get_audio_duration(file_path: str) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout)
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning(f"Could not get duration of file '{file_path}': {e}")
        return 0.0


def detect_silence_points(file_path: str, silence_thresh: str = SILENCE_THRESHOLD, 
                          silence_duration: float = SILENCE_MIN_DURATION,
                          total_duration: Optional[float] = None) -> List[Tuple[float, float]]:
    """
    Detect silence points in audio file using ffmpeg's silencedetect filter.
    
    Args:
        file_path: Path to audio file
        silence_thresh: Silence threshold in dB (e.g., "-40dB")
        silence_duration: Minimum silence duration in seconds
        total_duration: Total duration of audio (used to close trailing silence)
        
    Returns:
        List of tuples (silence_start, silence_end) in seconds
    """
    # Validate file exists
    if not os.path.exists(file_path):
        logger.error(f"Audio file '{file_path}' not found for silence detection")
        return []
    
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i", file_path,
        "-af", f"silencedetect=noise={silence_thresh}:d={silence_duration}",
        "-f", "null",
        "-"
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=SILENCE_DETECT_TIMEOUT)
        
        # Parse stderr output for silence intervals
        silence_points = []
        silence_start = None
        
        for line in result.stderr.splitlines():
            if 'silence_start:' in line:
                try:
                    silence_start = float(line.split('silence_start:')[1].split()[0])
                except (ValueError, IndexError):
                    silence_start = None
            elif 'silence_end:' in line and silence_start is not None:
                try:
                    silence_end = float(line.split('silence_end:')[1].split()[0])
                    silence_points.append((silence_start, silence_end))
                    silence_start = None
                except (ValueError, IndexError):
                    pass
        
        # Close trailing silence if audio ended during silence
        if silence_start is not None and total_duration is not None:
            silence_points.append((silence_start, total_duration))
        
        return silence_points
    except subprocess.TimeoutExpired:
        logger.warning(f"Silence detection exceeded {SILENCE_DETECT_TIMEOUT}s timeout")
        return []
    except (subprocess.CalledProcessError, OSError) as e:
        logger.error(f"Error running FFmpeg for silence detection: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error detecting silence: {e}")
        return []


def find_optimal_split_points(total_duration: float, target_chunk_duration: float, 
                               silence_points: List[Tuple[float, float]], 
                               search_window: float = SILENCE_SEARCH_WINDOW,
                               min_gap: float = MIN_SPLIT_GAP) -> List[float]:
    """
    Find optimal split points based on silence detection.
    
    Args:
        total_duration: Total audio duration in seconds
        target_chunk_duration: Target chunk size in seconds
        silence_points: List of (start, end) tuples for silence periods
        search_window: Search window in seconds around target split point
        min_gap: Minimum gap between split points to prevent 0-length chunks
        
    Returns:
        List of split points in seconds
    """
    if not silence_points or total_duration <= target_chunk_duration:
        return []
    
    split_points = []
    prev = 0.0
    num_chunks = math.ceil(total_duration / target_chunk_duration)
    
    for i in range(1, num_chunks):
        target_time = i * target_chunk_duration
        search_start = max(0.0, target_time - search_window)
        search_end = min(total_duration, target_time + search_window)
        
        # Find silence points that overlap with the search window
        candidates = [
            (start, end) for (start, end) in silence_points
            if start <= search_end and end >= search_start
        ]
        
        chosen = None
        if candidates:
            # Sort candidates by distance from target time
            candidates_sorted = sorted(
                candidates,
                key=lambda silence_range: abs(((silence_range[0] + silence_range[1]) / 2.0) - target_time)
            )
            # Find first candidate that satisfies minimum gap constraint
            for start, end in candidates_sorted:
                split_point = (start + end) / 2.0
                if split_point > prev + min_gap and split_point <= total_duration - min_gap:
                    chosen = split_point
                    break
        
        if chosen is None:
            # Fallback: target time, but enforce monotonicity and bounds
            chosen = max(prev + min_gap, min(target_time, total_duration - min_gap))
            # Ensure chosen doesn't exceed total_duration
            if chosen > total_duration:
                chosen = None  # Skip this split point if not feasible
        
        split_points.append(chosen)
        prev = chosen
    
    # Filter out None values if any splits were skipped
    split_points = [sp for sp in split_points if sp is not None]
    
    return split_points


def format_srt_time(seconds: float) -> str:
    delta = datetime.timedelta(seconds=seconds)
    s = str(delta)
    if "." in s:
        parts = s.split(".")
        integer_part = parts[0]
        fractional_part = parts[1][:3]
    else:
        integer_part = s
        fractional_part = "000"

    if len(integer_part.split(":")) == 2:
        integer_part = "0:" + integer_part

    return f"{integer_part},{fractional_part}"


def segments_to_srt(segments: list) -> str:
    srt_content = []
    for i, segment in enumerate(segments):
        start_time = format_srt_time(segment["start"])
        end_time = format_srt_time(segment["end"])
        text = segment["segment"].strip()

        if text:
            srt_content.append(str(i + 1))
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")

    return "\n".join(srt_content)


def segments_to_vtt(segments: list) -> str:
    vtt_content = ["WEBVTT", ""]
    for i, segment in enumerate(segments):
        start_time = format_srt_time(segment["start"]).replace(",", ".")
        end_time = format_srt_time(segment["end"]).replace(",", ".")
        text = segment["segment"].strip()

        if text:
            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(text)
            vtt_content.append("")
    return "\n".join(vtt_content)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/parakeet.png")
def serve_logo():
    return flask.send_file("parakeet.png", mimetype="image/png")


@app.route("/health")
def health():
    return jsonify(
        {"status": "healthy", "model": "parakeet-tdt-0.6b-v3", "speedup": "20.7x"}
    )


@app.route("/docs")
def swagger_ui():
    """Serve Swagger UI"""
    return render_template("swagger.html")


@app.route("/openapi.json")
def openapi_spec():
    """Return OpenAPI Specification"""
    return jsonify({
        "openapi": "3.0.0",
        "info": {
            "title": "Parakeet Transcription API",
            "description": "High-performance ONNX-optimized speech transcription API compatible with OpenAI.",
            "version": "1.0.0"
        },
        "servers": [{"url": "http://100.85.200.51:5092"}],
        "paths": {
            "/v1/audio/transcriptions": {
                "post": {
                    "summary": "Transcribe Audio",
                    "description": "Transcribes audio into the input language. Supports real-time streaming progress.",
                    "operationId": "transcribe_audio",
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "format": "binary",
                                            "description": "The audio file object (not file name) to transcribe."
                                        },
                                        "model": {
                                            "type": "string",
                                            "default": "whisper-1",
                                            "description": "ID of the model to use."
                                        },
                                        "response_format": {
                                            "type": "string",
                                            "default": "json",
                                            "enum": ["json", "text", "srt", "verbose_json", "vtt"],
                                            "description": "The format of the transcript output."
                                        }
                                    },
                                    "required": ["file"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Successful Response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string"}
                                        }
                                    }
                                },
                                "text/plain": {
                                    "schema": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        }
    })


@app.route("/progress/<job_id>")
def get_progress(job_id):
    """Get transcription progress for a job"""
    if job_id in progress_tracker:
        return jsonify(progress_tracker[job_id])
    return jsonify({"status": "not_found"}), 404


@app.route("/status")
def get_status():
    """Get status of the most recent active job"""
    for job_id, progress in progress_tracker.items():
        if progress.get("status") == "processing":
            return jsonify({"job_id": job_id, **progress})
    return jsonify({"status": "idle"})


@app.route("/metrics")
def get_metrics():
    """Get real-time CPU and RAM metrics"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    return jsonify({
        "cpu_percent": cpu_percent,
        "ram_percent": memory.percent,
        "ram_used_gb": round(memory.used / (1024**3), 2),
        "ram_total_gb": round(memory.total / (1024**3), 2)
    })


@app.route("/config")
def get_config():
    """Get current configuration"""
    return jsonify({
        "server": {
            "host": config.server_host,
            "port": config.server_port
        },
        "cpu": {
            "priority_mode": config.cpu_priority_mode,
            "threads": config.cpu_threads,
            "waitress_threads": config.waitress_threads
        },
        "model": {
            "name": config.model_name,
            "quantization": config.model_quantization,
            "provider": config.model_provider
        },
        "language": {
            "auto_detect": config.language_auto_detect,
            "default_language": config.default_language
        },
        "audio": {
            "chunk_duration_minutes": config.chunk_duration_minutes,
            "sample_rate": config.sample_rate,
            "channels": config.channels
        },
        "silence": {
            "threshold": config.silence_threshold,
            "min_duration": config.silence_min_duration,
            "search_window": config.silence_search_window,
            "detect_timeout": config.silence_detect_timeout,
            "min_split_gap": config.silence_min_split_gap
        },
        "upload": {
            "max_file_size_mb": config.max_file_size_mb,
            "upload_folder": config.upload_folder,
            "transcription_folder": config.transcription_folder
        },
        "logging": {
            "level": config.log_level,
            "log_file": config.log_file
        },
        "progress": {
            "enable_partial_text": config.enable_partial_text
        }
    })


@app.route("/config", methods=["POST"])
def update_config():
    """Update configuration (requires restart for some changes)"""
    try:
        data = request.get_json()
        
        # Update CPU priority mode
        if "cpu" in data and "priority_mode" in data["cpu"]:
            if data["cpu"]["priority_mode"] in ["high", "low"]:
                config.set("cpu.priority_mode", data["cpu"]["priority_mode"])
        
        # Update language settings
        if "language" in data:
            if "auto_detect" in data["language"]:
                config.set("language.auto_detect", bool(data["language"]["auto_detect"]))
            if "default_language" in data["language"]:
                config.set("language.default_language", data["language"]["default_language"])
        
        # Update audio settings
        if "audio" in data:
            if "chunk_duration_minutes" in data["audio"]:
                config.set("audio.chunk_duration_minutes", float(data["audio"]["chunk_duration_minutes"]))
        
        # Update silence settings
        if "silence" in data:
            if "threshold" in data["silence"]:
                config.set("silence.threshold", data["silence"]["threshold"])
            if "min_duration" in data["silence"]:
                config.set("silence.min_duration", float(data["silence"]["min_duration"]))
            if "search_window" in data["silence"]:
                config.set("silence.search_window", float(data["silence"]["search_window"]))
            if "min_split_gap" in data["silence"]:
                config.set("silence.min_split_gap", float(data["silence"]["min_split_gap"]))
        
        # Update progress settings
        if "progress" in data and "enable_partial_text" in data["progress"]:
            config.set("progress.enable_partial_text", bool(data["progress"]["enable_partial_text"]))
        
        # Save configuration to file
        config.save()
        
        logger.info("Configuration updated successfully")
        
        return jsonify({
            "status": "success",
            "message": "Configuration updated. Some changes may require restart.",
            "config": get_config().get_json()
        })
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}", exc_info=True)
        return jsonify({"error": "Failed to update configuration", "details": str(e)}), 500



def process_batch_job(job):
    """
    Process a single batch job in the background.
    Replicates the logic of transcribe_audio but adapted for background execution.
    """
    unique_id = job["job_id"]
    batch_id = job.get("batch_id")
    original_filename = job["filename"]
    temp_original_path = job["temp_path"]
    
    # Check if file exists
    if not os.path.exists(temp_original_path):
        logger.error(f"[{unique_id}] File not found: {temp_original_path}")
        job_queue.fail_job(unique_id, "File not found")
        return

    logger.info(f"[{unique_id}] Starting batch job for: {original_filename}")
    
    target_wav_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{unique_id}.wav")
    temp_files_to_clean = [temp_original_path, target_wav_path]
    
    try:
        # Default settings for batch
        language = job.get("language")
        if not language and config.language_auto_detect:
            language = None
        elif not language:
            language = config.default_language
            
        job_queue.update_job_progress(unique_id, {"status": "converting", "percent": 5})

        # 1. Convert to WAV
        ffmpeg_command = [
            "ffmpeg", "-nostdin", "-y", "-i", temp_original_path,
            "-ac", str(config.channels), "-ar", str(config.sample_rate),
            target_wav_path
        ]
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg conversion failed: {result.stderr}")

        # 2. Get Duration
        total_duration = get_audio_duration(target_wav_path)
        if total_duration == 0:
            raise Exception("Audio duration is 0")
        job_queue.set_job_duration(unique_id, total_duration)
        job_queue.update_job_progress(unique_id, {"status": "analyzing", "percent": 10})

        # 3. Intelligent Chunking
        CHUNK_DURATION_SECONDS = config.chunk_duration_minutes * 60
        split_points = []
        if total_duration > CHUNK_DURATION_SECONDS:
            silence_points = detect_silence_points(target_wav_path, total_duration=total_duration)
            if silence_points:
                split_points = find_optimal_split_points(
                    total_duration, CHUNK_DURATION_SECONDS, silence_points,
                    search_window=config.silence_search_window
                )

        if split_points:
            chunk_boundaries = [0.0] + split_points + [total_duration]
            num_chunks = len(chunk_boundaries) - 1
        else:
            num_chunks = math.ceil(total_duration / CHUNK_DURATION_SECONDS)
            chunk_boundaries = [min(i * CHUNK_DURATION_SECONDS, total_duration) for i in range(num_chunks + 1)]

        job_queue.update_job_progress(unique_id, {
            "status": "transcribing", 
            "total_chunks": num_chunks,
            "percent": 15
        })

        # 4. Process Chunks
        chunk_paths = []
        if num_chunks > 1:
            for i in range(num_chunks):
                start = chunk_boundaries[i]
                duration = chunk_boundaries[i+1] - start
                c_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{unique_id}_chunk_{i}.wav")
                chunk_paths.append(c_path)
                temp_files_to_clean.append(c_path)
                
                subprocess.run([
                    "ffmpeg", "-nostdin", "-y", "-ss", str(start), "-t", str(duration),
                    "-i", target_wav_path, "-ac", str(config.channels), "-ar", str(config.sample_rate),
                    "-c:a", "pcm_s16le", c_path
                ], capture_output=True)
        else:
            chunk_paths.append(target_wav_path)

        # 5. Inference Loop
        all_segments = []
        cumulative_time = 0.0
        
        # Resume Logic: Check for checkpoint
        checkpoint_path = os.path.join(config.transcription_folder, f"{unique_id}_partial.json")
        start_chunk_index = 0
        
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
                    all_segments = checkpoint_data.get("segments", [])
                    start_chunk_index = checkpoint_data.get("last_chunk_index", -1) + 1
                    cumulative_time = checkpoint_data.get("cumulative_time", 0.0)
                    logger.info(f"[{unique_id}] Resuming from checkpoint: chunk {start_chunk_index}, {len(all_segments)} segments loaded.")
            except Exception as e:
                logger.error(f"[{unique_id}] Failed to load checkpoint: {e}")
        
        # Calculate start time for resumption if needed
        # (cumulative_time should be accurate from checkpoint, but let's verify loop integrity)
        if start_chunk_index > 0 and start_chunk_index < len(chunk_boundaries) - 1:
             # If we just trust cumulative_time from checkpoint we are good.
             pass
        elif start_chunk_index > 0:
             # Recalculate cumulative time if not trusting checkpoint
             cumulative_time = 0.0
             for i in range(start_chunk_index):
                 # This logic mirrors the loop below
                 dur = chunk_boundaries[i+1] - chunk_boundaries[i] if i < len(chunk_boundaries)-1 else total_duration
                 cumulative_time += dur
        
        
        def clean_text(text):
            if not text: return ""
            text = text.replace("\u2581", " ").strip()
            text = re.sub(r"\s+", " ", text)
            text = text.replace(" '", "'")
            return re.sub(r"(\S)\$", r"\1 $", text)

        for i, c_path in enumerate(chunk_paths):
            if i < start_chunk_index:
                continue

            chunk_start_time = time.time()
            logger.info(f"[{unique_id}] Transcribing chunk {i + 1}/{num_chunks}...")

            result = asr_model.recognize(c_path)
            
            # --- Timing & ETA Logic ---
            chunk_duration = time.time() - chunk_start_time
            
            # Calculate RTF for this chunk
            if i < len(chunk_boundaries) - 1:
                 chunk_audio_duration = chunk_boundaries[i+1] - chunk_boundaries[i]
            else:
                 chunk_audio_duration = total_duration - chunk_boundaries[i]

            chunk_rtf = chunk_duration / chunk_audio_duration if chunk_audio_duration > 0 else 0
            
            # File ETA: avg time per chunk * remaining chunks
            remaining_chunks = num_chunks - (i + 1)
            file_eta_seconds = chunk_duration * remaining_chunks
            file_eta_str = f"{int(file_eta_seconds // 60)}m {int(file_eta_seconds % 60)}s" if file_eta_seconds >= 60 else f"{int(file_eta_seconds)}s"
            
            # Batch ETA: Use the same logic as the API endpoint for consistency
            batch_eta_str = "Calculating..."
            if batch_id:
                batch_status = job_queue.get_batch_status(batch_id)
                if batch_status:
                    b_total_dur = 0
                    b_completed_dur = 0
                    
                    for b_job in batch_status["jobs"]:
                        b_job_dur = b_job.get("duration", 0)
                        b_total_dur += b_job_dur
                        if b_job["status"] == "completed":
                            b_completed_dur += b_job_dur
                        elif b_job["job_id"] == unique_id:
                            # Use current file progress
                            b_completed_dur += b_job_dur * ((i + 1) / num_chunks)
                        elif b_job["status"] == "processing":
                            # Should not happen in single-worker but for safety:
                            b_completed_dur += b_job_dur * (b_job.get("progress", {}).get("percent", 0) / 100)
                    
                    b_remaining_dur = max(0, b_total_dur - b_completed_dur)
                    # Use a slightly more conservative RTF if it's the first chunk
                    # or just use the current one if it seems sane.
                    # Model RTF is usually 0.05-0.15. If it's > 1.0, it's likely overhead.
                    adj_rtf = min(chunk_rtf, 1.0) if i > 0 else min(chunk_rtf, 0.5)
                    
                    b_eta_sec = b_remaining_dur * adj_rtf
                    if b_eta_sec >= 60:
                        batch_eta_str = f"{int(b_eta_sec // 60)}m {int(b_eta_sec % 60)}s"
                    else:
                        batch_eta_str = f"{int(b_eta_sec)}s"

            logger.info(f"[{unique_id}] Chunk {i + 1}/{num_chunks} in {chunk_duration:.2f}s (RTF: {chunk_rtf:.3f}) | File ETA: {file_eta_str} | Batch ETA: {batch_eta_str}")

            if result and result.text:
                start_base = result.timestamps[0] if result.timestamps else 0
                end_base = result.timestamps[-1] if len(result.timestamps) > 1 else start_base + 0.1
                
                segment = {
                    "start": start_base + cumulative_time,
                    "end": end_base + cumulative_time,
                    "segment": clean_text(result.text)
                }
                all_segments.append(segment)
            
            # Update Progress
            chunk_dur = chunk_boundaries[i+1] - chunk_boundaries[i] if i < len(chunk_boundaries)-1 else total_duration
            cumulative_time += chunk_dur
            
            percent = 15 + int((i + 1) / num_chunks * 80)
            
            # Save Checkpoint
            try:
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "job_id": unique_id,
                        "segments": all_segments,
                        "last_chunk_index": i,
                        "cumulative_time": cumulative_time,
                        "updated_at": time.time()
                    }, f)
            except Exception as e:
                logger.warning(f"[{unique_id}] Failed to save checkpoint: {e}")

            job_queue.update_job_progress(unique_id, {
                "current_chunk": i + 1,
                "percent": percent,
                "eta": file_eta_str,
                "rtf": round(chunk_rtf, 3),
                "partial_text": " ".join([s["segment"] for s in all_segments[-3:]])
            })

        # 6. Save Results
        full_text = " ".join([seg["segment"] for seg in all_segments])
        txt_path = get_transcription_path(original_filename, ".txt")
        srt_path = get_transcription_path(original_filename, ".srt")
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)
            
        srt_output = segments_to_srt(all_segments)
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_output)
            
        logger.info(f"[{unique_id}] Batch job completed. Saved to {txt_path}")
        
        job_queue.complete_job(unique_id, {
            "text_path": str(txt_path),
            "srt_path": str(srt_path),
            "duration": total_duration
        })

    except Exception as e:
        logger.error(f"[{unique_id}] Batch job failed: {e}", exc_info=True)
        job_queue.fail_job(unique_id, str(e))
    finally:
        logger.debug(f"[{unique_id}] Cleaning up temporary files...")
        
        # Add checkpoint to cleanup list
        try:
            checkpoint_path = os.path.join(config.transcription_folder, f"{unique_id}_partial.json")
            abs_chk_path = os.path.abspath(checkpoint_path)
            
            if os.path.exists(abs_chk_path):
                 for attempt in range(3):
                    try:
                        os.remove(abs_chk_path)
                        logger.info(f"[{unique_id}] SUCCESSFULLY REMOVED checkpoint file: {abs_chk_path}")
                        break
                    except OSError as e:
                        logger.warning(f"[{unique_id}] Cleanup attempt {attempt+1} failed for {abs_chk_path}: {e}")
                        time.sleep(0.5)
            else:
                 logger.debug(f"[{unique_id}] Checkpoint file not found during cleanup: {abs_chk_path}")

            for f_path in temp_files_to_clean:
                if os.path.exists(f_path):
                    try:
                        os.remove(f_path)
                    except OSError:
                        pass
        except Exception as e:
             logger.error(f"[{unique_id}] Cleanup routine failed: {e}")
             
        logger.debug(f"[{unique_id}] Temporary files cleaned.")
def worker_loop():
    """Background worker thread loop"""
    logger.info("Worker thread started")
    while True:
        try:
            job = job_queue.claim_next_job()
            if job:
                process_batch_job(job)
            else:
                time.sleep(1)
        except Exception as e:
            logger.error(f"Worker thread error: {e}")
            time.sleep(5)

@app.route("/v1/audio/transcriptions/batch", methods=["POST"])
def create_batch_transcription():
    """Batch upload endpoint"""
    if "files" not in request.files:
        files = request.files.getlist("file")
        if not files:
             return jsonify({"error": "No files provided"}), 400
    else:
        files = request.files.getlist("files")

    file_job_info = []
    
    for file in files:
        if not file or not file.filename: continue
        
        original_filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        
        temp_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{unique_id}_{original_filename}")
        file.save(temp_path)
        
        # Pre-calculate duration for accurate batch ETA
        duration = get_audio_duration(temp_path)
        
        file_job_info.append({
            "filename": original_filename,
            "temp_path": temp_path,
            "duration": duration
        })
    
    if not file_job_info:
        return jsonify({"error": "No valid files processed"}), 400
        
    batch_id = job_queue.create_batch(file_job_info)
    logger.info(f"Created batch {batch_id} with {len(file_job_info)} files")
    
    return jsonify({
        "batch_id": batch_id,
        "status": "queued",
        "message": f"Successfully queued {len(file_job_info)} files"
    })

@app.route("/batch/<batch_id>")
def get_batch_info(batch_id):
    """Get batch status with calculated ETAs"""
    status = job_queue.get_batch_status(batch_id)
    if not status:
        return jsonify({"error": "Batch not found"}), 404
    
    # Calculate batch-wide metrics
    total_duration = 0
    completed_duration = 0
    active_job_rtf = 0.15  # Default RTF estimator (approx for CPU)
    rtf_samples = []
    
    for job in status["jobs"]:
        job_dur = job.get("duration", 0)
        total_duration += job_dur
        
        if job["status"] == "completed":
            completed_duration += job_dur
            if job.get("progress", {}).get("rtf"):
                rtf_samples.append(job["progress"]["rtf"])
        elif job["status"] == "processing":
            # Add progress-weighted duration
            completed_duration += job_dur * (job["progress"].get("percent", 0) / 100)
            if job["progress"].get("rtf"):
                rtf_samples.append(job["progress"]["rtf"])
                
    if rtf_samples:
        active_job_rtf = sum(rtf_samples) / len(rtf_samples)
        
    remaining_duration = max(0, total_duration - completed_duration)
    batch_eta_seconds = remaining_duration * active_job_rtf if remaining_duration > 0 else 0
    
    # Format batch ETA
    if batch_eta_seconds > 60:
        batch_eta_str = f"{int(batch_eta_seconds // 60)}m {int(batch_eta_seconds % 60)}s"
    else:
        batch_eta_str = f"{int(batch_eta_seconds)}s"

    status["batch_eta"] = batch_eta_str if batch_eta_seconds > 0 else "0s"
    status["total_duration"] = round(total_duration, 2)
    
    return jsonify(status)

@app.route("/v1/audio/transcriptions", methods=["POST"])
def transcribe_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400

    # OpenAI compatible parameters
    model_name = request.form.get("model", "whisper-1").lower()
    response_format = request.form.get("response_format", "json")
    
    # Language detection parameter (optional, overrides config)
    language_param = request.form.get("language")
    if language_param:
        # Use language from request if provided
        language = language_param.lower()
    elif config.language_auto_detect:
        # Auto-detect language (model will detect automatically)
        language = None
    else:
        # Use default language from config
        language = config.default_language

    # Legacy support
    if model_name == "parakeet_srt_words":
        pass

    original_filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())

    logger.info(f"[{unique_id}] Request received: {original_filename} | Model: {model_name} | Format: {response_format} | Language: {language or 'auto-detect'}")
    temp_original_path = os.path.join(
        app.config["UPLOAD_FOLDER"], f"{unique_id}_{original_filename}"
    )
    target_wav_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{unique_id}.wav")

    temp_files_to_clean = []

    try:
        # Start timing
        request_start_time = time.time()
        chunk_times = []
        
        file.save(temp_original_path)
        temp_files_to_clean.append(temp_original_path)

        logger.debug(f"[{unique_id}] Converting '{original_filename}' to standard WAV format...")
        ffmpeg_command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            temp_original_path,
            "-ac",
            str(config.channels),
            "-ar",
            str(config.sample_rate),
            target_wav_path,
        ]
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"[{unique_id}] FFmpeg error: {result.stderr}")
            return jsonify(
                {"error": "File conversion failed", "details": result.stderr}
            ), 500
        temp_files_to_clean.append(target_wav_path)

        CHUNK_DURATION_SECONDS = CHUNK_MINUTE * 60
        total_duration = get_audio_duration(target_wav_path)
        if total_duration == 0:
            return jsonify({"error": "Cannot process audio with 0 duration"}), 400

        # Use intelligent chunking based on silence detection
        chunk_paths = []
        split_points = []
        
        if total_duration > CHUNK_DURATION_SECONDS:
            logger.debug(f"[{unique_id}] Detecting silence points for intelligent chunking...")
            silence_points = detect_silence_points(target_wav_path, total_duration=total_duration)
            
            if silence_points:
                logger.debug(f"[{unique_id}] Found {len(silence_points)} silence periods")
                split_points = find_optimal_split_points(
                    total_duration,
                    CHUNK_DURATION_SECONDS,
                    silence_points,
                    search_window=SILENCE_SEARCH_WINDOW
                )
                logger.debug(f"[{unique_id}] Optimal split points: {[f'{sp:.2f}s' for sp in split_points]}")
            else:
                logger.debug(f"[{unique_id}] No silence detected, using time-based chunking")
        
        # Create chunks based on split points (or use time-based if no silence found)
        if split_points:
            # Silence-based chunking
            chunk_boundaries = [0.0] + split_points + [total_duration]
            num_chunks = len(chunk_boundaries) - 1
        else:
            # Time-based chunking (fallback)
            num_chunks = math.ceil(total_duration / CHUNK_DURATION_SECONDS)
            chunk_boundaries = [min(i * CHUNK_DURATION_SECONDS, total_duration) for i in range(num_chunks + 1)]
        
        # Initialize progress tracking
        progress_tracker[unique_id] = {
            "status": "processing",
            "current_chunk": 0,
            "total_chunks": num_chunks,
            "progress_percent": 0,
            "partial_text": "" if config.enable_partial_text else None
        }
        
        logger.info(f"[{unique_id}] Total duration: {total_duration:.2f}s. Splitting into {num_chunks} chunks.")

        if num_chunks > 1:
            for i in range(num_chunks):
                start_time = chunk_boundaries[i]
                duration = chunk_boundaries[i + 1] - start_time
                chunk_path = os.path.join(
                    app.config["UPLOAD_FOLDER"], f"{unique_id}_chunk_{i}.wav"
                )
                chunk_paths.append(chunk_path)
                temp_files_to_clean.append(chunk_path)

                logger.debug(f"[{unique_id}] Creating chunk {i + 1}/{num_chunks} ({start_time:.2f}s - {chunk_boundaries[i+1]:.2f}s)...")
                chunk_command = [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-ss",
                    str(start_time),
                    "-t",
                    str(duration),
                    "-i",
                    target_wav_path,
                    "-ac",
                    str(config.channels),
                    "-ar",
                    str(config.sample_rate),
                    "-c:a",
                    "pcm_s16le",
                    chunk_path,
                ]
                result = subprocess.run(chunk_command, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning(f"[{unique_id}] Chunk extraction failed: {result.stderr}")
        else:
            chunk_paths.append(target_wav_path)

        all_segments = []
        all_words = []
        cumulative_time_offset = 0.0
        
        # Store chunk durations for offset calculation
        chunk_durations = []
        if num_chunks > 1:
            for i in range(num_chunks):
                duration = chunk_boundaries[i + 1] - chunk_boundaries[i]
                chunk_durations.append(duration)
        else:
            chunk_durations.append(total_duration)

        def clean_text(text):
            """Clean up spacing artifacts from token joining"""
            if not text:
                return ""
            # Handle potential SentencePiece underline
            text = text.replace("\u2581", " ")
            text = text.strip()
            # Collapse multiple spaces
            text = re.sub(r"\s+", " ", text)
            # Standard cleaning
            text = text.replace(" '", "'")
            # Ensure space before dollar signs
            text = re.sub(r"(\S)\$", r"\1 $", text)
            return text

        for i, chunk_path in enumerate(chunk_paths):
            chunk_start_time = time.time()
            progress_tracker[unique_id].update({
                "current_chunk": i + 1,
                "progress_percent": int((i + 1) / num_chunks * 100)
            })
            logger.info(f"[{unique_id}] Transcribing chunk {i + 1}/{num_chunks}...")

            result = asr_model.recognize(chunk_path)
            
            # Track chunk timing
            chunk_duration = time.time() - chunk_start_time
            chunk_times.append(chunk_duration)
            
            # Calculate and update estimated time remaining
            if chunk_times:
                avg_chunk_time = sum(chunk_times) / len(chunk_times)
                remaining_chunks = num_chunks - (i + 1)
                etr = avg_chunk_time * remaining_chunks
                progress_tracker[unique_id]["estimated_time_remaining"] = round(etr, 2)
            
            # Calculate RTF for this chunk
            chunk_audio_duration = chunk_durations[i]
            chunk_rtf = chunk_duration / chunk_audio_duration if chunk_audio_duration > 0 else 0
            logger.info(f"[{unique_id}] Chunk {i + 1}/{num_chunks} transcribed in {chunk_duration:.2f}s (RTF: {chunk_rtf:.3f})")

            if result and result.text:
                start_time = result.timestamps[0] if result.timestamps else 0
                end_time = (
                    result.timestamps[-1]
                    if len(result.timestamps) > 1
                    else start_time + 0.1
                )

                cleaned_text = clean_text(result.text)

                segment = {
                    "start": start_time + cumulative_time_offset,
                    "end": end_time + cumulative_time_offset,
                    "segment": cleaned_text,
                }
                all_segments.append(segment)
                
                # Update partial text for real-time streaming (if enabled)
                if config.enable_partial_text:
                    progress_tracker[unique_id]["partial_text"] += cleaned_text + " "

                for j, (token, timestamp) in enumerate(
                    zip(result.tokens, result.timestamps)
                ):
                    if j < len(result.timestamps) - 1:
                        word_end = result.timestamps[j + 1]
                    else:
                        word_end = end_time

                    # Clean tokens too
                    clean_token = token.replace("\u2581", " ").strip()
                    word = {
                        "start": timestamp + cumulative_time_offset,
                        "end": word_end + cumulative_time_offset,
                        "word": clean_token,
                    }
                    all_words.append(word)

            # Use planned chunk duration instead of ffprobe
            cumulative_time_offset += chunk_durations[i]

        logger.info(f"[{unique_id}] All chunks transcribed, merging results.")
        
        # Calculate final timing metrics
        total_processing_time = time.time() - request_start_time
        rtf = total_processing_time / total_duration if total_duration > 0 else 0
        
        # Update progress to complete
        progress_tracker[unique_id]["status"] = "complete"
        progress_tracker[unique_id]["progress_percent"] = 100
        progress_tracker[unique_id]["total_processing_time"] = round(total_processing_time, 2)
        progress_tracker[unique_id]["rtf"] = round(rtf, 3)
        
        logger.info(f"[{unique_id}] Transcription complete: {total_processing_time:.2f}s total, RTF: {rtf:.3f}")

        if not all_segments:
            # Return empty structure if nothing found, consistent with failures or silence?
            # OpenAI sometimes returns empty json text.
            pass

        # Formatting Output
        full_text = " ".join([seg["segment"] for seg in all_segments])
        
        # Auto-save transcription files
        txt_path = get_transcription_path(original_filename, ".txt")
        srt_path = get_transcription_path(original_filename, ".srt")
        
        try:
            # Save .txt file
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            logger.info(f"[{unique_id}] Saved transcription to: {txt_path}")
        except Exception as e:
            logger.error(f"[{unique_id}] Failed to save .txt file: {e}")
        
        try:
            # Save .srt file
            srt_output = segments_to_srt(all_segments)
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_output)
            logger.info(f"[{unique_id}] Saved SRT to: {srt_path}")
        except Exception as e:
            logger.error(f"[{unique_id}] Failed to save .srt file: {e}")

        if response_format == "srt" or model_name == "parakeet_srt_words":
            srt_output = segments_to_srt(all_segments)
            if model_name == "parakeet_srt_words":
                json_str_list = [
                    {"start": it["start"], "end": it["end"], "word": it["word"]}
                    for it in all_words
                ]
                srt_output += "----..----" + json.dumps(json_str_list)
            response = Response(srt_output, mimetype="text/plain")
            response.headers['X-Processing-Time'] = str(round(total_processing_time, 2))
            response.headers['X-Audio-Duration'] = str(round(total_duration, 2))
            response.headers['X-RTF'] = str(round(rtf, 3))
            return response

        elif response_format == "vtt":
            response = Response(segments_to_vtt(all_segments), mimetype="text/plain")
            response.headers['X-Processing-Time'] = str(round(total_processing_time, 2))
            response.headers['X-Audio-Duration'] = str(round(total_duration, 2))
            response.headers['X-RTF'] = str(round(rtf, 3))
            return response

        elif response_format == "text":
            response = Response(full_text, mimetype="text/plain")
            response.headers['X-Processing-Time'] = str(round(total_processing_time, 2))
            response.headers['X-Audio-Duration'] = str(round(total_duration, 2))
            response.headers['X-RTF'] = str(round(rtf, 3))
            return response

        elif response_format == "verbose_json":
            # Minimal verbose_json structure
            response = jsonify(
                {
                    "task": "transcribe",
                    "language": language or "auto-detected",
                    "duration": total_duration,
                    "processing_time": round(total_processing_time, 2),
                    "rtf": round(rtf, 3),
                    "text": full_text,
                    "segments": [
                        {
                            "id": idx,
                            "seek": 0,
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg["segment"],
                            "tokens": [],  # Populate if needed
                            "temperature": 0.0,
                            "avg_logprob": 0.0,
                            "compression_ratio": 0.0,
                            "no_speech_prob": 0.0,
                        }
                        for idx, seg in enumerate(all_segments)
                    ],
                }
            )
            response.headers['X-Processing-Time'] = str(round(total_processing_time, 2))
            response.headers['X-Audio-Duration'] = str(round(total_duration, 2))
            response.headers['X-RTF'] = str(round(rtf, 3))
            return response

        else:
            # Default JSON
            response = jsonify({"text": full_text})
            response.headers['X-Job-ID'] = unique_id
            response.headers['X-Processing-Time'] = str(round(total_processing_time, 2))
            response.headers['X-Audio-Duration'] = str(round(total_duration, 2))
            response.headers['X-RTF'] = str(round(rtf, 3))
            return response

    except Exception as e:
        logger.error(f"[{unique_id}] Error during processing: {e}", exc_info=True)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
    finally:
        logger.debug(f"[{unique_id}] Cleaning up temporary files...")
        
        # Add checkpoint to cleanup list
        checkpoint_path = os.path.join(config.transcription_folder, f"{unique_id}_partial.json")
        if os.path.exists(checkpoint_path):
             try:
                os.remove(checkpoint_path)
                logger.debug(f"[{unique_id}] Removed checkpoint file.")
             except OSError as e:
                logger.warning(f"[{unique_id}] Failed to remove checkpoint: {e}")

        for f_path in temp_files_to_clean:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except OSError:
                    pass
        logger.debug(f"[{unique_id}] Temporary files cleaned.")


@app.route("/stats")
def get_stats():
    """System resource usage statistics"""
    try:
        cpu = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        return jsonify({
            "cpu_percent": cpu,
            "memory_percent": memory.percent,
            "memory_used_gb": round(memory.used / (1024 ** 3), 1),
            "memory_total_gb": round(memory.total / (1024 ** 3), 1)
        })
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error": str(e)}), 500


def openweb():
    import webbrowser, time

    time.sleep(5)
    webbrowser.open_new_tab(f"http://127.0.0.1:{port}")


if __name__ == "__main__":
    logger.info(f"Starting server...")
    logger.info(f"Web interface: http://127.0.0.1:{port}")
    logger.info(f"API Endpoint: POST http://{host}:{port}/v1/audio/transcriptions")
    logger.info(f"Running with {threads} threads.")
    
    # Start background worker
    threading.Thread(target=worker_loop, daemon=True).start()
    
    logger.info(f"Starting web browser thread...")
    threading.Thread(target=openweb).start()
    logger.info(f"Starting waitress server...")
    serve(app, host=host, port=port, threads=threads)
    logger.info(f"Server started!")
