# hawki — Edge AI Situational Awareness System

Real-time multi-object detection, tracking, and LLM-powered situational awareness running entirely on an NVIDIA Spark (Jetson Orin Nano). No cloud required — every component runs locally on-device.

> **Status:** Phase 1 in progress — YOLOv8 baseline on live webcam feed

---

## What it does

hawki ingests a live camera feed and runs a full AI pipeline on the edge:

1. **Detect** — YOLOv8 identifies objects in each frame in real time
2. **Track** — ByteTrack assigns persistent IDs so the system recognizes the same object across frames *(Phase 2)*
3. **Reason** — A rule engine flags behavioral anomalies: loitering, zone crossing, crowd density *(Phase 3)*
4. **Report** — A local Llama 3.1 8B LLM generates natural-language situation reports from the event stream *(Phase 4)*
5. **Display** — A FastAPI web dashboard shows the live feed, object counts, event log, and LLM summaries *(Phase 5)*

## Hardware

| Component | Spec |
|-----------|------|
| Device | NVIDIA Spark (Jetson Orin Nano) |
| GPU | NVIDIA Ampere — 1024 CUDA cores, 32 Tensor Cores, ~67 TOPS |
| RAM | 8 GB (shared CPU/GPU) |
| Storage | 1 TB NVMe SSD |
| OS | Ubuntu-based JetPack (L4T) |

## Pipeline architecture

```
Camera (OpenCV)
    │
    ▼
Object Detection        ← YOLOv8n, TensorRT-optimized
    │
    ▼
Multi-Object Tracking   ← ByteTrack (persistent object IDs)
    │
    ▼
Anomaly / Event Engine  ← loitering, zone crossing, crowd density
    │
    ▼
Event Logger            ← SQLite: timestamp, object ID, event type, frame snapshot
    │
    ▼
LLM Reasoning Layer     ← Llama 3.1 8B via Ollama (async, non-blocking)
    │
    ▼
Web Dashboard           ← FastAPI + WebSockets, live feed + event log
```

## Performance targets

| Metric | Target |
|--------|--------|
| Detection latency | < 30 ms/frame |
| Tracking overhead | < 5 ms/frame |
| End-to-end FPS | ≥ 20 FPS |
| LLM report latency | < 10 s (async) |
| SQLite write latency | < 2 ms/event |

## Getting started

### Prerequisites

- NVIDIA Spark with JetPack SDK installed (PyTorch and OpenCV come pre-installed)
- Python 3.10+

### Install dependencies

```bash
pip install --break-system-packages -r requirements.txt
```

> `--break-system-packages` is required on JetPack's managed Python environment.

### Run the detection pipeline

```bash
python src/capture.py
```

YOLOv8n weights (~6 MB) are downloaded automatically on first run. Press `q` to quit.

### Test the detector on a single image

```bash
python src/detect.py path/to/image.jpg
```

### Export model to TensorRT *(Phase 2)*

```bash
python scripts/export_trt.py --model yolov8n.pt --imgsz 640
```

### Run benchmark

```bash
python scripts/benchmark.py
```

### Start the dashboard *(Phase 5)*

```bash
uvicorn src.dashboard.server:app --host 0.0.0.0 --port 8000
```

## Project structure

```
hawki/
├── config.yaml             ← all tunable parameters (thresholds, paths, etc.)
├── requirements.txt
├── src/
│   ├── capture.py          ← main pipeline entry point (webcam loop)
│   ├── detect.py           ← YOLOv8 inference wrapper
│   ├── track.py            ← ByteTrack integration (Phase 2)
│   ├── rules.py            ← behavioral anomaly engine (Phase 3)
│   ├── logger.py           ← SQLite event logging (Phase 3)
│   ├── llm.py              ← Ollama wrapper + prompt builder (Phase 4)
│   └── dashboard/          ← FastAPI backend + JS frontend (Phase 5)
├── models/                 ← model weights (not committed to git)
├── data/visDrone/          ← training data (not committed to git)
├── notebooks/
│   └── finetune.ipynb      ← VisDrone fine-tuning workflow
├── scripts/
│   ├── export_trt.py       ← PyTorch → TensorRT export
│   └── benchmark.py        ← FPS, latency, memory profiling
└── logs/                   ← SQLite DB and frame snapshots (not committed)
```

## Key design decisions

- **Fully offline** — designed for air-gapped and bandwidth-constrained environments
- **Async LLM** — situation reports are generated in a background thread so they never block the video pipeline
- **Config-driven** — all thresholds and paths live in `config.yaml`, nothing is hardcoded
- **Modular pipeline** — each stage (detect, track, rules, llm) is independently runnable and testable

## Build phases

| Phase | Weeks | Goal |
|-------|-------|------|
| 1 | 1–2 | YOLOv8 baseline on live webcam at real-time FPS ✅ |
| 2 | 3–4 | ByteTrack integration + VisDrone fine-tuning |
| 3 | 5–6 | Behavioral rule engine + SQLite event logging |
| 4 | 7–8 | Local LLM (Ollama) + situation report pipeline |
| 5 | 9–10 | FastAPI dashboard + demo recording + benchmarks |
