# Grain Dynamics Analysis

A lightweight Python pipeline for tracking particle and grain trajectories from raw video files (`.raw`), featuring parallel processing, automated density histogram plotting, and modern project management via **`uv`**.

---

## 📁 Project Structure

```text
grain-dynamics-analysis/
├── processing_results/       # Output plots and export files
├── src/                      # Source code (main.py)
├── pyproject.toml            # Dependencies and project setup
├── uv.lock                   # Lockfile (uv)
└── README.md

```

---

## ⚡ Quick Start

### 1. Using `uv` (Recommended)

Install dependencies and run the pipeline:

```bash
uv sync
uv run python src/main.py

```

## ✨ Features

* **RAW Video Processing:** Directly reads raw binary frames ($2448 \times 2048$, Mono8).
* **Multi-Processing:** Parallel video analysis with dedicated `tqdm` progress bars per file.
* **Particle Tracking:** Background subtraction with gap-tolerant tracking.
* **Automated Output Reports:**
* `_combined.png` — Full trajectory overview.
* `_layout_histograms.png` — Trajectory slice lines paired with $Y$-position density histograms.
* `_sample_frame.jpg` — Rotated sample preview frame.
* `_trajectories.txt` — Exported coordinates (`ball_id`, `frame`, `x`, `y`).



---

## ⚙️ Main Parameters (`src/main.py`)

Adjust tracking parameters directly inside `extract_trajectories_from_raw()`:

| Parameter | Default | Description |
| --- | --- | --- |
| `width` / `height` | `2448` / `2048` | Frame size in pixels |
| `threshold_value` | `12` | Background subtraction sensitivity |
| `min_area` / `max_area` | `8` / `8000` | Particle size bounds ($\text{px}^2$) |
| `max_jump_distance` | `180` | Max allowed displacement per frame ($\text{px}$) |
| `max_gap_frames` | `8` | Frame loss tolerance before ending a trajectory |
