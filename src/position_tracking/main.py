import concurrent.futures
import glob
import os
import matplotlib

# Set non-interactive backend for maximum plot rendering performance
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import cv2
from matplotlib.patches import ConnectionPatch
from tqdm import tqdm


# ==========================================
# 0. TRAJECTORY POST-FILTERING
# ==========================================
def filter_trajectories(
    trajectories,
    min_points=15,             # Must persist for at least 15 frames
    min_x_travel=150,          # Must travel at least 150px vertically along stream
    valid_y_range=(500, 1600), # Y range where active grain stream flows
):
    """Purges spurious noise, static specks, and border reflection artifacts."""
    cleaned = []
    for traj in trajectories:
        if len(traj) < min_points:
            continue

        x_pts = traj[:, 1]
        y_pts = traj[:, 2]

        # 1. Total X displacement (particles must move along the stream)
        x_span = np.ptp(x_pts)  # max(x) - min(x)
        if x_span < min_x_travel:
            continue

        # 2. Spatial channel constraint (median Y must lie within the main stream)
        median_y = np.median(y_pts)
        if not (valid_y_range[0] <= median_y <= valid_y_range[1]):
            continue

        cleaned.append(traj)

    return cleaned


# ==========================================
# 1. RAW VIDEO PROCESSING -> TRAJECTORIES
# ==========================================
def extract_trajectories_from_raw(
    raw_file_path,
    width=2448,
    height=2048,
    channels=1,
    num_bg_frames=10,
    threshold_value=12,
    min_area=40,             # Increased from 8 to ignore tiny pixel specks
    max_area=8000,
    min_x_allowed=100,
    max_x_allowed=2350,
    min_y_allowed=500,       # Tightened from 50 to cut left margin noise
    max_y_allowed=1600,      # Tightened from 2000 to cut right margin noise
    max_jump_distance=180,
    max_gap_frames=8,
    min_points=15,           # Increased from 8 to reject short spurious tracks
    position=0,
):
    filename = os.path.basename(raw_file_path)
    frame_byte_size = width * height * channels
    shape = (height, width) if channels == 1 else (height, width, channels)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    file_size = os.path.getsize(raw_file_path)
    total_frames = file_size // frame_byte_size

    trajectories = []

    with open(raw_file_path, "rb") as f:
        bg_frames = []
        for _ in range(num_bg_frames):
            raw_bytes = f.read(frame_byte_size)
            if len(raw_bytes) < frame_byte_size:
                break
            bg_frames.append(
                np.frombuffer(raw_bytes, dtype=np.uint8)
                .reshape(shape)
                .astype(np.float32)
            )

        if not bg_frames:
            return [], None

        mean_background = np.mean(bg_frames, axis=0)
        f.seek(0)

        sample_raw = f.read(frame_byte_size)
        if len(sample_raw) == frame_byte_size:
            sample_frame = np.frombuffer(sample_raw, dtype=np.uint8).reshape(shape)
            sample_frame = cv2.rotate(sample_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            sample_frame = None

        f.seek(0)

        current_trajectory = []
        is_tracking = False
        frames_lost = 0
        frame_idx = 0

        with tqdm(
            total=total_frames,
            desc=f"📹 {filename[:22]}...",
            position=position,
            leave=True,
            unit="frame",
            dynamic_ncols=True,
        ) as pbar:

            while True:
                raw_bytes = f.read(frame_byte_size)
                if len(raw_bytes) < frame_byte_size:
                    break

                frame = (
                    np.frombuffer(raw_bytes, dtype=np.uint8)
                    .reshape(shape)
                    .astype(np.float32)
                )
                diff = np.abs(frame - mean_background).astype(np.uint8)

                # Apply ROI Masking
                diff[:, :min_x_allowed] = 0
                diff[:, max_x_allowed:] = 0
                diff[:min_y_allowed, :] = 0
                diff[max_y_allowed:, :] = 0

                _, binary = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
                binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

                contours, _ = cv2.findContours(
                    binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                valid_contours = [
                    cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area
                ]

                ball_pos = None
                if valid_contours:
                    largest = max(valid_contours, key=cv2.contourArea)
                    M = cv2.moments(largest)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        ball_pos = (cx, cy)

                if not is_tracking:
                    if ball_pos is not None:
                        is_tracking = True
                        current_trajectory = [(frame_idx, ball_pos[0], ball_pos[1])]
                        frames_lost = 0
                else:
                    if ball_pos is not None:
                        prev_x, prev_y = current_trajectory[-1][1], current_trajectory[-1][2]
                        curr_x, curr_y = ball_pos[0], ball_pos[1]

                        dist = np.hypot(curr_x - prev_x, curr_y - prev_y)

                        if dist > max_jump_distance:
                            if len(current_trajectory) >= min_points:
                                trajectories.append(np.array(current_trajectory))
                            current_trajectory = [(frame_idx, ball_pos[0], ball_pos[1])]
                            frames_lost = 0
                        else:
                            current_trajectory.append((frame_idx, ball_pos[0], ball_pos[1]))
                            frames_lost = 0
                    else:
                        frames_lost += 1

                    if frames_lost > max_gap_frames:
                        if len(current_trajectory) >= min_points:
                            trajectories.append(np.array(current_trajectory))
                        is_tracking = False
                        current_trajectory = []
                        frames_lost = 0

                frame_idx += 1
                pbar.update(1)

            if is_tracking and len(current_trajectory) >= min_points:
                trajectories.append(np.array(current_trajectory))

    # Apply strict post-filtering pass
    trajectories = filter_trajectories(trajectories)

    return trajectories, sample_frame


# ==========================================
# 2. PLOT GENERATION
# ==========================================
def generate_plots(trajectories, output_base_path, image_width=2448, image_height=2048):
    if not trajectories:
        return

    # 1. Combined trajectory plot
    fig, ax = plt.subplots(figsize=(8, 10))
    for i, data in enumerate(trajectories):
        x = data[:, 1]
        y = data[:, 2]
        ax.plot(x, y, "o-", markersize=3, label=f"Ball {i+1}", alpha=0.7)

    ax.set_xlim(0, image_width)
    ax.set_ylim(0, image_height)
    ax.set_title("Combined Trajectories", fontsize=14, pad=12)
    ax.set_xlabel("X Position [px]", fontsize=12)
    ax.set_ylabel("Y Position [px]", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.savefig(f"{output_base_path}_combined.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2. Layout density histograms
    all_x = [pt for d in trajectories for pt in d[:, 1]]
    all_y = [pt for d in trajectories for pt in d[:, 2]]

    if not all_x:
        return

    x_min_active = np.percentile(all_x, 10)
    x_max_active = np.percentile(all_x, 90)

    if x_max_active <= x_min_active:
        x_min_active, x_max_active = np.min(all_x), np.max(all_x)

    x_slices = [
        x_min_active + (x_max_active - x_min_active) * 0.8,
        (x_min_active + x_max_active) / 2.0,
        x_min_active + (x_max_active - x_min_active) * 0.2,
    ]
    slice_colors = ["#00e676", "#ff007f", "#2979ff"]

    fig = plt.figure(figsize=(11, 8.5), dpi=120)
    gs = fig.add_gridspec(
        3, 2, width_ratios=(1.8, 1), hspace=0.45, wspace=0.35,
        top=0.92, bottom=0.08, left=0.08, right=0.92
    )

    ax_traj = fig.add_subplot(gs[:, 0])
    ax_hists = [fig.add_subplot(gs[i, 1]) for i in range(3)]

    y_min_lim, y_max_lim = np.min(all_y) - 30, np.max(all_y) + 30

    for data in trajectories:
        x_arr, y_arr = data[:, 1], data[:, 2]
        ax_traj.plot(y_arr, x_arr, color="gray", alpha=0.35, linewidth=1)

    ax_traj.set_xlim(y_min_lim, y_max_lim)
    ax_traj.set_ylim(np.min(all_x) - 50, np.max(all_x) + 50)
    ax_traj.set_title("Trajectories", fontsize=12, fontweight="bold", pad=12)
    ax_traj.set_xlabel("Y Position [px]", fontsize=10)
    ax_traj.set_ylabel("X Position [px]", fontsize=10)
    ax_traj.grid(True, linestyle=":", alpha=0.5)

    for i, (x_sl, col) in enumerate(zip(x_slices, slice_colors)):
        y_at_slice = []
        for data in trajectories:
            x_arr, y_arr = data[:, 1], data[:, 2]
            sort_idx = np.argsort(x_arr)

            y_interp = np.interp(x_sl, x_arr[sort_idx], y_arr[sort_idx], left=np.nan, right=np.nan)
            if not np.isnan(y_interp):
                y_at_slice.append(y_interp)

        y_data = np.array(y_at_slice)

        ax_traj.axhline(y=x_sl, color=col, linewidth=2.5, zorder=4)

        ax_h = ax_hists[i]

        if len(y_data) > 0:
            ax_h.hist(y_data, density=True, bins=10, color=col, alpha=0.65, edgecolor="black", linewidth=0.8)

            np.random.seed(42)
            y_jitter = np.random.uniform(-0.0006, -0.0001, size=len(y_data))
            ax_h.scatter(y_data, y_jitter, color="#222222", s=20, alpha=0.75, zorder=4)

        ax_h.axhline(y=0, color=col, linewidth=2, zorder=5)
        ax_h.set_title(f"Histogram {i + 1} (X = {x_sl:.1f} px)", fontsize=11, fontweight="bold", pad=8, loc="left")
        ax_h.yaxis.tick_right()
        ax_h.yaxis.set_label_position("right")
        ax_h.set_ylabel("Density", fontsize=9, labelpad=8)
        ax_h.set_xlim(y_min_lim, y_max_lim)
        ax_h.grid(True, linestyle=":", alpha=0.4)

        if i == 2:
            ax_h.set_xlabel("Y Position [px]", fontsize=10)

        con = ConnectionPatch(
            xyA=(ax_traj.get_xlim()[1], x_sl),
            coordsA=ax_traj.transData,
            xyB=(0, 0),
            coordsB=ax_h.transAxes,
            color=col,
            linewidth=1.8,
            linestyle="--",
        )
        fig.add_artist(con)

    fig.savefig(f"{output_base_path}_layout_histograms.png", bbox_inches="tight")
    plt.close(fig)


# ==========================================
# 3. WORKER FUNCTION
# ==========================================
def process_single_file(raw_file_path, position=0, output_dir="processing_results"):
    filename = os.path.basename(raw_file_path)
    base_name = os.path.splitext(filename)[0]
    out_base = os.path.join(output_dir, base_name)

    trajectories, sample_frame = extract_trajectories_from_raw(raw_file_path, position=position)

    if sample_frame is not None:
        cv2.imwrite(f"{out_base}_sample_frame.jpg", sample_frame)

    if not trajectories:
        return

    generate_plots(trajectories, out_base)

    txt_path = f"{out_base}_trajectories.txt"
    with open(txt_path, "w") as f:
        f.write("ball_id\tframe\tx\ty\n")
        for ball_id, traj in enumerate(trajectories):
            for frame, x, y in traj:
                f.write(f"{ball_id}\t{int(frame)}\t{int(x)}\t{int(y)}\n")


# ==========================================
# 4. PARALLEL EXECUTOR
# ==========================================
if __name__ == "__main__":
    RAW_FOLDER = "/home/marian/mounted1/Marysia/position_tracking/"
    OUTPUT_FOLDER = "processing_results"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    raw_files = sorted(glob.glob(os.path.join(RAW_FOLDER, "*.raw")))

    if not raw_files:
        print(f"❌ No .raw files found in directory '{RAW_FOLDER}'")
    else:
        max_workers = min(len(raw_files), os.cpu_count())
        print(f"🚀 Found {len(raw_files)} raw video files in {RAW_FOLDER}")
        print(f"⚡ Processing using {max_workers} worker processes...\n")

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_file, raw_file, idx, OUTPUT_FOLDER)
                for idx, raw_file in enumerate(raw_files)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        print("\n" * (len(raw_files) + 1) + "🎉 ALL FILES PROCESSED SUCCESSFULLY!")