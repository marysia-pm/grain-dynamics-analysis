import concurrent.futures
import glob
import os
import cv2
import matplotlib

# Set non-interactive backend for maximum plot rendering performance
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from tqdm import tqdm


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
    min_area=8,
    max_area=8000,  # Upper area limit to catch motion-blurred balls
    min_x_allowed=100,  # Mild margin cropping
    max_x_allowed=2350,
    min_y_allowed=50,
    max_y_allowed=2000,
    max_jump_distance=180,  # Max allowed 2D distance (px) between frames
    max_gap_frames=8,
    min_points=8,
    position=0,  # Dedicated terminal line for individual tqdm progress bar
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

    # Read Frame 0 as a sample image and rotate it 90 degrees
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

    # Dedicated progress bar for this video file
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

        _, binary = cv2.threshold(
            diff, threshold_value, 255, cv2.THRESH_BINARY
        )
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        valid_contours = [
            cnt
            for cnt in contours
            if min_area <= cv2.contourArea(cnt) <= max_area
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
            prev_x, prev_y = (
                current_trajectory[-1][1],
                current_trajectory[-1][2],
            )
            curr_x, curr_y = ball_pos[0], ball_pos[1]

            # 2D Euclidean Distance calculation
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

  return trajectories, sample_frame


# ==========================================
# 2. PLOT GENERATION
# ==========================================
def generate_plots(
    trajectories, output_base_path, image_width=2448, image_height=2048
):
  if not trajectories:
    return

  # Extract Y min and max across trajectories (d[:, 2] is Y)
  y_mins = [np.min(d[:, 2]) for d in trajectories]
  y_maxs = [np.max(d[:, 2]) for d in trajectories]
  common_y_min = np.min(y_mins)
  common_y_max = np.max(y_maxs)

  if common_y_max <= common_y_min:
    return

  # Horizontal Y-slice positions (ordered top-to-bottom to match subplots 1..3)
  y_slices = [
      common_y_max - (common_y_max - common_y_min) * 0.1,
      (common_y_max + common_y_min) / 2.0,
      common_y_min + (common_y_max - common_y_min) * 0.1,
  ]

  slice_colors = ["#2979ff", "#ff007f", "#00e676"]  # Top to bottom slice colors

  fig = plt.figure(figsize=(11, 8.5), dpi=120)
  gs = fig.add_gridspec(
      3,
      2,
      width_ratios=(1.8, 1),
      hspace=0.45,
      wspace=0.35,
      top=0.92,
      bottom=0.08,
      left=0.08,
      right=0.92,
  )

  ax_traj = fig.add_subplot(gs[:, 0])
  ax_hists = [fig.add_subplot(gs[i, 1]) for i in range(3)]

  all_x = []
  for data in trajectories:
    x_arr, y_arr = data[:, 1], data[:, 2]
    ax_traj.plot(x_arr, y_arr, color="gray", alpha=0.35, linewidth=1)
    all_x.extend(x_arr)

  # Configure main Trajectory plot (X vs Y)
  ax_traj.set_xlim(0, image_width)
  ax_traj.set_ylim(0, image_height)
  ax_traj.set_title("Trajectories", fontsize=12, fontweight="bold", pad=12)
  ax_traj.set_xlabel("X Position [px]", fontsize=10)
  ax_traj.set_ylabel("Y Position [px]", fontsize=10)
  ax_traj.grid(True, linestyle=":", alpha=0.5)

  x_min_lim, x_max_lim = np.min(all_x) - 30, np.max(all_x) + 30

  for i, (y_sl, col) in enumerate(zip(y_slices, slice_colors)):
    x_at_slice = []
    for data in trajectories:
      x_arr, y_arr = data[:, 1], data[:, 2]
      sort_idx = np.argsort(y_arr)
      # Interpolate X coordinate at fixed Y slice
      x_interp = np.interp(y_sl, y_arr[sort_idx], x_arr[sort_idx])
      x_at_slice.append(x_interp)

    x_data = np.array(x_at_slice)

    # 1. Draw horizontal slice line on trajectory plot
    ax_traj.axhline(y=y_sl, color=col, linewidth=2.5, zorder=4)

    # 2. Draw histogram for X positions
    ax_h = ax_hists[i]
    ax_h.hist(
        x_data,
        density=True,
        bins=10,
        color=col,
        alpha=0.65,
        edgecolor="black",
        linewidth=0.8,
    )

    ax_h.set_title(
        f"Histogram {i + 1} (Y = {y_sl:.1f} px)",
        fontsize=11,
        fontweight="bold",
        pad=8,
        loc="left",
    )
    ax_h.yaxis.tick_right()
    ax_h.yaxis.set_label_position("right")
    ax_h.set_ylabel("Density", fontsize=9, labelpad=8)
    ax_h.set_xlim(x_min_lim, x_max_lim)
    ax_h.grid(True, linestyle=":", alpha=0.4)

    # Label X-axis on bottom histogram
    if i == 2:
      ax_h.set_xlabel("X Position [px]", fontsize=10)

    # 3. Connection lines linking horizontal slice line to histogram
    con = ConnectionPatch(
        xyA=(ax_traj.get_xlim()[1], y_sl),
        coordsA=ax_traj.transData,
        xyB=(0, 0.5),
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
def process_single_file(
    raw_file_path, position=0, output_dir="processing_results"
):
  filename = os.path.basename(raw_file_path)
  base_name = os.path.splitext(filename)[0]
  out_base = os.path.join(output_dir, base_name)

  trajectories, sample_frame = extract_trajectories_from_raw(
      raw_file_path, position=position
  )

  # Save a rotated sample frame from the video
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
  RAW_FOLDER = "/home/marian/Desktop/Marysia/videos/"
  OUTPUT_FOLDER = "processing_results"

  os.makedirs(OUTPUT_FOLDER, exist_ok=True)
  raw_files = sorted(glob.glob(os.path.join(RAW_FOLDER, "*.raw")))

  if not raw_files:
    print(f"❌ No .raw files found in directory '{RAW_FOLDER}'")
  else:
    max_workers = min(len(raw_files), os.cpu_count())
    print(f"🚀 Found {len(raw_files)} raw video files in {RAW_FOLDER}")
    print(f"⚡ Processing using {max_workers} worker processes...\n")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:
      futures = [
          executor.submit(process_single_file, raw_file, idx, OUTPUT_FOLDER)
          for idx, raw_file in enumerate(raw_files)
      ]
      for future in concurrent.futures.as_completed(futures):
        future.result()

    print(
        "\n" * (len(raw_files) + 1) + "🎉 ALL FILES PROCESSED SUCCESSFULLY!"
    )