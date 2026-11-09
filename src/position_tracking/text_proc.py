import concurrent.futures
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch


# ==========================================
# 1. TRAJECTORY LOADER
# ==========================================
def load_trajectories_from_txt(txt_path):
    """Loads trajectories from a single .txt file into a list of NumPy arrays."""
    try:
        df = pd.read_csv(txt_path, sep=r"\s+")
        df.columns = [c.lower().strip() for c in df.columns]

        req_cols = ["ball_id", "frame", "x", "y"]
        if not all(col in df.columns for col in req_cols):
            print(
                f"Skipping {os.path.basename(txt_path)}: missing required columns {req_cols}"
            )
            return []

        trajectories = []
        for _, group in df.groupby("ball_id"):
            traj_data = group[["frame", "x", "y"]].to_numpy()
            trajectories.append(traj_data)

        return trajectories
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
        return []


# ==========================================
# 2. COLLECTIVE PLOT GENERATION
# ==========================================
def generate_plots(trajectories, output_base_path, folder_name=""):
    if not trajectories:
        return

    all_x = [pt for d in trajectories for pt in d[:, 1]]
    all_y = [pt for d in trajectories for pt in d[:, 2]]

    if not all_x or not all_y:
        return

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    x_pad = max((x_max - x_min) * 0.05, 10)
    y_pad = max((y_max - y_min) * 0.05, 10)

    # ------------------------------------------
    # Plot 1: Combined trajectory plot across all runs
    # ------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 10))
    for i, data in enumerate(trajectories):
        x = data[:, 1]
        y = data[:, 2]
        ax.plot(x, y, "o-", markersize=2, alpha=0.4, linewidth=0.8)

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_title(f"Collective Trajectories — {folder_name}", fontsize=14, pad=12)
    ax.set_xlabel("X Position [px]", fontsize=12)
    ax.set_ylabel("Y Position [px]", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.savefig(f"{output_base_path}_combined.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------
    # Plot 2: Collective layout density histograms
    # ------------------------------------------
    x_min_active = np.percentile(all_x, 10)
    x_max_active = np.percentile(all_x, 90)

    if x_max_active <= x_min_active:
        x_min_active, x_max_active = x_min, x_max

    x_slices = [
        x_min_active + (x_max_active - x_min_active) * 0.8,
        (x_min_active + x_max_active) / 2.0,
        x_min_active + (x_max_active - x_min_active) * 0.2,
    ]
    slice_colors = ["#00e676", "#ff007f", "#2979ff"]

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

    y_min_lim, y_max_lim = y_min - y_pad, y_max + y_pad

    for data in trajectories:
        x_arr, y_arr = data[:, 1], data[:, 2]
        ax_traj.plot(y_arr, x_arr, color="gray", alpha=0.25, linewidth=0.8)

    ax_traj.set_xlim(y_min_lim, y_max_lim)
    ax_traj.set_ylim(x_min - x_pad, x_max + x_pad)
    ax_traj.set_title(
        f"Trajectories — {folder_name}", fontsize=12, fontweight="bold", pad=12
    )
    ax_traj.set_xlabel("Y Position [px]", fontsize=10)
    ax_traj.set_ylabel("X Position [px]", fontsize=10)
    ax_traj.grid(True, linestyle=":", alpha=0.5)

    for i, (x_sl, col) in enumerate(zip(x_slices, slice_colors)):
        y_at_slice = []
        for data in trajectories:
            x_arr, y_arr = data[:, 1], data[:, 2]
            sort_idx = np.argsort(x_arr)

            y_interp = np.interp(
                x_sl, x_arr[sort_idx], y_arr[sort_idx], left=np.nan, right=np.nan
            )
            if not np.isnan(y_interp):
                y_at_slice.append(y_interp)

        y_data = np.array(y_at_slice)
        ax_traj.axhline(y=x_sl, color=col, linewidth=2.5, zorder=4)

        ax_h = ax_hists[i]
        if len(y_data) > 0:
            ax_h.hist(
                y_data,
                density=True,
                bins=15,
                color=col,
                alpha=0.65,
                edgecolor="black",
                linewidth=0.8,
            )

            np.random.seed(42)
            y_jitter = np.random.uniform(-0.0006, -0.0001, size=len(y_data))
            ax_h.scatter(y_data, y_jitter, color="#222222", s=15, alpha=0.6, zorder=4)

        ax_h.axhline(y=0, color=col, linewidth=2, zorder=5)
        ax_h.set_title(
            f"Histogram {i + 1} (X = {x_sl:.1f} px)",
            fontsize=11,
            fontweight="bold",
            pad=8,
            loc="left",
        )
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
# 3. FOLDER-LEVEL WORKER
# ==========================================
def process_folder(folder_path, root_data_folder, output_root):
    rel_folder = os.path.relpath(folder_path, root_data_folder)
    folder_name = os.path.basename(folder_path)

    folder_out_dir = os.path.join(output_root, rel_folder)
    os.makedirs(folder_out_dir, exist_ok=True)

    # Match trajectory text files properly
    txt_files = glob.glob(os.path.join(folder_path, "*_trajectories.txt"))
    if not txt_files:
        txt_files = [
            f
            for f in glob.glob(os.path.join(folder_path, "*.txt"))
            if not f.endswith("_collective.txt")
        ]

    if not txt_files:
        return

    folder_trajectories = []
    for txt_file in txt_files:
        trajs = load_trajectories_from_txt(txt_file)
        folder_trajectories.extend(trajs)

    if not folder_trajectories:
        return

    out_base = os.path.join(folder_out_dir, f"{folder_name}_collective")
    generate_plots(folder_trajectories, out_base, folder_name=folder_name)
    print(
        f"Processed '{folder_name}': Combined {len(folder_trajectories)} trajectories across {len(txt_files)} file(s)."
    )


# ==========================================
# 4. MAIN EXECUTOR
# ==========================================
if __name__ == "__main__":
    DATA_FOLDER = "processing_results"
    OUTPUT_FOLDER = "processing_results/connected"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    abs_output = os.path.abspath(OUTPUT_FOLDER)

    measurement_folders = []
    for root, _, files in os.walk(DATA_FOLDER):
        # Exclude the output destination folder from being scanned recursively
        if os.path.abspath(root).startswith(abs_output):
            continue
        if any(f.endswith("_trajectories.txt") or f.endswith(".txt") for f in files):
            measurement_folders.append(root)

    if not measurement_folders:
        print(f"No folders containing .txt trajectory files found in '{DATA_FOLDER}'")
    else:
        max_workers = min(len(measurement_folders), os.cpu_count() or 1)
        print(f"Found {len(measurement_folders)} measurement folder(s).")
        print(
            f"Processing folders in parallel using {max_workers} worker processes...\n"
        )

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [
                executor.submit(process_folder, folder, DATA_FOLDER, OUTPUT_FOLDER)
                for folder in measurement_folders
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        print("\nALL COLLECTIVE FOLDER PLOTS GENERATED SUCCESSFULLY!")
