import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch


def plot_layout_trajectories_with_histograms_fixed(
    trajectories_folder="trajektorie_wynikowe",
    output_png="05_layout_trajektorie_histogramy_fixed.png",
):
    """Generuje czytelny, poprawnie zorientowany wykres z trajektoriami po lewej

    oraz 3 chronologicznymi histogramami ewolucji rozkładu po prawej.
    """
    if not os.path.exists(trajectories_folder):
        print(f"Błąd: Folder '{trajectories_folder}' nie istnieje.")
        return

    files = sorted(glob.glob(os.path.join(trajectories_folder, "*.txt")))
    if not files:
        print("Brak plików z trajektoriami.")
        return

    trajectories = []
    x_mins, x_maxs = [], []

    for fpath in files:
        data = np.loadtxt(fpath, comments="#")
        if data.size == 0:
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)

        x = data[:, 1]
        y = data[:, 2]

        trajectories.append((x, y))
        x_mins.append(np.min(x))
        x_maxs.append(np.max(x))

    if not trajectories:
        print("Brak poprawnych danych.")
        return

    common_x_min = np.max(x_mins)
    common_x_max = np.min(x_maxs)

    # 1. CHRONOLOGICZNY PORZĄDEK PRZEKROJÓW (Start -> Środek -> Koniec lotu)
    # Start lotu jest przy wyższych wartościach X (~2000 px), koniec przy niższych (~700 px)
    x_slices = [
        common_x_max - (common_x_max - common_x_min) * 0.1,  # Etap 1: Początek
        (common_x_max + common_x_min) / 2.0,  # Etap 2: Środek
        common_x_min + (common_x_max - common_x_min) * 0.1,  # Etap 3: Koniec
    ]

    # Kolory przekrojów: Zielony (Start), Różowy (Środek), Niebieski (Koniec)
    slice_colors = ["#00e676", "#ff007f", "#2979ff"]

    # 2. ROZMIAR FIGURY I ODSTĘPY (Rozwiązuje problem ciasnoty)
    fig = plt.figure(figsize=(11, 8.5), dpi=150)
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=(1, 1.8),
        hspace=0.45,  # Odstęp pionowy między histogramami
        wspace=0.35,  # Odstęp poziomy między panelami
        top=0.92,
        bottom=0.08,
        left=0.08,
        right=0.92,
    )

    ax_traj = fig.add_subplot(gs[:, 0])
    ax_hists = [fig.add_subplot(gs[i, 1]) for i in range(3)]

    # 3. RYSOWANIE TRAJEKTORII (Lewy panel)
    all_y = []
    for x_arr, y_arr in trajectories:
        ax_traj.plot(y_arr, x_arr, color="gray", alpha=0.35, linewidth=1)
        all_y.extend(y_arr)

    # POPRAWKA ORIENTACJI: START (X ~ 2200) na GÓRZE, KONIEC (X ~ 600) na DOLE
    ax_traj.set_ylim(common_x_min - 120, common_x_max + 120)

    ax_traj.set_title("Trajektorie", fontsize=12, fontweight="bold", pad=12)
    ax_traj.set_xlabel("Pozycja Y [px]", fontsize=10)
    ax_traj.set_ylabel("Pozycja X [px]", fontsize=10)
    ax_traj.grid(True, linestyle=":", alpha=0.5)

    y_min_lim, y_max_lim = np.min(all_y) - 30, np.max(all_y) + 30

    # 4. RYSOWANIE HISTOGRAMÓW (Prawy panel)
    for i, (x_sl, col) in enumerate(zip(x_slices, slice_colors)):
        y_at_slice = []
        for x_arr, y_arr in trajectories:
            sort_idx = np.argsort(x_arr)
            y_interp = np.interp(x_sl, x_arr[sort_idx], y_arr[sort_idx])
            y_at_slice.append(y_interp)

        y_data = np.array(y_at_slice)

        # Linia pozioma cięcia na trajektorii
        ax_traj.axhline(y=x_sl, color=col, linewidth=2.5, zorder=4)

        # Histogram
        ax_h = ax_hists[i]
        ax_h.hist(
            y_data,
            density=True,
            bins=10,
            color=col,
            alpha=0.65,
            edgecolor="black",
            linewidth=0.8,
        )

        # Punkty pomiarowe tuż przy osi zerowej
        np.random.seed(42)
        y_jitter = np.random.uniform(-0.0006, -0.0001, size=len(y_data))
        ax_h.scatter(y_data, y_jitter, color="#222222", s=20, alpha=0.75, zorder=4)

        # Linia bazowa histogramu
        ax_h.axhline(y=0, color=col, linewidth=2, zorder=5)

        ax_h.set_title(
            f"Histogram {i + 1} (X = {x_sl:.1f} px)",
            fontsize=11,
            fontweight="bold",
            pad=8,
            loc="left",
        )

        # PRZENIESIENIE OSI Y NA PRAWĄ STRONĘ (Brak kolizji z tekstami)
        ax_h.yaxis.tick_right()
        ax_h.yaxis.set_label_position("right")
        ax_h.set_ylabel("Gęstość", fontsize=9, labelpad=8)

        ax_h.set_xlim(y_min_lim, y_max_lim)
        ax_h.grid(True, linestyle=":", alpha=0.4)

        if i == 2:
            ax_h.set_xlabel("Pozycja Y [px]", fontsize=10)

        # LINIA ŁĄCZĄCA PANELA (ConnectionPatch)
        con = ConnectionPatch(
            xyA=(ax_traj.get_xlim()[1], x_sl),
            coordsA=ax_traj.transData,
            xyB=(ax_h.get_xlim()[0], 0),
            coordsB=ax_h.transData,
            color=col,
            linewidth=2,
            linestyle="-",
        )
        fig.add_artist(con)

    fig.savefig(output_png, bbox_inches="tight")
    print(f"Zapisano poprawiony wykres: {output_png}")
    plt.show()


def plot_single_trajectory(
    file_path,
    output_png=None,
    image_width=2448,
    image_height=2048,
    fps=1000,
):
    """Rysuje trajektorię z jednego pliku .txt i zapisuje wykres do PNG."""
    if not os.path.exists(file_path):
        print(f"Błąd: Plik '{file_path}' nie istnieje.")
        return

    data = np.loadtxt(file_path, comments="#")
    if data.size == 0:
        print(f"Plik '{file_path}' jest pusty.")
        return

    if data.ndim == 1:
        data = data.reshape(1, -1)

    frames = data[:, 0]
    x = data[:, 2]
    y = data[:, 1]

    time_axis = (frames - frames[0]) / fps if fps else frames
    cbar_label = "Czas [s]" if fps else "Numer klatki"

    fig, ax = plt.subplots(figsize=(8, 10))

    scatter = ax.scatter(
        x,
        y,
        c=time_axis,
        cmap="jet",
        edgecolors="k",
        s=40,
        zorder=3,
        label="Pozycja kulki",
    )
    ax.plot(x, y, "--", color="gray", alpha=0.5, zorder=2)

    ax.set_xlim(0, image_height)
    ax.set_ylim(0, image_width)

    filename = os.path.basename(file_path)
    ax.set_title(f"Trajektoria: {filename}", fontsize=14, pad=12)
    ax.set_xlabel("Pozycja X [px]", fontsize=12)
    ax.set_ylabel("Pozycja Y [px]", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(cbar_label, fontsize=11)

    plt.tight_layout()

    save_path = output_png or file_path.replace(".txt", ".png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Zapisano wykres: {save_path}")


def plot_all_trajectories_individually(
    trajectories_folder="trajektorie_wynikowe",
    output_folder="wykresy_pojedyncze",
    image_width=2448,
    image_height=2048,
    fps=1000,
):
    """Przechodzi w pętli przez wszystkie pliki .txt i generuje dla każdego osobny obraz PNG."""
    os.makedirs(output_folder, exist_ok=True)
    files = sorted(glob.glob(os.path.join(trajectories_folder, "*.txt")))

    if not files:
        print(f"Brak plików .txt w folderze '{trajectories_folder}'.")
        return

    print(f"Znaleziono {len(files)} trajektorii. Generowanie pojedynczych wykresów...")
    for file_path in files:
        filename_png = os.path.basename(file_path).replace(".txt", ".png")
        out_png_path = os.path.join(output_folder, filename_png)

        plot_single_trajectory(
            file_path=file_path,
            output_png=out_png_path,
            image_width=image_width,
            image_height=image_height,
            fps=fps,
        )


def filter_trajectories(
    trajectories_folder="trajektorie_wynikowe",
    min_points=10,  # Odrzuca zbyt krótkie trajektorie (np. 2-5 punktów)
):
    files = glob.glob(os.path.join(trajectories_folder, "*.txt"))
    valid_count = 0
    removed_count = 0

    print(f"Analiza {len(files)} plików trajektorii...")

    for file_path in files:
        data = np.loadtxt(file_path, comments="#")
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # 1. Warunek długości
        if len(data) < min_points:
            os.remove(file_path)
            removed_count += 1
            continue

        valid_count += 1

    print(
        f"Koniec filtracji. Zostawiono: {valid_count}, Usunięto błędnych: {removed_count}"
    )


def plot_all_trajectories_combined(
    trajectories_folder="trajektorie_wynikowe",
    output_png="wszystkie_trajektorie.png",
    image_width=2448,
    image_height=2048,
):
    """Rysuje wszystkie wykryte trajektorie z folderu na jednym wspólnym wykresie."""
    files = sorted(glob.glob(os.path.join(trajectories_folder, "*.txt")))
    if not files:
        print(f"Brak plików .txt w folderze '{trajectories_folder}'.")
        return

    fig, ax = plt.subplots(figsize=(8, 10))

    for file_path in files:
        data = np.loadtxt(file_path, comments="#")
        if data.size == 0:
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)

        x = data[:, 2]
        y = data[:, 1]
        label = os.path.basename(file_path).replace(".txt", "")

        ax.plot(x, y, "o-", markersize=3, label=label, alpha=0.7)

    ax.set_xlim(0, image_height)
    ax.set_ylim(0, image_width)
    ax.set_title("Zbiorczy wykres wszystkich trajektorii", fontsize=14, pad=12)
    ax.set_xlabel("Pozycja X [px]", fontsize=12)
    ax.set_ylabel("Pozycja Y [px]", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"Zapisano wykres zbiorczy: {output_png}")


# --- UŻYCIE ---

# 0. Filtrowanie plików .txr

filter_trajectories()

# 1. Wyplotowanie wszystkich plików .txt w pętli do osobnych grafik PNG:
plot_all_trajectories_individually(
    trajectories_folder="trajektorie_wynikowe",
    output_folder="wykresy_pojedyncze",
    fps=1000,
)

# 2. Wyplotowanie wszystkich kulek na jednym zbiorczym obrazie:
plot_all_trajectories_combined(
    trajectories_folder="trajektorie_wynikowe",
    output_png="wszystkie_trajektorie.png",
)


# 3. Wyplotowanie rozkładu gęstosci
plot_layout_trajectories_with_histograms_fixed(
    trajectories_folder="trajektorie_wynikowe",
    output_png="05_layout_trajektorie_histogramy_fixed.png",
)
