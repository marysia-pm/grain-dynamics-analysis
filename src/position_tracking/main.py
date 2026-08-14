import os
import cv2
import numpy as np


def track_and_split_trajectories(
    raw_file_path,
    width=2448,
    height=2048,
    channels=1,
    num_bg_frames=10,
    threshold_value=12,
    min_area=8,
    max_x_allowed=2300,  # ODCINANIE PRAWEJ KRAWĘDZI (ignoruje mechanizm/kartkę dla X > 2300)
    jump_x_threshold=80,  # SKOK W PRAWO: Jeśli X_aktualne > X_poprzednie + 80px -> NOWA KULKA
    max_gap_frames=8,
    output_trajectories_folder="trajektorie_wynikowe",
):
    os.makedirs(output_trajectories_folder, exist_ok=True)
    frame_byte_size = width * height * channels
    shape = (height, width) if channels == 1 else (height, width, channels)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def save_trajectory_file(traj, count):
        """Pomocnicza funkcja do zapisu pliku TXT."""
        if len(traj) < 4:  # Ignoruj bardzo krótkie szumy (mniej niż 4 punkty)
            return count

        count += 1
        out_path = os.path.join(
            output_trajectories_folder, f"trajektoria_kulki_{count:03d}.txt"
        )
        with open(out_path, "w") as f_out:
            f_out.write("# NrKlatki X Y\n")
            for f_num, x, y in traj:
                f_out.write(f"{f_num} {x} {y}\n")

        print(
            f"[Kulka #{count:03d}] ZAPISANO: {out_path} ({len(traj)} punktów, X_start={traj[0][1]}, X_stop={traj[-1][1]})"
        )
        return count

    with open(raw_file_path, "rb") as f:
        # 1. Wyznaczenie średniego tła
        print("Wyznaczanie tła...")
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

        mean_background = np.mean(bg_frames, axis=0)
        f.seek(0)

        current_trajectory = []
        is_tracking = False
        frames_lost = 0
        ball_counter = 0
        frame_idx = 0

        print(
            f"--- PRZETWARZANIE (Maska X < {max_x_allowed}, Próg przeskoku X = {jump_x_threshold}px) ---"
        )

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

            # --- MASKA ROZMIAROWA (ROI) ---
            # Sztuczne wyzerowanie prawego marginesu obrazu, gdzie znajduje się mechanizm
            diff[:, max_x_allowed:] = 0

            # Progowanie i czyszczenie szumu
            _, binary = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
            binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            valid_contours = [
                cnt for cnt in contours if cv2.contourArea(cnt) >= min_area
            ]

            ball_pos = None
            if valid_contours:
                largest = max(valid_contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    ball_pos = (cx, cy)

            # --- LOGIKA ŚLEDZENIA I DZIELENIA TRAJEKTORII ---
            if not is_tracking:
                if ball_pos is not None:
                    is_tracking = True
                    current_trajectory = [(frame_idx, ball_pos[0], ball_pos[1])]
                    frames_lost = 0
            else:
                if ball_pos is not None:
                    prev_x = current_trajectory[-1][1]
                    curr_x = ball_pos[0]

                    # KLUCZOWY WARUNEK: Jeśli nowa pozycja X przeskoczyła w prawo (jest większa o próg),
                    # oznacza to, że weszła nowa kulka!
                    if (curr_x - prev_x) > jump_x_threshold:
                        ball_counter = save_trajectory_file(
                            current_trajectory, ball_counter
                        )
                        # Start nowej trajektorii od obecnej klatki
                        current_trajectory = [(frame_idx, ball_pos[0], ball_pos[1])]
                        frames_lost = 0
                    else:
                        current_trajectory.append((frame_idx, ball_pos[0], ball_pos[1]))
                        frames_lost = 0
                else:
                    frames_lost += 1

                # Koniec śledzenia przy utracie widoczności obiekty
                if frames_lost > max_gap_frames:
                    ball_counter = save_trajectory_file(
                        current_trajectory, ball_counter
                    )
                    is_tracking = False
                    current_trajectory = []
                    frames_lost = 0

            frame_idx += 1

        # Zapis ostatniej trajektorii po zakończeniu pliku
        if is_tracking and current_trajectory:
            ball_counter = save_trajectory_file(current_trajectory, ball_counter)

    print(f"\nGotowe! Rozdzielono i zapisano {ball_counter} czystych trajektorii.")


# Uruchomienie skryptu:
track_and_split_trajectories(
    raw_file_path="film.raw",
    width=2448,
    height=2048,
    max_x_allowed=2300,  # Dostosuj tę wartość, jeśli mechanizm sięga głębiej
    jump_x_threshold=80,  # Przeskok w prawo o >80px wymusza nowy plik
)
