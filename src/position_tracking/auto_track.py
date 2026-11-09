import os
import sys
import tkinter as tk
from tkinter import simpledialog
from datetime import datetime

# 1. Set the MVS environment variable so the SDK finds 'libMvCameraControl.so'
if not os.environ.get('MVCAM_COMMON_RUNENV'):
    os.environ['MVCAM_COMMON_RUNENV'] = '/opt/MVS/lib'

# 2. Add the Hikrobot MVS Python SDK bindings path
mvs_path = "/opt/MVS/Samples/64/Python/MvImport"
if os.path.exists(mvs_path):
    sys.path.append(mvs_path)
else:
    print(f"Error: MvImport path not found at {mvs_path}")
    sys.exit(1)

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plot generation
import matplotlib.pyplot as plt

from MvCameraControl_class import *

def prompt_for_filename():
    """Opens a small popup dialog to gather measurement metadata and generate a filename."""
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    root.attributes('-topmost', True) # Bring popup to the front
    
    # 1. Composite or Isotropic
    mat_type = simpledialog.askstring("Measurement Metadata", "Material type (com / iso):", initialvalue="com", parent=root)
    if not mat_type:
        root.destroy()
        return None
        
    # 2. Used Sandpaper
    sandpaper = simpledialog.askstring("Measurement Metadata", "Used sandpaper (e.g., 400-80, 120--):", initialvalue="400-80", parent=root)
    if not sandpaper:
        root.destroy()
        return None
        
    # 3. Angle
    angle = simpledialog.askstring("Measurement Metadata", "Angle (e.g., 30):", initialvalue="30", parent=root)
    if not angle:
        root.destroy()
        return None
        
    # 4. Grain Size
    grain_size = simpledialog.askstring("Measurement Metadata", "Grain size (e.g., 1.5, 2.5):", initialvalue="2.5", parent=root)
    if not grain_size:
        root.destroy()
        return None
        
    root.destroy()

    # Generate timestamp (Year-Month-Day_Hour-Minute-Second)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Construct clean filename with metadata and timestamp
    filename = f"processing_results/{mat_type}_{sandpaper}_ang{angle}_grain{grain_size}_{timestamp}_trajectories.txt"
    return filename
    

def render_trajectory_plot(completed_trajectories, width, height):
    """Renders all completed trajectories onto an image to display in a second OpenCV window."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(height, 0)
    ax.set_ylim(0, width)  # Invert Y to match native image coordinates
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    for traj in completed_trajectories:
        if len(traj) > 0:
            xs = traj[:, 2]
            ys = traj[:, 1]
            ax.plot(xs, ys, color='cyan', alpha=0.7, linewidth=1.5)

    ax.set_title("Live Trajectories Summary", color='white', fontsize=10)
    ax.set_xlabel("X (pixels)", color='white')
    ax.set_ylabel("Y (pixels)", color='white')
    
    fig.tight_layout()
    
    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    plot_img_rgba = np.asarray(rgba_buffer)
    plt.close(fig)
    
    plot_img_bgr = cv2.cvtColor(plot_img_rgba, cv2.COLOR_RGBA2BGR)
    return plot_img_bgr

def main():
    print("--- 1. Starting main script ---")
    
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tDeviceType = MV_GIGE_DEVICE | MV_USB_DEVICE
    
    ret = MvCamera.MV_CC_EnumDevices(tDeviceType, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        print("❌ No Hikrobot devices found or enumeration failed!")
        sys.exit(1)

    stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    cam = MvCamera()

    if cam.MV_CC_CreateHandle(stDeviceList) != 0 or cam.MV_CC_OpenDevice() != 0:
        print("❌ Failed to open camera device!")
        sys.exit(1)

    cam.MV_CC_SetEnumValue("TriggerMode", 0)

    if cam.MV_CC_StartGrabbing() != 0:
        print("❌ Failed to start grabbing!")
        sys.exit(1)

    # Configure Camera Parameters
    cam.MV_CC_SetFloatValue("ExposureTime", 630.0)
    cam.MV_CC_SetFloatValue("Gain", 5.0)

    stExposure = MVCC_FLOATVALUE()
    memset(byref(stExposure), 0, sizeof(stExposure))
    cam.MV_CC_GetFloatValue("ExposureTime", stExposure)
    print(f"Current Exposure Time: {stExposure.fCurValue} microseconds")

    stGain = MVCC_FLOATVALUE()
    memset(byref(stGain), 0, sizeof(stGain))
    cam.MV_CC_GetFloatValue("Gain", stGain)
    print(f"Current Gain: {stGain.fCurValue} dB")

    stFrameRate = MVCC_FLOATVALUE()
    memset(byref(stFrameRate), 0, sizeof(stFrameRate))
    cam.MV_CC_GetFloatValue("ResultingFrameRate", stFrameRate)
    print(f"Current Resulting Frame Rate: {stFrameRate.fCurValue} FPS")

    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(stParam))
    cam.MV_CC_GetIntValue("PayloadSize", stParam)
    buf_size = stParam.nCurValue

    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    data_buf = (c_ubyte * buf_size)()

    ret_frame = cam.MV_CC_GetOneFrameTimeout(byref(data_buf), buf_size, stFrameInfo, 1000)
    if ret_frame != 0:
        print(f"❌ Failed to grab initial frame. Error code: {ret_frame:x}")
        sys.exit(1)

    height = stFrameInfo.nHeight
    width = stFrameInfo.nWidth
    
    print(f"✅ Camera Connected Successfully! Native Resolution: {width}x{height}")
    print(f"Capturing background model... Please keep the stream clear.")
    
    # 1. CAPTURE BACKGROUND MODEL
    num_bg_frames = 15
    bg_frames = []
    
    for _ in range(num_bg_frames):
        if cam.MV_CC_GetOneFrameTimeout(byref(data_buf), buf_size, stFrameInfo, 1000) == 0:
            frame_np = np.frombuffer(data_buf, dtype=np.uint8, count=stFrameInfo.nFrameLen)
            nDim = 3 if stFrameInfo.enPixelType in [PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_BGR8_Packed] else 1
            frame_np = frame_np.reshape((height, width)) if nDim == 1 else frame_np.reshape((height, width, nDim))
            
            if nDim > 1:
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
            bg_frames.append(frame_np.astype(np.float32))

    mean_background = np.mean(bg_frames, axis=0)
    print("Background model established!")

    # 2. SETUP TRACKING PARAMETERS
    threshold_value = 20
    min_area = 20
    max_area = 8000
    min_x_allowed, max_x_allowed = 100, width - 100
    min_y_allowed, max_y_allowed = 200, height - 200  
    max_jump_distance = 800
    max_gap_frames = 8
    min_points = 3

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    current_trajectory = []
    completed_trajectories = []
    is_tracking = False
    frames_lost = 0
    frame_idx = 0
    ball_id_counter = 0

    # Create resizable OpenCV Windows
    cv2.namedWindow("Hikrobot MVS - Real-Time Stream Tracking", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Live Trajectory Plot", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hikrobot MVS - Real-Time Stream Tracking", 800, 800)
    cv2.resizeWindow("Live Trajectory Plot", 600, 600)

    try:
        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(byref(data_buf), buf_size, stFrameInfo, 1000)
            if ret != 0:
                continue

            nDim = 3 if stFrameInfo.enPixelType in [PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_BGR8_Packed] else 1
            img_np = np.asarray(data_buf)[:stFrameInfo.nFrameLen]
            
            if nDim == 1:
                img_np = img_np.reshape((height, width))
                gray_frame = img_np.astype(np.float32)
                raw_display = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            else:
                img_np = img_np.reshape((height, width, nDim))
                raw_display = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if stFrameInfo.enPixelType == PixelType_Gvsp_RGB8_Packed else img_np.copy()
                gray_frame = cv2.cvtColor(raw_display, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # --- Background Subtraction & Tracking in Native Space ---
            diff = np.abs(gray_frame - mean_background).astype(np.uint8)
            diff[:, :min_x_allowed] = 0
            diff[:, max_x_allowed:] = 0
            diff[:min_y_allowed, :] = 0
            diff[max_y_allowed:, :] = 0

            _, binary = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
            binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]

            ball_pos = None
            if valid_contours:
                largest = max(valid_contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    ball_pos = (cx, cy)

            # State Machine & File Logging
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
                            traj_np = np.array(current_trajectory)
                            completed_trajectories.append(traj_np)
                            ball_id_counter += 1

                        current_trajectory = [(frame_idx, ball_pos[0], ball_pos[1])]
                        frames_lost = 0
                    else:
                        current_trajectory.append((frame_idx, ball_pos[0], ball_pos[1]))
                        frames_lost = 0
                else:
                    frames_lost += 1

                if frames_lost > max_gap_frames:
                    if len(current_trajectory) >= min_points:
                        traj_np = np.array(current_trajectory)
                        completed_trajectories.append(traj_np)
                        ball_id_counter += 1

                    is_tracking = False
                    current_trajectory = []
                    frames_lost = 0

            # --- VISUALIZATION: Rotate display frame 90° CCW for viewing ---
            display_frame = cv2.rotate(raw_display, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if ball_pos is not None:
                bx_rot = ball_pos[1]
                by_rot = width - 1 - ball_pos[0]
                cv2.circle(display_frame, (bx_rot, by_rot), 5, (0, 0, 255), -1)

            if len(current_trajectory) > 1:
                pts_rot = np.array([[pt[2], width - 1 - pt[1]] for pt in current_trajectory], np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts_rot], isClosed=False, color=(0, 255, 0), thickness=2)

            frame_idx += 1

            # Throttle UI rendering
            if frame_idx % 10 == 0:
                cv2.imshow("Hikrobot MVS - Real-Time Stream Tracking", display_frame)
                plot_frame = render_trajectory_plot(completed_trajectories, width, height)
                cv2.imshow("Live Trajectory Plot", plot_frame)

            # CRITICAL: cv2.waitKey is required to keep OpenCV windows responsive and detect Esc key
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc pressed
                if len(current_trajectory) >= min_points:
                    traj_np = np.array(current_trajectory)
                    completed_trajectories.append(traj_np)
                break

    finally:
        # Clean up camera safely first
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()

        # --- POPUP DIALOG FOR FILENAME & SAVING ON EXIT ---
        if completed_trajectories:
            print("Measurement ended. Opening naming popup...")
            custom_filename = prompt_for_filename()
            
            if custom_filename:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                output_txt_path = os.path.join(script_dir, custom_filename)
                
                with open(output_txt_path, "w") as txt_file:
                    txt_file.write("ball_id\tframe\tx\ty\n")
                    for b_id, traj in enumerate(completed_trajectories):
                        for f_num, x, y in traj:
                            txt_file.write(f"{b_id}\t{f_num}\t{x}\t{y}\n")
                print(f"✅ Trajectories successfully saved to: {output_txt_path}")
            else:
                print("⚠️ Naming cancelled or closed. Data not saved.")
        else:
            print("⚠️ No valid trajectories collected during this session.")

if __name__ == "__main__":
    main()