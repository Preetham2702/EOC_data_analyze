from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
import os
import csv

def image_data_matrix(csv_file):    
    rows = []
    start = False

    with open(csv_file, "r") as f:
        for line in f:
            line = line.strip()

            if line.lower().startswith("image data"):
                start = True
                continue
            if start and not line:
                break

            if start:
                parts = line.replace(';', ',').split(',')

                nums =  [float(p) for p in parts if p.strip() != ""]
                if nums:
                    rows.append(nums)
    
    matrix = np.array(rows)
    #np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=False) # prints all lines
    return matrix


def MP_area(matrix, threshold):
    count = np.sum(matrix > threshold)
    #print(f"area (values > {threshold}): {count}")
    return count

    

def max_val_matrix(matrix, threshold):
    max_val = 0
    max_r = -1
    max_c = -1

    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    for i in range(rows):
        for j in range(cols):
            val = matrix[i][j]
            if val > threshold and val > max_val:
                max_val = val
                max_r = i
                max_c = j
    #print(f"max_val: {max_val}, max_r: {max_r}, max_c: {max_c}")
    return max_val, max_r, max_c


def max_values_avg_temp(matrix, threshold):
    arr = np.array(matrix)
    
    # Filter values greater than threshold
    values_above = arr[arr > threshold]

    if len(values_above) == 0:
        return 950.0  # default if no values above threshold
    
    avg_val = round(values_above.mean(), 1)
    #print(avg_val)
    return avg_val

    
def neighbours_of_max_value(matrix, max_val, center_row, center_col):
    if center_row == -1 or center_col == -1:
        return []

    rows = len(matrix)
    cols = len(matrix[0])
    neighbours = []

    for i in range(center_row - 1, center_row + 2):   # r-1, r, r+1
        if 0 <= i < rows:
            row_vals = []
            for j in range(center_col - 1, center_col + 2):  # c-1, c, c+1
                if 0 <= j < cols:
                    row_vals.append(matrix[i][j])
            neighbours.append(row_vals)
    # debug
    # print(neighbours)
    return neighbours
                

def avg_mp_temp(neighbours):
    total = 0
    count = 0
    avg_temp = 0
    for i in range(len(neighbours)):
        for j in range(len(neighbours[i])):
            total += neighbours[i][j]
            count += 1

    if count == 0:
        return 950.0
    # print(count)
    avg_temp = round(total / count, 1)
    # print(avg_temp)
    return avg_temp


def smooth_matrix(matrix, passes=2):
    a = matrix.astype(float)
    for _ in range(passes):
        pad = np.pad(a, 1, mode="edge")
        a = (
            pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
            pad[1:-1, :-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:] +
            pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]
        ) / 9.0
    return a


def export_isotherm_video_fast(folder_path, out_video_path="isotherm_video.mp4",
                               fps=8, vmin=1550, vmax=2000, cmap="inferno"):
    folder = Path(folder_path)
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    # yuv420p makes the MP4 widely compatible
    writer = imageio.get_writer(
        out_video_path,
        fps=fps,
        codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
        ffmpeg_log_level="warning",
    )
    try:
        for i, csv_path in enumerate(csv_files, 1):
            matrix = image_data_matrix(csv_path)

            rows, cols = matrix.shape
            fig, ax = plt.subplots(figsize=(cols/5, rows/5), dpi=120)
            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation="none", aspect="equal", origin="upper")

            # contours (same as before)
            iso = [1604, 1750, 1850, 2000]
            sm = smooth_matrix(matrix, passes=2)
            cs = ax.contour(sm, levels=iso, colors="white", linewidths=1.0, origin="upper")
            ax.clabel(cs, fmt="%d", inline=True, fontsize=6, colors="white")

            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout(pad=0.2)

            # figure -> RGB numpy
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            h, w = fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0]
            frame = frame.reshape(h, w, 3)

            writer.append_data(frame)
            plt.close(fig)

            if i % 200 == 0:
                print(f"Rendered {i}/{len(csv_files)} frames...")
    finally:
        writer.close()
    print(f"✅ Wrote video: {out_video_path}")



def graphs(folder_path, window_ma=5, window_med=5):
    folder_name = os.path.basename(os.path.normpath(folder_path))  # e.g. AR_80_Top

    mp_area_csv = os.path.join("mp_area", "mp_area_vs_time.csv")
    mp_avg_temp_csv = os.path.join("mp_avg_temp", "mp_avg_temp_vs_time.csv")

    df_area = pd.read_csv(mp_area_csv)
    df_avg  = pd.read_csv(mp_avg_temp_csv)

    # Ensure numeric
    df_area["Time (s)"] = pd.to_numeric(df_area["Time (s)"], errors="coerce")
    df_area["MP_Area"]  = pd.to_numeric(df_area["MP_Area"], errors="coerce")

    df_avg["Time (s)"]      = pd.to_numeric(df_avg["Time (s)"], errors="coerce")
    df_avg["MP_Avg_Temp"]   = pd.to_numeric(df_avg["MP_Avg_Temp"], errors="coerce")

    # Moving Average (MA)
    df_area["MP_Area_MA"] = df_area["MP_Area"].rolling(window=window_ma, center=True, min_periods=1).mean()
    df_avg["MP_Avg_Temp_MA"] = df_avg["MP_Avg_Temp"].rolling(window=window_ma, center=True, min_periods=1).mean()

    # Median filter
    df_area["MP_Area_MED"] = df_area["MP_Area"].rolling(window=window_med, center=True, min_periods=1).median()
    df_avg["MP_Avg_Temp_MED"] = df_avg["MP_Avg_Temp"].rolling(window=window_med, center=True, min_periods=1).median()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=120)

    # ====== LEFT: MP AREA ======
    axes[0].plot(df_area["Time (s)"], df_area["MP_Area"], color="lightblue", label="Raw", linewidth=1)
    axes[0].plot(df_area["Time (s)"], df_area["MP_Area_MA"], color="blue", label=f"MA ({window_ma})", linewidth=2)
    axes[0].plot(df_area["Time (s)"], df_area["MP_Area_MED"], color="navy", linestyle="--", label=f"Median ({window_med})", linewidth=1.5)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("MP Area")
    axes[0].set_title("MP Area vs Time")
    axes[0].legend()
    axes[0].set_ylim(0, 1200)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # ====== RIGHT: MP AVG TEMP ======
    axes[1].plot(df_avg["Time (s)"], df_avg["MP_Avg_Temp"], color="salmon", label="Raw", linewidth=1)
    axes[1].plot(df_avg["Time (s)"], df_avg["MP_Avg_Temp_MA"], color="red", label=f"MA ({window_ma})", linewidth=2)
    axes[1].plot(df_avg["Time (s)"], df_avg["MP_Avg_Temp_MED"], color="darkred", linestyle="--", label=f"Median ({window_med})", linewidth=1.5)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("MP Avg Temp")
    axes[1].set_title("MP Avg Temp vs Time")
    axes[1].legend()
    axes[1].set_ylim(0, 2500)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # ====== Clickable coordinate display ======
    # When you click a point, it will show (x, y) coordinates in terminal
    def onclick(event):
        if event.inaxes:
            print(f"Clicked at x={event.xdata:.3f}, y={event.ydata:.3f}")

    fig.canvas.mpl_connect("button_press_event", onclick)

    fig.tight_layout()

    output_filename = f"{folder_name}_graphs.png"
    fig.savefig(output_filename, bbox_inches="tight")
    plt.show()
    #plt.close(fig)



def print_missing_frames(folder_path):
    folder = Path(folder_path)
    csv_files = sorted(folder.glob("*.csv"))
    
    # collect frame numbers
    frame_numbers = [int(f.stem.split("_")[-1]) for f in csv_files]

    if not frame_numbers:
        print("No CSV files found.")
        return

    min_frame, max_frame = min(frame_numbers), max(frame_numbers)
    full_range = set(range(min_frame, max_frame + 1))
    existing = set(frame_numbers)
    missing = sorted(list(full_range - existing))

    print(f"Total frames expected: {len(full_range)}")
    print(f"Frames present: {len(existing)}")
    print(f"Missing frames: {len(missing)}")

    if missing:
        print("Missing frame numbers:")
        for f in missing:
            print(f"Image_{f:08d}.csv")
    else:
        print("No missing frames detected.")


def cooling_heating_points(avg_temp_csv):
    df = pd.read_csv(avg_temp_csv)
    df["Time (s)"] = pd.to_numeric(df["Time (s)"], errors="coerce")
    df["MP_Avg_Temp"] = pd.to_numeric(df["MP_Avg_Temp"], errors="coerce")

    df = df.dropna(subset=["Time (s)", "MP_Avg_Temp"]).reset_index(drop = True)

    rates = []
    for i in range(1,len(df)):
        delta_temp = df.loc[i, "MP_Avg_Temp"] - df.loc[i-1, "MP_Avg_Temp"]
        delta_time = df.loc[i, "Time (s)"] - df.loc[i-1, "Time (s)"]

        if delta_time != 0:
            rate = delta_temp / delta_time
            rates.append(rate)
        else:
            rates.append(0)

    print(rates)
    return rates



if __name__ == "__main__": 


    Folder_path = "/Users/preethamreddy/Desktop/Data_analyse LAMP Lab/EOC_data_analyze/AR_500_Top"
    folder = Path(Folder_path)
    if not folder.exists():
        raise FileNotFoundError("Folder not found")
    
    
    csv_files = sorted(folder.glob("*.csv"))

    #print_missing_frames(Folder_path)

    threshold = 1550

    frame_map = {}
    min_frame = None
    max_frame = None

    for csv_path in csv_files:
        stem = csv_path.stem                
        frame_no = int(stem.split("_")[-1])   

        frame_map[frame_no] = csv_path

        if min_frame is None or frame_no < min_frame:
            min_frame = frame_no
        if max_frame is None or frame_no > max_frame:
            max_frame = frame_no

    # 3) make output dirs
    os.makedirs("mp_area", exist_ok=True)
    os.makedirs("mp_avg_temp", exist_ok=True)

    area_csv_path = os.path.join("mp_area", "mp_area_vs_time.csv")
    avg_csv_path  = os.path.join("mp_avg_temp", "mp_avg_temp_vs_time.csv")

    last_avg_val = None   # carry-forward for missing frames

    with open(area_csv_path, "w", newline="") as fa, open(avg_csv_path, "w", newline="") as ft:
        area_w = csv.writer(fa)
        avg_w  = csv.writer(ft)

        # header
        area_w.writerow(["Time (s)", "MP_Area"])
        avg_w.writerow(["Time (s)", "MP_Avg_Temp"])

        # walk through EVERY frame number, even if file is missing
        for frame_no in range(min_frame, max_frame + 1):
            time_sec = frame_no / 27.0

            if frame_no in frame_map:
                csv_path = frame_map[frame_no]
                mat = image_data_matrix(csv_path)

                if mat is None or mat.size == 0:
                    mp_val  = 0.0
                    avg_val = 950.0
                else:
                    mp_val  = MP_area(mat, threshold)
                    avg_val = max_values_avg_temp(mat, threshold)

                # update carry-forward after computing a *real* frame
                last_avg_val = avg_val

            else:
                mp_val  = 0.0
                avg_val = last_avg_val if last_avg_val is not None else 950.0

            # write to CSVs
            area_w.writerow([f"{time_sec:.3f}", mp_val])
            avg_w.writerow([f"{time_sec:.3f}", avg_val])

    print("wrote values successfully")
    graphs(Folder_path)

    # export_isotherm_video_fast(Folder_path, out_video_path="isotherm_video.mp4", fps=8, vmin=1550, vmax=2000, cmap="inferno")

    # matrix  = image_data_matrix(csv_files[0])
    # max_val,r, c = max_val_matrix(matrix, threshold)
    # neighbours = neighbours_of_max_value(matrix, r, c,max_val)
    # avg_mp_temp(neighbours)
    
    
    # csv_file = "/Users/preethamreddy/Desktop/Data_analyse LAMP Lab/EOC_data_analyze/mp_avg_temp/mp_avg_temp_vs_time.csv"
    # cooling_heating_points(csv_file)