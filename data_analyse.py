from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
import os
import csv


Folder_path = "/Users/preethamreddy/Desktop/Data_analyse LAMP Lab/sample1"
folder = Path(Folder_path)
if not folder.exists():
    raise FileNotFoundError("Folder not found")


threshold = 1400


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
    print("file" , csv_file)
    return matrix


def MP_area(matrix, threshold):
    count = np.sum(matrix > threshold)
    #print(f"area (values > {threshold}): {count}")
    return count

    

def max_val_matrix(matrix, threshold):
    if matrix is None or matrix.size == 0:
        return 0, -1, -1

    rows = len(matrix)
    cols = len(matrix[0])
    max_val = 0
    max_row = -1
    max_col = -1

    for i in range(rows):
        for j in range(cols):
            value = matrix[i][j]
            # check each and every value
            if value > threshold and value > max_val:
                max_val = value
                max_row = i
                max_col = j

    # If we never found any value > threshold
    if max_row == -1:
        return 0, -1, -1

    print(f"Max value: {max_val} (Row: {max_row}, Col: {max_col})")
    return max_val, max_row, max_col

    
# def neighbours_max_value(matrix, center_row, center_col,max_val):
#     if max_val == 0 or center_row == -1 or center_col == -1:
#         return []

#     rows = len(matrix)
#     cols = len(matrix[0])
#     neighbours = []

#     for i in range(center_row - 1, center_row + 2):   # row-1, row, row+1
#         if 0 <= i < rows:
#             row_vals = []
#             for j in range(center_col - 1, center_col + 2):  # col-1, col, col+1
#                 if 0 <= j < cols:
#                     row_vals.append(matrix[i][j])
#             neighbours.append(row_vals)
#     print(neighbours)
#     return neighbours
                
# def avg_mp_temp(neighbours):
#     total = 0
#     count = 0
#     avg_temp = 0
#     for i in range(len(neighbours)):
#         for j in range(len(neighbours[i])):
#             total += neighbours[i][j]
#             count += 1

#     if count == 0:
#         return 950.0

#     avg_temp = round(total / count, 1)
#     print(avg_temp)
#     return avg_temp



def avg_above_threshold(matrix, threshold):
    # Convert to NumPy array for fast filtering (optional)
    arr = np.array(matrix)
    
    # Filter values greater than threshold
    values_above = arr[arr > threshold]

    if len(values_above) == 0:
        return 950.0  # default if no values above threshold
    
    avg_val = round(values_above.mean(), 1)
    print(avg_val)
    return avg_val

def write_area_and_avg_csv(csv_files, threshold):
    os.makedirs("mp_area", exist_ok=True)
    os.makedirs("mp_avg_temp", exist_ok=True)
    area_csv = os.path.join("mp_area", "mp_area_vs_time.csv")
    avg_csv  = os.path.join("mp_avg_temp", "mp_avg_temp_vs_time.csv")

    with open(area_csv, "w", newline="") as fa, open(avg_csv, "w", newline="") as ft:
        area_w = csv.writer(fa)
        avg_w  = csv.writer(ft)
        area_w.writerow(["Time (s)", "MP_Area"])
        avg_w.writerow(["Time (s)", "MP_Avg_Temp"])

        for idx, csv_path in enumerate(csv_files):
            mat = image_data_matrix(csv_path)
            time_sec = (idx + 1) / 27

            # Skip or handle empty matrix safely
            if mat is None or mat.size == 0:
                mp_val = 0
                avg_val = 950
            else:
                mp_val = MP_area(mat, threshold)
                max_val, r, c = max_val_matrix(mat, threshold)
                neigh = neighbours_max_value(mat, r, c, max_val)
                avg_val = avg_mp_temp(neigh)

            # Write each frame result
            area_w.writerow([f"{time_sec:.3f}", mp_val])
            avg_w.writerow([f"{time_sec:.3f}", avg_val])

            # Optional debug print
            print(f"{csv_path.name}: max={max_val:.1f}, avg={avg_val:.1f}")

# lightweight smoothing to make contours smooth
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



def graphs(window_ma=5, window_med=5):
    mp_area_csv = os.path.join("mp_area", "mp_area_vs_time.csv")
    mp_avg_temp_csv = os.path.join("mp_avg_temp", "mp_avg_temp_vs_time.csv")

    df_area = pd.read_csv(mp_area_csv)
    df_avg  = pd.read_csv(mp_avg_temp_csv)

    # ensure numeric
    df_area["Time (s)"] = pd.to_numeric(df_area["Time (s)"], errors="coerce")
    df_area["MP_Area"]  = pd.to_numeric(df_area["MP_Area"], errors="coerce")

    df_avg["Time (s)"]      = pd.to_numeric(df_avg["Time (s)"], errors="coerce")
    df_avg["MP_Avg_Temp"]   = pd.to_numeric(df_avg["MP_Avg_Temp"], errors="coerce")

    # ---- 1) MOVING AVERAGE (rolling mean) ----
    # center=True so the smoothing follows the data, not lagging
    df_area["MP_Area_MA"] = df_area["MP_Area"].rolling(window=window_ma, center=True, min_periods=1).mean()
    df_avg["MP_Avg_Temp_MA"] = df_avg["MP_Avg_Temp"].rolling(window=window_ma, center=True, min_periods=1).mean()

    # ---- 2) MEDIAN FILTER (rolling median) ----
    df_area["MP_Area_MED"] = df_area["MP_Area"].rolling(window=window_med, center=True, min_periods=1).median()
    df_avg["MP_Avg_Temp_MED"] = df_avg["MP_Avg_Temp"].rolling(window=window_med, center=True, min_periods=1).median()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # ====== LEFT: MP AREA ======
    axes[0].plot(df_area["Time (s)"], df_area["MP_Area"], color="lightblue", label="Raw", linewidth=1)
    axes[0].plot(df_area["Time (s)"], df_area["MP_Area_MA"], color="blue", label=f"MA ({window_ma})", linewidth=2)
    axes[0].plot(df_area["Time (s)"], df_area["MP_Area_MED"], color="navy", linestyle="--", label=f"Median ({window_med})", linewidth=1.5)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("MP Area")
    axes[0].set_title("MP Area vs Time")
    axes[0].legend()

    # ====== RIGHT: MP AVG TEMP ======
    axes[1].plot(df_avg["Time (s)"], df_avg["MP_Avg_Temp"], color="salmon", label="Raw", linewidth=1)
    axes[1].plot(df_avg["Time (s)"], df_avg["MP_Avg_Temp_MA"], color="red", label=f"MA ({window_ma})", linewidth=2)
    axes[1].plot(df_avg["Time (s)"], df_avg["MP_Avg_Temp_MED"], color="darkred", linestyle="--", label=f"Median ({window_med})", linewidth=1.5)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("MP Avg Temp")
    axes[1].set_title("MP Avg Temp vs Time")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("graphs.png", bbox_inches="tight")
    plt.close(fig)


# def cooling_point():
#     avg_mp_temp = 1


# def melting_point():


if __name__ == "__main__": 
    csv_files = sorted(folder.glob("*.csv"))
    print(f"found {len(csv_files)} csv files in {folder}")
    write_area_and_avg_csv(csv_files, threshold)
    # export_isotherm_video_fast(Folder_path, out_video_path="isotherm_video.mp4", fps=8, vmin=1550, vmax=2000, cmap="inferno")
    # graphs()
    # matrix  = image_data_matrix(csv_files[0])
    # max_val,r, c = max_val_matrix(matrix, threshold)
    # neighbours = neighbours_max_value(matrix, r, c,max_val)
    # avg_mp_temp(neighbours)