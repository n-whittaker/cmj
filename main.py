import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from read_c3d import read_c3d

# Setting the directory to run through
root_dir = "/Users/nick/Documents/University/Research Project/HT/AB 127331 Retest"

# Set display option to show all columns and prevent truncation
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_seq_items', None)


# Get CMJ c3d files
def getFiles():
    files = []

    # Walk through directory, check for "new session", go through files in new session, return CMJ c3ds
    # list.
    for (dirpath, dirnames, filenames) in os.walk(root_dir):

        if "New Session" in dirnames:
            new_session_path = os.path.join(dirpath, "New Session")

            for session_file in os.listdir(new_session_path):
                if session_file.endswith(f".c3d") and "CMJ" in session_file and "SL" not in session_file:
                    file_path = os.path.join(new_session_path, session_file)
                    files.append(file_path)
    return files


cmjs = getFiles()  # All the cmj c3d's in the directory provided

# FOR EACH CMJ
for file in cmjs:
    data = read_c3d(file, read_mocap=True)  # Read the c3d data...

    mocap = data["MoCap"]  # And separate it into the motion capture data..
    grf = data["GRF"]  # The GRF data
    info = data["Info"]  # And general info

    mocapDF = pd.DataFrame(mocap)
    grfDF = pd.DataFrame(grf)  # Create a dataFrame from the GRF data
    infoDF = pd.DataFrame(info)

    # print(mocap.columns)  # Printing column headings, showing everything I can pull out of the data
    # print(grfDF.columns)

    # Sum from the two forces to get total vertical GRF
    grfTotal = grfDF["Fz1"] + grfDF["Fz2"]
    com_vel_z = mocapDF["COMVelocity_z"] / 1000
    sampling_rate = 1000

    # bodyweight phase
    n_body_weight = 200  # assuming first 50 frames are quiet standing
    body_weight = np.mean(grfTotal[:n_body_weight])

    # Unloading phase (BW reduced by at least 2.5%), find first frame below that - Uses GRF
    unloading_threshold = body_weight - (0.025 * body_weight)
    unloading_framesBelow = np.where(grfTotal < unloading_threshold)[0]
    unloading_start = unloading_framesBelow[0] / sampling_rate

    # Braking phase - uses GRF
    braking_threshold = body_weight
    grf_after_unloading = grfTotal[unloading_start:]
    braking_framesBelow = np.where(grf_after_unloading >= braking_threshold)[0]
    unloading_start_frame = unloading_framesBelow[0]
    braking_start_frame = braking_framesBelow[0] + unloading_start_frame
    braking_start = braking_start_frame / sampling_rate

    # Propulsive phase: COM velocity goes above 0 AFTER braking phase.
    propulsive_threshold = 0
    velocity_after_braking = com_vel_z[braking_start:]
    propulsive_above_threshold = velocity_after_braking[velocity_after_braking > 0]
    propulsive_start = propulsive_above_threshold.index[0]

    # Flight phase
    flight_threshold = 0
    flight_below_threshold = np.where(grfTotal <= flight_threshold)[0]
    flight_start = flight_below_threshold[0] / sampling_rate

    # Landing Phase
    landing_start = flight_below_threshold[-1] / sampling_rate


    # -------------------------
    # Plotting
    # -------------------------
    # Define colors for clarity

    velocity_graph_color = "black"
    phase_color = "black"
    end_time = len(grfTotal) / sampling_rate

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot GRF on left axis
    # ax1.plot(grfTotal, grf_graph_color, label="GRF (N)")
    # ax1.set_xlabel("Frame")
    # ax1.set_ylabel("GRF (N)", color=grf_graph_color)
    # ax1.tick_params(axis="y", labelcolor=grf_graph_color)

    # FP1 = Right, FP2 = Left

    ax1.plot(grfDF["Fz1"], "red", label="Right")
    ax1.plot(grfDF["Fz2"], "blue", label="Left")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Force (N)")

    # Plot COM Velocity on right axis
    ax2 = ax1.twinx()
    ax2.plot(com_vel_z, velocity_graph_color, linestyle="--",label="Velocity (m/s)")
    ax2.set_ylabel("Velocity (m/s)", color=velocity_graph_color)
    ax2.tick_params(axis="y", labelcolor=velocity_graph_color)
    ax2.set_ylim(-3, 8)








    # Draw phase marker lines
    # ax1.axhline(body_weight, color="gray", linestyle=":" , label='Body weight')
    ax1.axvline(unloading_start, color="#f4f1de", linestyle="-")
    ax1.axvline(braking_start, color="#e07a5f", linestyle="-")
    ax1.axvline(propulsive_start, color="#3d405b", linestyle="-")
    ax1.axvline(flight_start, color="#81b29a", linestyle="-")
    ax1.axvline(landing_start, color="#f2cc8f", linestyle="-")


    # Spans
    ax1.axvspan(0, unloading_start, color='lightgray', alpha=0.1, label='Stance')
    ax1.axvspan(unloading_start, braking_start, color='#f4f1de', alpha=0.5, label='Unloading')
    ax1.axvspan(braking_start, propulsive_start, color='#e07a5f', alpha=0.3, label='Braking')
    ax1.axvspan(propulsive_start, flight_start, color='#3d405b', alpha=0.3, label='Propulsive')
    ax1.axvspan(flight_start, landing_start, color='#81b29a', alpha=0.3, label='Flight')
    ax1.axvspan(landing_start, end_time, color='#f2cc8f', alpha=0.3, label='Landing')

    # Set the title (using the file name for reference)
    ax1.set_title(f"CMJ Phases: {os.path.basename(file)}")

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.show()

    # Print phase start times (in seconds)
    # print("Unloading start (s):", unloading_start)
    # print("Braking start (s):", braking_start)
    # print("Propulsive start (s):", propulsive_start)
    # print("Flight start (s):", flight_start)
    # print("Landing start (s):", landing_start)

    print(info)