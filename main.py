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

    max_velocity = com_vel_z.idxmax()
    vel_before_max_velocity = com_vel_z[:max_velocity]

    # Eccentric Deceleration Phase (Max neg-vel to zero)
    ED_start = vel_before_max_velocity.idxmin()  # Lowest vel before flight phase
    ED_end = vel_before_max_velocity[ED_start:].ge(0).idxmax()  # Find first 0 value after ED_start

    # Concentric Phase (zero vel to takeoff)
    con_start = ED_start  # Zero vel
    con_end = max_velocity  # takeoff

    # Landing Phase (Landing to Zero vel)
    top_of_flight_vel = com_vel_z[con_end:].ge(0).idxmin()  # mid-point of flight where vel = 0
    landing_start = grfTotal[top_of_flight_vel:].idxmin()
    landing_end = com_vel_z[landing_start:].ge(0).idxmax()


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
    ax2.plot(com_vel_z, velocity_graph_color, linestyle="--", label="Velocity (m/s)")
    ax2.set_ylabel("Velocity (m/s)", color=velocity_graph_color)
    ax2.tick_params(axis="y", labelcolor=velocity_graph_color)
    ax2.set_ylim(-3, 8)

    # Draw phase marker lines
    ax2.axhline(0, color="gray", linestyle=":")
    ax2.axvline(ED_start, color="red")
    ax2.axvline(ED_end, color="red")
    # Not plotting con start as it = ED_end
    ax2.axvline(con_end, color="blue")
    ax2.axvline(landing_start, color="green")
    ax2.axvline(landing_end, color="green")

    # Spans
    ax2.axvspan(ED_start, ED_end, color="red", alpha=0.1)  # Eccentric Deceleration
    ax2.axvspan(ED_end, con_end, color="blue", alpha=0.1)  # Concentric
    ax2.axvspan(landing_start, landing_end, color="green", alpha=0.1)  # Concentric

    # Set the title (using the file name for reference)
    ax1.set_title(f"CMJ Phases: {os.path.basename(file)}")

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.show()



