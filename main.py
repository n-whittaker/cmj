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

trial_number = 1

# FOR EACH CMJ
for file in cmjs:
    data = read_c3d(file, read_mocap=True)  # Read the c3d data...

    mocap = data["MoCap"]  # And separate it into the motion capture data..
    grf = data["GRF"]  # The GRF data
    info = data["Info"]  # And general info

    mocapDF = pd.DataFrame(mocap)
    grfDF = pd.DataFrame(grf)  # Create a dataFrame from the GRF data
    infoDF = pd.DataFrame(info)

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

    # Draw phase marker lines (COMMENTED OUT AS MAKES GRAPH EASIER TO LOOK AT)
    # ax2.axhline(0, color="gray", linestyle=":")
    # ax2.axvline(ED_start, color="red")
    # ax2.axvline(ED_end, color="red")
    # # Not plotting con start as it = ED_end
    # ax2.axvline(con_end, color="blue")
    # ax2.axvline(landing_start, color="green")
    # ax2.axvline(landing_end, color="green")

    # Spans
    ax2.axvspan(ED_start, ED_end, color="red", alpha=0.1, label="Eccentric Deceleration")  # Eccentric Deceleration
    ax2.axvspan(ED_end, con_end, color="blue", alpha=0.1, label="Concentric")  # Concentric
    ax2.axvspan(landing_start, landing_end, color="green", alpha=0.1, label="Landing")  # Concentric

    # Set the title (using the file name for reference)
    ax1.set_title(f"CMJ Phases: {os.path.basename(file)}")

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    # This is just for looking at MOMENT graphs to see if they look right
    ax3 = ax1.twinx()
    # ax3.plot(mocapDF["LHipMoment_x"], label="Hip Moment", color="black")
    # ax3.plot(mocapDF["LKneeMoment_x"], label="Knee Moment", color="black")
    # ax3.plot(mocapDF["LAnkleMoment_x"], label="Ankle Moment", color="black")
    # ax3.plot(mocapDF["LHipAngles_x"], label="Hip  Angle", color="black")
    # ax3.plot(mocapDF["LKneeAngles_x"], label="Knee Angle", color="black")
    # ax3.plot(mocapDF["LAnkleAngles_x"], label="Ankle Angle", color="black")

    plt.tight_layout()
    plt.show()

    # -------------------------
    # Variable calculations
    # -------------------------

    # Jump height
    g = 9.81  # Gravity
    v_takeoff = com_vel_z.loc[con_end]  # Extract velocity at takeoff
    # Calculate jump height using the impulse-momentum equation
    jump_height = (v_takeoff ** 2) / (2 * g)
    jump_height_cm = jump_height * 100
    jump_height_cm_rounded = round(jump_height_cm, 2)  # Round to 2 decimal places

    # Peak GRFV Left + Right
    peak_GRFv_Left = round(grfDF["Fz2"].max(), 2)
    peak_GRFv_Right = round(grfDF["Fz1"].max(), 2)

    # Peak HKA Extension Moments
    peak_H_moment_Left = round(mocapDF["LHipMoment_x"].max(), 2)
    peak_H_moment_Right = round(mocapDF["RHipMoment_x"].max(), 2)
    peak_K_moment_Left = round(mocapDF["LKneeMoment_x"].max(), 2)
    peak_K_moment_Right = round(mocapDF["RKneeMoment_x"].max(), 2)
    peak_A_moment_Left = round(mocapDF["LAnkleMoment_x"].max(), 2)
    peak_A_moment_Right = round(mocapDF["RAnkleMoment_x"].max(), 2)

    # Peak HKA Flexion Angles
    peak_H_angle_Left = round(mocapDF["LHipAngles_x"].max(), 2)
    peak_H_angle_Right = round(mocapDF["RHipAngles_x"].max(), 2)
    peak_K_angle_Left = round(mocapDF["LKneeAngles_x"].max(), 2)
    peak_K_angle_Right = round(mocapDF["RKneeAngles_x"].max(), 2)
    peak_A_angle_Left = round(mocapDF["LAnkleAngles_x"].max(), 2)
    peak_A_angle_Right = round(mocapDF["RAnkleAngles_x"].max(), 2)

    # Impulse using trapezoid rule
    # Time step based on  sampling rate
    dt = 1.0 / sampling_rate
    # Eccentric deceleration phase
    impulse_ED_right = round(np.trapz(grfDF["Fz1"][ED_start:ED_end], dx=dt), 2)
    impulse_ED_left = round(np.trapz(grfDF["Fz2"][ED_start:ED_end], dx=dt), 2)
    # Concentric phase
    impulse_con_right = round(np.trapz(grfDF["Fz1"][ED_end:con_end], dx=dt), 2)
    impulse_con_left = round(np.trapz(grfDF["Fz2"][ED_end:con_end], dx=dt), 2)
    # Landing phase:
    impulse_landing_right = round(np.trapz(grfDF["Fz1"][landing_start:landing_end], dx=dt), 2)
    impulse_landing_left = round(np.trapz(grfDF["Fz2"][landing_start:landing_end], dx=dt), 2)

    # Asymmetry Indices
    # AI = ((Dominant vs Non-Dominant) / (Maximum of dominant and non-dominant)) * 100
    # May need to look at spreadsheet to determine dominant legs

    # -------------------------
    # Print Variables
    # -------------------------
    print("Trial number: ", trial_number)
    print("Jump Height (cm): ", jump_height_cm_rounded)
    print("Peak GRFv Left (N): ", peak_GRFv_Left)
    print("Peak GRFv Right (N): ", peak_GRFv_Right)
    print("Peak H Extension Moment (N): LEFT: ", peak_H_moment_Left, " RIGHT: ", peak_H_moment_Right)
    print("Peak K Extension Moment (N): LEFT: ", peak_K_moment_Left, " RIGHT: ", peak_K_moment_Right)
    print("Peak A Extension Moment (N): LEFT: ", peak_A_moment_Left, " RIGHT: ", peak_A_moment_Right)
    print("Peak H Flexion Angle (N): LEFT: ", peak_H_angle_Left, " RIGHT: ", peak_H_angle_Right)
    print("Peak K Flexion Angle (N): LEFT: ", peak_K_angle_Left, " RIGHT: ", peak_K_angle_Right)
    print("Peak A Flexion Angle (N): LEFT: ", peak_A_angle_Left, " RIGHT: ", peak_A_angle_Right)
    print("ED Impulse (N.s) LEFT:", impulse_ED_left, " RIGHT:", impulse_ED_right)
    print("CON Impulse (N.s) LEFT:", impulse_con_left, " RIGHT:", impulse_con_right)
    print("Landing Impulse (N.s) LEFT:", impulse_landing_left, " RIGHT:", impulse_landing_right)

    print("")  # Spacer for readability in the terminal

    trial_number += 1  # iterate trial number


