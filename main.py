import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from read_c3d import read_c3d

# Set display option to show all columns and prevent truncation
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_seq_items', None)

# Setting the directory to run through
root_dir = "/Users/nick/Documents/University/Research Project/HT/AB 127331 Retest"

# Checking ENF file for injured side
enf_file_path = None
for filename in os.listdir(root_dir):
    if filename.endswith(".enf"):
        enf_file_path = os.path.join(root_dir, filename)
        break

# Check if an ENF file was found and process it
if enf_file_path:
    with open(enf_file_path, "r") as file:
        for line in file:
            if line.startswith("INJURY="):
                injured_side = line.split("=")[1].strip()
                break
    print("Injured side:", injured_side)
else:
    print("No ENF file found in the directory.")


# Get CMJ c3d files
def getFiles():
    files = []
    # Walk through directory, check for "New Session" and ENF file.
    for (dirpath, dirnames, filenames) in os.walk(root_dir):

        # Look for CMJ c3d files in "New Session"
        if "New Session" in dirnames:
            new_session_path = os.path.join(dirpath, "New Session")
            for session_file in os.listdir(new_session_path):
                if session_file.endswith(".c3d") and "CMJ" in session_file and "SL" not in session_file:
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

    # Jump height sing the impulse-momentum equation
    g = 9.81  # Gravity
    v_takeoff = com_vel_z.loc[con_end]  # Extract velocity at takeoff
    jump_height_cm = round(((v_takeoff ** 2) / (2 * g)) * 100, 2)

    # Impulse using trapezoid rule
    dt = 1.0 / sampling_rate  # Time step based on  sampling rate
    impulse_ED_right = round(np.trapz(grfDF["Fz1"][ED_start:ED_end], dx=dt), 2)
    impulse_ED_left = round(np.trapz(grfDF["Fz2"][ED_start:ED_end], dx=dt), 2)
    impulse_con_right = round(np.trapz(grfDF["Fz1"][ED_end:con_end], dx=dt), 2)
    impulse_con_left = round(np.trapz(grfDF["Fz2"][ED_end:con_end], dx=dt), 2)
    impulse_landing_right = round(np.trapz(grfDF["Fz1"][landing_start:landing_end], dx=dt), 2)
    impulse_landing_left = round(np.trapz(grfDF["Fz2"][landing_start:landing_end], dx=dt), 2)

    # Peak GRFv
    peak_GRFv_ED_Right = round(grf["Fz1"][ED_start:ED_end].max(), 2)
    peak_GRFv_con_Right = round(grf["Fz1"][con_start:con_end].max(), 2)
    peak_GRFv_landing_Right = round(grf["Fz1"][landing_start:landing_end].max(), 2)
    peak_GRFv_ED_Left = round(grf["Fz2"][ED_start:ED_end].max(), 2)
    peak_GRFv_con_Left = round(grf["Fz2"][con_start:con_end].max(), 2)
    peak_GRFv_landing_Left = round(grf["Fz2"][landing_start:landing_end].max(), 2)

    # Peak Hip Flexion Angles
    peak_hip_angle_ED_Left = round(mocapDF["LHipAngles_x"][landing_start:landing_end].max(), 2)
    peak_hip_angle_con_Left = round(mocapDF["LHipAngles_x"][con_start:con_end].max(), 2)
    peak_hip_angle_landing_Left = round(mocapDF["LHipAngles_x"][landing_start:landing_end].max(), 2)
    peak_hip_angle_ED_Right = round(mocapDF["RHipAngles_x"][landing_start:landing_end].max(), 2)
    peak_hip_angle_con_Right = round(mocapDF["RHipAngles_x"][con_start:con_end].max(), 2)
    peak_hip_angle_landing_Right = round(mocapDF["RHipAngles_x"][landing_start:landing_end].max(), 2)

    # Peak Knee Flexion Angles
    peak_knee_angle_ED_Left = round(mocapDF["LKneeAngles_x"][landing_start:landing_end].max(), 2)
    peak_knee_angle_con_Left = round(mocapDF["LKneeAngles_x"][con_start:con_end].max(), 2)
    peak_knee_angle_landing_Left = round(mocapDF["LKneeAngles_x"][landing_start:landing_end].max(), 2)
    peak_knee_angle_ED_Right = round(mocapDF["RKneeAngles_x"][landing_start:landing_end].max(), 2)
    peak_knee_angle_con_Right = round(mocapDF["RKneeAngles_x"][con_start:con_end].max(), 2)
    peak_knee_angle_landing_Right = round(mocapDF["RKneeAngles_x"][landing_start:landing_end].max(), 2)

    # Peak Ankle Flexion Angles
    peak_ankle_angle_ED_Left = round(mocapDF["LKneeAngles_x"][landing_start:landing_end].max(), 2)
    peak_ankle_angle_con_Left = round(mocapDF["LKneeAngles_x"][con_start:con_end].max(), 2)
    peak_ankle_angle_landing_Left = round(mocapDF["LKneeAngles_x"][landing_start:landing_end].max(), 2)
    peak_ankle_angle_ED_Right = round(mocapDF["RKneeAngles_x"][landing_start:landing_end].max(), 2)
    peak_ankle_angle_con_Right = round(mocapDF["RKneeAngles_x"][con_start:con_end].max(), 2)
    peak_ankle_angle_landing_Right = round(mocapDF["RKneeAngles_x"][landing_start:landing_end].max(), 2)

    # Peak Hip Extension Moment
    peak_hip_moment_ED_Left = round(mocapDF["LHipMoment_x"][landing_start:landing_end].max(), 2)
    peak_hip_moment_con_Left = round(mocapDF["LHipMoment_x"][con_start:con_end].max(), 2)
    peak_hip_moment_landing_Left = round(mocapDF["LHipMoment_x"][landing_start:landing_end].max(), 2)
    peak_hip_moment_ED_Right = round(mocapDF["RHipMoment_x"][landing_start:landing_end].max(), 2)
    peak_hip_moment_con_Right = round(mocapDF["RHipMoment_x"][con_start:con_end].max(), 2)
    peak_hip_moment_landing_Right = round(mocapDF["RHipMoment_x"][landing_start:landing_end].max(), 2)

    # Peak Knee Extension Moment
    peak_knee_moment_ED_Left = round(mocapDF["LKneeMoment_x"][landing_start:landing_end].max(), 2)
    peak_knee_moment_con_Left = round(mocapDF["LKneeMoment_x"][con_start:con_end].max(), 2)
    peak_knee_moment_landing_Left = round(mocapDF["LKneeMoment_x"][landing_start:landing_end].max(), 2)
    peak_knee_moment_ED_Right = round(mocapDF["RKneeMoment_x"][landing_start:landing_end].max(), 2)
    peak_knee_moment_con_Right = round(mocapDF["RKneeMoment_x"][con_start:con_end].max(), 2)
    peak_knee_moment_landing_Right = round(mocapDF["RKneeMoment_x"][landing_start:landing_end].max(), 2)

    # Peak Knee Extension Moment
    peak_ankle_moment_ED_Left = round(mocapDF["LKneeMoment_x"][landing_start:landing_end].max(), 2)
    peak_ankle_moment_con_Left = round(mocapDF["LKneeMoment_x"][con_start:con_end].max(), 2)
    peak_ankle_moment_landing_Left = round(mocapDF["LKneeMoment_x"][landing_start:landing_end].max(), 2)
    peak_ankle_moment_ED_Right = round(mocapDF["RKneeMoment_x"][landing_start:landing_end].max(), 2)
    peak_ankle_moment_con_Right = round(mocapDF["RKneeMoment_x"][con_start:con_end].max(), 2)
    peak_ankle_moment_landing_Right = round(mocapDF["RKneeMoment_x"][landing_start:landing_end].max(), 2)

    # DESCRIPTIVE VARIABLES INTO A DICTIONARY
    # var_outputs = {
    #     "Jump Height (cm)": jump_height_cm,
    #     "Impulse ED Right (N·s)": impulse_ED_right,
    #     "Impulse ED Left (N·s)": impulse_ED_left,
    #     "Impulse Con Right (N·s)": impulse_con_right,
    #     "Impulse Con Left (N·s)": impulse_con_left,
    #     "Impulse Landing Right (N·s)": impulse_landing_right,
    #     "Impulse Landing Left (N·s)": impulse_landing_left,
    #     "Peak GRFv ED Right (N)": peak_GRFv_ED_Right,
    #     "Peak GRFv Con Right (N)": peak_GRFv_con_Right,
    #     "Peak GRFv Landing Right (N)": peak_GRFv_landing_Right,
    #     "Peak GRFv ED Left (N)": peak_GRFv_ED_Left,
    #     "Peak GRFv Con Left (N)": peak_GRFv_con_Left,
    #     "Peak GRFv Landing Left (N)": peak_GRFv_landing_Left,
    #     "Peak Hip Angle ED Left (°)": peak_hip_angle_ED_Left,
    #     "Peak Hip Angle Con Left (°)": peak_hip_angle_con_Left,
    #     "Peak Hip Angle Landing Left (°)": peak_hip_angle_landing_Left,
    #     "Peak Hip Angle ED Right (°)": peak_hip_angle_ED_Right,
    #     "Peak Hip Angle Con Right (°)": peak_hip_angle_con_Right,
    #     "Peak Hip Angle Landing Right (°)": peak_hip_angle_landing_Right,
    #     "Peak Knee Angle ED Left (°)": peak_knee_angle_ED_Left,
    #     "Peak Knee Angle Con Left (°)": peak_knee_angle_con_Left,
    #     "Peak Knee Angle Landing Left (°)": peak_knee_angle_landing_Left,
    #     "Peak Knee Angle ED Right (°)": peak_knee_angle_ED_Right,
    #     "Peak Knee Angle Con Right (°)": peak_knee_angle_con_Right,
    #     "Peak Knee Angle Landing Right (°)": peak_knee_angle_landing_Right,
    #     "Peak Ankle Angle ED Left (°)": peak_ankle_angle_ED_Left,
    #     "Peak Ankle Angle Con Left (°)": peak_ankle_angle_con_Left,
    #     "Peak Ankle Angle Landing Left (°)": peak_ankle_angle_landing_Left,
    #     "Peak Ankle Angle ED Right (°)": peak_ankle_angle_ED_Right,
    #     "Peak Ankle Angle Con Right (°)": peak_ankle_angle_con_Right,
    #     "Peak Ankle Angle Landing Right (°)": peak_ankle_angle_landing_Right,
    #     "Peak Hip Moment ED Left (Nm)": peak_hip_moment_ED_Left,
    #     "Peak Hip Moment Con Left (Nm)": peak_hip_moment_con_Left,
    #     "Peak Hip Moment Landing Left (Nm)": peak_hip_moment_landing_Left,
    #     "Peak Hip Moment ED Right (Nm)": peak_hip_moment_ED_Right,
    #     "Peak Hip Moment Con Right (Nm)": peak_hip_moment_con_Right,
    #     "Peak Hip Moment Landing Right (Nm)": peak_hip_moment_landing_Right,
    #     "Peak Knee Moment ED Left (Nm)": peak_knee_moment_ED_Left,
    #     "Peak Knee Moment Con Left (Nm)": peak_knee_moment_con_Left,
    #     "Peak Knee Moment Landing Left (Nm)": peak_knee_moment_landing_Left,
    #     "Peak Knee Moment ED Right (Nm)": peak_knee_moment_ED_Right,
    #     "Peak Knee Moment Con Right (Nm)": peak_knee_moment_con_Right,
    #     "Peak Knee Moment Landing Right (Nm)": peak_knee_moment_landing_Right,
    #     "Peak Ankle Moment ED Left (Nm)": peak_ankle_moment_ED_Left,
    #     "Peak Ankle Moment Con Left (Nm)": peak_ankle_moment_con_Left,
    #     "Peak Ankle Moment Landing Left (Nm)": peak_ankle_moment_landing_Left,
    #     "Peak Ankle Moment ED Right (Nm)": peak_ankle_moment_ED_Right,
    #     "Peak Ankle Moment Con Right (Nm)": peak_ankle_moment_con_Right,
    #     "Peak Ankle Moment Landing Right (Nm)": peak_ankle_moment_landing_Right
    # }

    # Asymmetry Indices
    # AI = ((Uninjured vs ACLR limb) / (whichever is biggest)) * 100
    # Positive number means uninjured performed BETTER than injured
    def AI_calc(right, left, inj_side):
        asymIndex = None

        if inj_side == "Right":
            asymIndex = ((left - right) / max(right, left)) * 100
        elif inj_side == "Left":
            asymIndex = ((right - left) / max(right, left)) * 100

        return asymIndex





    # -------------------------
    # Print Variables
    # -------------------------
    print("Trial number: ", trial_number)
    print(AI_calc(peak_knee_moment_con_Right, peak_knee_moment_con_Left, injured_side))
    print("")  # Spacer for readability in the terminal

    trial_number += 1  # iterate trial number
