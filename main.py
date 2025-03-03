import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from read_c3d import read_c3d

root_dir = "/Users/nick/Documents/University/Research Project/HT"


# Setting the directory to run through
# patient_dir = "/Users/nick/Documents/University/Research Project/HT/AB 127331 Retest"


def calcPatient(patient_dir):
    # print(os.listdir(patient_dir)) # to see which file is being processed currently

    # Checking ENF file for injured side
    enf_file_path = None
    for filename in os.listdir(patient_dir):
        if filename.endswith(".enf"):
            enf_file_path = os.path.join(patient_dir, filename)
            break

    # Check if an ENF file was found and process it
    if enf_file_path:
        with open(enf_file_path, "r") as file:
            for line in file:
                if line.startswith("INJURY="):
                    injured_side = line.split("=")[1].strip()
                    break
    else:
        print("No ENF file found in the directory.")

    # Get CMJ c3d files
    def getFiles():
        files = []
        # Walk through directory, check for any folder with "New Session" in its name
        for (dirpath, dirnames, filenames) in os.walk(patient_dir):
            for d in dirnames:
                if "New Session" in d:  # This checks if the substring is in the directory name
                    new_session_path = os.path.join(str(dirpath), str(d))
                    for session_file in os.listdir(new_session_path):
                        session_file = str(session_file)  # Ensure session_file is a string
                        if session_file.endswith(".c3d") and "CMJ" in session_file and "SL" not in session_file:
                            file_path = os.path.join(new_session_path, session_file)
                            files.append(file_path)

        return files

    cmjs = getFiles()  # All the cmj c3d's in the directory provided

    trial_results = []  # List to store results for all trials for this patient
    trial_number = 1

    # FOR EACH CMJ
    for file in cmjs:
        patient_name = os.path.basename(os.path.dirname(os.path.dirname(file)))

        data = read_c3d(file, read_mocap=True)  # Read the c3d data...

        mocap = data["MoCap"]  # And separate it into the motion capture data..
        grf = data["GRF"]  # The GRF data
        info = data["Info"]  # And general info

        mocapDF = pd.DataFrame(mocap)
        grfDF = pd.DataFrame(grf)  # Create a dataFrame from the GRF data
        infoDF = pd.DataFrame(info)

        # Sum from the two forces to get total vertical GRF
        grfTotal = grfDF["Fz1"] + grfDF["Fz2"]

        com_vel_z = None
        if "COMVelocity_z" not in mocapDF.columns:  # Some data is missing Com velocity, this is to check
            print("Column 'COMVelocity_z' not found. Available columns:", mocapDF.columns.tolist())
            # You may choose to handle the error, e.g., skip this file or assign a default value.
        else:
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

        # velocity_graph_color = "black"
        # phase_color = "black"
        # end_time = len(grfTotal) / sampling_rate
        #
        # fig, ax1 = plt.subplots(figsize=(10, 6))
        #
        # # FP1 = Right, FP2 = Left
        #
        # ax1.plot(grfDF["Fz1"], "red", label="Right")
        # ax1.plot(grfDF["Fz2"], "blue", label="Left")
        # ax1.set_xlabel("Time (s)")
        # ax1.set_ylabel("Force (N)")
        #
        # # Plot COM Velocity on right axis
        # ax2 = ax1.twinx()
        # ax2.plot(com_vel_z, velocity_graph_color, linestyle="--", label="Velocity (m/s)")
        # ax2.set_ylabel("Velocity (m/s)", color=velocity_graph_color)
        # ax2.tick_params(axis="y", labelcolor=velocity_graph_color)
        # ax2.set_ylim(-3, 8)
        #
        # # Spans
        # ax2.axvspan(ED_start, ED_end, color="red", alpha=0.1, label="Eccentric Deceleration")  # Eccentric Deceleration
        # ax2.axvspan(ED_end, con_end, color="blue", alpha=0.1, label="Concentric")  # Concentric
        # ax2.axvspan(landing_start, landing_end, color="green", alpha=0.1, label="Landing")  # Concentric
        #
        # # Set the title (using the file name for reference)
        # ax1.set_title(f"CMJ Phases: {os.path.basename(file)}")
        #
        # # Combine legends from both axes
        # handles1, labels1 = ax1.get_legend_handles_labels()
        # handles2, labels2 = ax2.get_legend_handles_labels()
        # ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
        #
        # plt.tight_layout()
        # plt.show()

        # -------------------------
        # Variable calculations
        # -------------------------

        # Jump height sing the impulse-momentum equation
        g = 9.81  # Gravity
        v_takeoff = com_vel_z.loc[con_end]  # Extract velocity at takeoff
        jump_height_cm = (v_takeoff ** 2) / (2 * g) * 100

        # Impulse using trapezoid rule
        dt = 1.0 / sampling_rate  # Time step based on  sampling rate
        impulse_ED_Right = np.trapz(grfDF["Fz1"][ED_start:ED_end], dx=dt)
        impulse_ED_Left = np.trapz(grfDF["Fz2"][ED_start:ED_end], dx=dt)
        impulse_con_Right = np.trapz(grfDF["Fz1"][ED_end:con_end], dx=dt)
        impulse_con_Left = np.trapz(grfDF["Fz2"][ED_end:con_end], dx=dt)
        impulse_landing_Right = np.trapz(grfDF["Fz1"][landing_start:landing_end], dx=dt)
        impulse_landing_Left = np.trapz(grfDF["Fz2"][landing_start:landing_end], dx=dt)

        # Peak GRFv
        peak_GRFv_ED_Right = grf["Fz1"][ED_start:ED_end].max()
        peak_GRFv_con_Right = grf["Fz1"][con_start:con_end].max()
        peak_GRFv_landing_Right = grf["Fz1"][landing_start:landing_end].max()
        peak_GRFv_ED_Left = grf["Fz2"][ED_start:ED_end].max()
        peak_GRFv_con_Left = grf["Fz2"][con_start:con_end].max()
        peak_GRFv_landing_Left = grf["Fz2"][landing_start:landing_end].max()

        # Peak Hip Flexion Angles
        peak_hip_angle_ED_Left = mocapDF["LHipAngles_x"][landing_start:landing_end].max()
        peak_hip_angle_con_Left = mocapDF["LHipAngles_x"][con_start:con_end].max()
        peak_hip_angle_landing_Left = mocapDF["LHipAngles_x"][landing_start:landing_end].max()
        peak_hip_angle_ED_Right = mocapDF["RHipAngles_x"][landing_start:landing_end].max()
        peak_hip_angle_con_Right = mocapDF["RHipAngles_x"][con_start:con_end].max()
        peak_hip_angle_landing_Right = mocapDF["RHipAngles_x"][landing_start:landing_end].max()

        # Peak Knee Flexion Angles
        peak_knee_angle_ED_Left = mocapDF["LKneeAngles_x"][landing_start:landing_end].max()
        peak_knee_angle_con_Left = mocapDF["LKneeAngles_x"][con_start:con_end].max()
        peak_knee_angle_landing_Left = mocapDF["LKneeAngles_x"][landing_start:landing_end].max()
        peak_knee_angle_ED_Right = mocapDF["RKneeAngles_x"][landing_start:landing_end].max()
        peak_knee_angle_con_Right = mocapDF["RKneeAngles_x"][con_start:con_end].max()
        peak_knee_angle_landing_Right = mocapDF["RKneeAngles_x"][landing_start:landing_end].max()

        # Peak Ankle Flexion Angles
        peak_ankle_angle_ED_Left = mocapDF["LKneeAngles_x"][landing_start:landing_end].max()
        peak_ankle_angle_con_Left = mocapDF["LKneeAngles_x"][con_start:con_end].max()
        peak_ankle_angle_landing_Left = mocapDF["LKneeAngles_x"][landing_start:landing_end].max()
        peak_ankle_angle_ED_Right = mocapDF["RKneeAngles_x"][landing_start:landing_end].max()
        peak_ankle_angle_con_Right = mocapDF["RKneeAngles_x"][con_start:con_end].max()
        peak_ankle_angle_landing_Right = mocapDF["RKneeAngles_x"][landing_start:landing_end].max()

        # Peak Hip Extension Moment
        peak_hip_moment_ED_Left = mocapDF["LHipMoment_x"][landing_start:landing_end].max()
        peak_hip_moment_con_Left = mocapDF["LHipMoment_x"][con_start:con_end].max()
        peak_hip_moment_landing_Left = mocapDF["LHipMoment_x"][landing_start:landing_end].max()
        peak_hip_moment_ED_Right = mocapDF["RHipMoment_x"][landing_start:landing_end].max()
        peak_hip_moment_con_Right = mocapDF["RHipMoment_x"][con_start:con_end].max()
        peak_hip_moment_landing_Right = mocapDF["RHipMoment_x"][landing_start:landing_end].max()

        # Peak Knee Extension Moment
        peak_knee_moment_ED_Left = mocapDF["LKneeMoment_x"][landing_start:landing_end].max()
        peak_knee_moment_con_Left = mocapDF["LKneeMoment_x"][con_start:con_end].max()
        peak_knee_moment_landing_Left = mocapDF["LKneeMoment_x"][landing_start:landing_end].max()
        peak_knee_moment_ED_Right = mocapDF["RKneeMoment_x"][landing_start:landing_end].max()
        peak_knee_moment_con_Right = mocapDF["RKneeMoment_x"][con_start:con_end].max()
        peak_knee_moment_landing_Right = mocapDF["RKneeMoment_x"][landing_start:landing_end].max()

        # Peak Knee Extension Moment
        peak_ankle_moment_ED_Left = mocapDF["LKneeMoment_x"][landing_start:landing_end].max()
        peak_ankle_moment_con_Left = mocapDF["LKneeMoment_x"][con_start:con_end].max()
        peak_ankle_moment_landing_Left = mocapDF["LKneeMoment_x"][landing_start:landing_end].max()
        peak_ankle_moment_ED_Right = mocapDF["RKneeMoment_x"][landing_start:landing_end].max()
        peak_ankle_moment_con_Right = mocapDF["RKneeMoment_x"][con_start:con_end].max()
        peak_ankle_moment_landing_Right = mocapDF["RKneeMoment_x"][landing_start:landing_end].max()

        # DESCRIPTIVE VARIABLES INTO A DICTIONARY
        var_outputs = {
            # "Patient": patient_name, # ONLY ADD WHEN GENERATING DESCRIPTIVES!!!
            # "Trial Number": trial_number,
            "Jump Height (cm)": jump_height_cm,
            "Impulse ED Right (N·s)": impulse_ED_Right,
            "Impulse ED Left (N·s)": impulse_ED_Left,
            "Impulse Con Right (N·s)": impulse_con_Right,
            "Impulse Con Left (N·s)": impulse_con_Left,
            "Impulse Landing Right (N·s)": impulse_landing_Right,
            "Impulse Landing Left (N·s)": impulse_landing_Left,
            "Peak GRFv ED Right (N)": peak_GRFv_ED_Right,
            "Peak GRFv Con Right (N)": peak_GRFv_con_Right,
            "Peak GRFv Landing Right (N)": peak_GRFv_landing_Right,
            "Peak GRFv ED Left (N)": peak_GRFv_ED_Left,
            "Peak GRFv Con Left (N)": peak_GRFv_con_Left,
            "Peak GRFv Landing Left (N)": peak_GRFv_landing_Left,
            "Peak Hip Angle ED Left (°)": peak_hip_angle_ED_Left,
            "Peak Hip Angle Con Left (°)": peak_hip_angle_con_Left,
            "Peak Hip Angle Landing Left (°)": peak_hip_angle_landing_Left,
            "Peak Hip Angle ED Right (°)": peak_hip_angle_ED_Right,
            "Peak Hip Angle Con Right (°)": peak_hip_angle_con_Right,
            "Peak Hip Angle Landing Right (°)": peak_hip_angle_landing_Right,
            "Peak Knee Angle ED Left (°)": peak_knee_angle_ED_Left,
            "Peak Knee Angle Con Left (°)": peak_knee_angle_con_Left,
            "Peak Knee Angle Landing Left (°)": peak_knee_angle_landing_Left,
            "Peak Knee Angle ED Right (°)": peak_knee_angle_ED_Right,
            "Peak Knee Angle Con Right (°)": peak_knee_angle_con_Right,
            "Peak Knee Angle Landing Right (°)": peak_knee_angle_landing_Right,
            "Peak Ankle Angle ED Left (°)": peak_ankle_angle_ED_Left,
            "Peak Ankle Angle Con Left (°)": peak_ankle_angle_con_Left,
            "Peak Ankle Angle Landing Left (°)": peak_ankle_angle_landing_Left,
            "Peak Ankle Angle ED Right (°)": peak_ankle_angle_ED_Right,
            "Peak Ankle Angle Con Right (°)": peak_ankle_angle_con_Right,
            "Peak Ankle Angle Landing Right (°)": peak_ankle_angle_landing_Right,
            "Peak Hip Moment ED Left (Nm)": peak_hip_moment_ED_Left,
            "Peak Hip Moment Con Left (Nm)": peak_hip_moment_con_Left,
            "Peak Hip Moment Landing Left (Nm)": peak_hip_moment_landing_Left,
            "Peak Hip Moment ED Right (Nm)": peak_hip_moment_ED_Right,
            "Peak Hip Moment Con Right (Nm)": peak_hip_moment_con_Right,
            "Peak Hip Moment Landing Right (Nm)": peak_hip_moment_landing_Right,
            "Peak Knee Moment ED Left (Nm)": peak_knee_moment_ED_Left,
            "Peak Knee Moment Con Left (Nm)": peak_knee_moment_con_Left,
            "Peak Knee Moment Landing Left (Nm)": peak_knee_moment_landing_Left,
            "Peak Knee Moment ED Right (Nm)": peak_knee_moment_ED_Right,
            "Peak Knee Moment Con Right (Nm)": peak_knee_moment_con_Right,
            "Peak Knee Moment Landing Right (Nm)": peak_knee_moment_landing_Right,
            "Peak Ankle Moment ED Left (Nm)": peak_ankle_moment_ED_Left,
            "Peak Ankle Moment Con Left (Nm)": peak_ankle_moment_con_Left,
            "Peak Ankle Moment Landing Left (Nm)": peak_ankle_moment_landing_Left,
            "Peak Ankle Moment ED Right (Nm)": peak_ankle_moment_ED_Right,
            "Peak Ankle Moment Con Right (Nm)": peak_ankle_moment_con_Right,
            "Peak Ankle Moment Landing Right (Nm)": peak_ankle_moment_landing_Right
        }

        # -------------------------
        # Asymmetry calculations
        # AI = ((Uninjured vs ACLR limb) / (whichever is biggest)) * 100
        # Positive number means uninjured performed BETTER than injured
        # -------------------------

        # Calculate asymmetry
        def AI_calc(right, left, inj_side):
            asymIndex = None

            if inj_side.lower().startswith("r"):  # Input can vary e.g "Right" or "R"
                asymIndex = ((left - right) / max(right, left)) * 100
            elif inj_side.lower().startswith("l"):
                asymIndex = ((right - left) / max(right, left)) * 100
            else:
                raise ValueError("Injured side not recognised: ", patient_dir)
            return asymIndex

        var_asymmetries = {}

        # Making the new dictionary
        for key in var_outputs:
            if "Right" in key:
                # Create the corresponding left key by replacing "Right" with "Left"
                right_key = key
                left_key = key.replace("Right", "Left")
                # Remove the units and side indicator from the key to create a base label
                # Example: "Impulse ED Right (N·s)" -> "Impulse ED"
                base_key = right_key.split("Right")[0].strip()  # everything before "Right"
                base_key = base_key.split(" (")[0].strip()  # remove units if present

                # Retrieve the values for right and left
                right_value = var_outputs[right_key]
                left_value = var_outputs[left_key]

                # Calculate the asymmetry index using your injured side (inj_side)
                ai_value = AI_calc(right_value, left_value, injured_side)

                # Store it in the new dictionary with the base key
                var_asymmetries[base_key] = ai_value

        # Absolute Asymmetry Indices
        absolute_asymmetries = var_asymmetries.copy()
        # Iterate and update each value with its AAI = should just be the same value but with direction taken away.
        for key, value in absolute_asymmetries.items():
            # Square root of the squared AI
            absolute_asymmetries[key] = abs(value)

        # -------------------------
        # Print Variables
        # -------------------------
        # print("Trial number: ", trial_number)
        # print("Base variables:")
        # print(var_outputs)
        # print("Asymmetries:")
        # print(var_asymmetries)
        # print("Absolute asymmetries:")
        # print(absolute_asymmetries)
        # print("")  # Spacer for readability in the terminal

        trial_results.append(absolute_asymmetries)
        trial_number += 1

    return trial_results


# Now, iterate through each patient folder in the root directory
all_results = []

# Assume each subfolder in root_dir is a patient folder
for folder in os.listdir(root_dir):
    patient_path = os.path.join(root_dir, folder)
    if os.path.isdir(patient_path):
        patient_data = calcPatient(patient_path)
        if patient_data:  # Only add if data was returned
            all_results.extend(patient_data)

# Convert results to a DataFrame and export to Excel or CSV
df = pd.DataFrame(all_results)
print(df)
excel_output_path = "/Users/nick/Documents/University/Research Project/DATA OUTPUT SPREADSHEETS/HT/HT_AAI.xlsx"
df.to_excel(excel_output_path, index=False)
# Alternatively:
# df.to_csv("AllPatients.csv", index=False)

# To use if I just want to look at one person
# calcPatient("/Users/nick/Documents/University/Research Project/HT/CMcN 162530 Retest")
