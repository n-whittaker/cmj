import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from read_c3d import read_c3d

# Setting the directory to run through
root_dir = "/Users/nick/Desktop/Work Stuff/AC 115555 T2"


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


cmjs = getFiles()

for file in cmjs:
    data = read_c3d(file, read_mocap=True)
    mocap = data["MoCap"]
    grf = data["GRF"]
    info = data["Info"]

    grfDF = pd.DataFrame(grf)

    plt.plot(grfDF["Fz1"])
    print(grfDF["Fz1"])
    plt.show()
