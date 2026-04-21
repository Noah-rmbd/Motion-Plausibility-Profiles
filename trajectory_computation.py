import pickle
import numpy as np
import os

if os.path.exists("frequent-poster.log"):
    with open("frequent-poster.log", "r") as f:
        trajectories = []
        trajectory = []
        line_elements = []
        prev_date = ""

        for line in f:
            line = line.strip()
            if line.startswith("===="):
                continue
            line_elements = []
            line_elements = line.split(",")
            
            # For each first observation in a day
            if len(line_elements)==6:
                # Ignore the obscured observations
                if (line_elements[0].startswith("iN-o")):
                    continue
                else:
                    if (line_elements[3]!=prev_date):
                        if(trajectory!=[]):
                            if(len(trajectory)>2):
                                trajectories.append(trajectory)
                            trajectory=[]
                            trajectory_point=(float(line_elements[1]), float(line_elements[2]), line_elements[4], 0.0)
                            trajectory.append(trajectory_point)
                        prev_date=line_elements[3]
            # For the other observations in a day        
            elif len(line_elements)==8:
                # If the observation is unobscured
                if(prev_date==line_elements[3] and not line_elements[0].startswith("iN-o")):
                    trajectory_point = (float(line_elements[1]), float(line_elements[2]), line_elements[4], float(line_elements[7].replace("km/h", "")))
                    trajectory.append(trajectory_point)
    f.close()

with open('trajectories.pkl', 'wb') as f:
    pickle.dump(trajectories, f)
f.close()
    
with open('trajectories.pkl', 'rb') as f:
    trajectories_load = pickle.load(f)
f.close()