import pandas as pd
import pickle
import numpy as np
import os
import json
import math
from sklearn.preprocessing import RobustScaler, MinMaxScaler
try:
    from Dataset_Preprocessing.utils import compute_bearing
except ModuleNotFoundError:
    from utils import compute_bearing
import matplotlib.pyplot as plt

# Select what shape of data you want to precompute
preprocess_location_points = True
preprocess_transitions = True

# List and return .csv files in a directory
def open_dataset(directory):
    list_files = []
    for element in os.scandir(directory):
        if element.name.endswith(".csv"):
            list_files.append(str(directory+"/"+element.name))
    if not list_files:
        print(f"No .csv file found in {directory} directory.") 
    return list_files

# Converts to a pandas dataframe iNaturalist files with required metrics
def open_file(path):
    df = pd.read_csv(path)
    if not {"id", "user_id", "user_login", "observed_on", "time_observed_at", "coordinates_obscured", "latitude", "longitude", "taxon_family_name"}.issubset(df.columns):
        print(f"{path} is not a valid file, it does not include the required columns")
        return None
    return df

# Takes a directory as input and preprocess location points in it
def location_point_preprocessing(directory: str) -> [dict, dict, list, list, list] :
    location_points = {} # Store and cluster observations by taxon_family_name (used for ML algorithms)
    location_points_id = {} # Store observation ids that are located at the same index than the id in location_points (used for ML algorithms) 
    users_dictionary = {} # Temporarily store the number of observations posted by every user_id (then converted to users_list)
    users_list = [] # List of every user_id, associated with its number of observations (converted to .json for interactive visualisation)
    species_list = [] # List of every taxon_family_name, associated with the list of every observation of this species (converted to .json for interactive visualisation)
    observations = [] # List of unobscured observations
    obscured_observations = [] # List of obscured observations

    list_files = open_dataset(directory)
    
    for file in list_files:
        df = open_file(file)
        if not df.empty:
            for uid, id, observed_on, time_observed_at, coordinates_obscured, latitude, longitude, taxon_family_name in zip(df["user_id"], df["id"], df["observed_on"], df["time_observed_at"], df["coordinates_obscured"], df["latitude"], df["longitude"], df["taxon_family_name"]):
                # Handle obscured coordinates
                is_obscured = False
                if pd.notna(coordinates_obscured):
                    if coordinates_obscured is True or str(coordinates_obscured).strip().lower() == "true":
                        is_obscured = True

                if is_obscured:
                    obscured_observations.append({
                        "user_id": uid,
                        "observation_id": id,
                        "date": observed_on,
                        "lat": None if pd.isna(latitude) else latitude,
                        "long": None if pd.isna(longitude) else longitude
                    })
                    continue
                
                observations.append({
                        "user_id": uid,
                        "observation_id": id,
                        "date": observed_on,
                        "lat": None if pd.isna(latitude) else latitude,
                        "long": None if pd.isna(longitude) else longitude
                })

                if uid in users_dictionary.keys():
                    users_dictionary[uid] += 1
                else:
                    users_dictionary[uid] = 1

                if taxon_family_name in location_points.keys():
                    location_points[taxon_family_name].append((id, latitude, longitude))
                    location_points_id[taxon_family_name].append(id) #Maybe would it be better to delete this dictionary later
                else:
                    location_points[taxon_family_name] = [(id, latitude, longitude)]
                    location_points_id[taxon_family_name] = [id]
    
    # Convert the dictionnaries into lists of dictionnaries for the .json conversion
    for user in users_dictionary.keys():
        users_list.append({"username": user, "nb_observations": users_dictionary[user]})
        
    # Sort the users list by number of unobscured observations descending
    users_list.sort(key=lambda x: x["nb_observations"], reverse=True)
    
    for species in location_points.keys():
        species_observations = []
        for observation in location_points[species]:
            species_observations.append({"id": observation[0], "lat": observation[1], "long": observation[2]})
        species_list.append({"taxon_family_name": species, "observations": species_observations})

    return location_points, location_points_id, users_list, species_list, obscured_observations, observations

def normalize_transitions_and_trajectories(transitions, trajectories):
    # Save unnormalized trajectories before scaling
    unnormalized_trajectories = [np.array(traj).copy() for traj in trajectories]
    
    # Normalize each parameter of a trip (speed, elapsed time, distance, acceleration, bearing change)
    transitions = np.array(transitions, dtype=object)
    transitions_size = np.shape(transitions)[0]

    speed_scaler = MinMaxScaler()
    elapsed_time_scaler = MinMaxScaler()
    distance_scaler = MinMaxScaler()
    acceleration_scaler = MinMaxScaler()
    # Bearing change is converted to Sine and Cosine (which are already correctly bounded [-1, 1])

    # Extract speeds to fit the scaler without taking >900 values into account
    speeds = transitions[:,0].astype(float)
    valid_speeds = speeds[speeds <= 900.0]
    if len(valid_speeds) > 0:
        speed_scaler.fit(np.log1p(valid_speeds).reshape(-1,1))
    else:
        speed_scaler.fit(np.log1p(speeds).reshape(-1,1))

    # Create new transitions array with 6 columns
    new_transitions = np.zeros((transitions_size, 6), dtype=float)
    new_transitions[:,0] = speed_scaler.transform(np.log1p(speeds).reshape(-1,1)).reshape(transitions_size,)
    new_transitions[:,1] = elapsed_time_scaler.fit_transform(transitions[:,1].reshape(-1,1)).reshape(transitions_size,)
    new_transitions[:,2] = distance_scaler.fit_transform(transitions[:,2].reshape(-1,1)).reshape(transitions_size,)
    new_transitions[:,3] = acceleration_scaler.fit_transform(transitions[:,3].reshape(-1,1)).reshape(transitions_size,)
    new_transitions[:,4] = (np.sin(transitions[:,4].astype(float)) + 1.0) / 2.0
    new_transitions[:,5] = (np.cos(transitions[:,4].astype(float)) + 1.0) / 2.0
    
    # Save unnormalized trajectories before scaling
    unnormalized_trajectories = [np.array(traj).copy() for traj in trajectories]

    # Normalize the values stored in each trajectory
    for i in range(len(trajectories)):
        trajectory = np.array(trajectories[i], dtype=object)
        trajectory_size = np.shape(trajectory)[0]
        traj_speeds = trajectory[:,0].astype(float)
        
        new_traj = np.zeros((trajectory_size, 6), dtype=float)
        new_traj[:,0] = speed_scaler.transform(np.log1p(traj_speeds).reshape(-1,1)).reshape(trajectory_size,)
        new_traj[:,1] = elapsed_time_scaler.transform(trajectory[:,1].reshape(-1,1)).reshape(trajectory_size,)
        new_traj[:,2] = distance_scaler.transform(trajectory[:,2].reshape(-1,1)).reshape(trajectory_size,)
        new_traj[:,3] = acceleration_scaler.transform(trajectory[:,3].reshape(-1,1)).reshape(trajectory_size,)
        new_traj[:,4] = (np.sin(trajectory[:,4].astype(float)) + 1.0) / 2.0
        new_traj[:,5] = (np.cos(trajectory[:,4].astype(float)) + 1.0) / 2.0
        trajectories[i] = new_traj
    
    return new_transitions, trajectories, unnormalized_trajectories

def transitions_and_trajectory_preprocessing():
    # Computes valid transitions and trajectories between valid observations
    transition_nb = 0 # Variable to determine the id of a transition
    transitions = [] # List of every transitions between valid observations in the database
    transitions_id = [] # List of transition_id that correspond to the transition located at the same index in transitions[], store in addition the ids of its observations
    transitions_list = [] # List of transitions for the interactive visualization

    trajectory_nb = 0 # Variable to determine the id of a trajectory
    trajectory = [] # Store temporarily a new trajectory at each iteration
    trajectories = [] # List of every valid trajectory in the dataset
    trajectories_id = [] # List of trajectory_id that correspond to the trajectory located at the same index in trajectories[], store in addition the id of its inner transitions
    trajectories_list = [] # List of trajectories for the interactive visualization

    trajectory_user = {} # Dictionary that takes the trajectory_id and returns its corresponding user

    # We use data from the frequent-poster.log file that was generated by _inaturalist.py script
    if os.path.exists("frequent-poster.log"):
        with open("frequent-poster.log", "r") as f:
            trajectory_id = [trajectory_nb]
            prev_date = ""
            prev_user = ""
            prev_point = None

            for line in f:
                line = line.strip()
                if line.startswith("===="):
                    continue
                
                line_elements = []
                line_elements = line.split(",")

                if len(line_elements) == 2:
                    prev_user = line_elements[0]
                
                # For each first observation in a day
                if len(line_elements) == 6:
                    # Ignore the obscured observations
                    if line_elements[0].startswith("iN-o"):
                        continue
                    else:
                        if line_elements[3] != prev_date:
                            if trajectory != []:
                                # A trajectory should contain at least 2 transitions (3 valid observations)
                                if len(trajectory) >= 2:
                                    trajectories.append(trajectory)
                                    trajectories_id.append(trajectory_id)
                                    trajectory_nb +=1
                                trajectory = []
                                trajectory_id = [trajectory_nb]
                            # We store informations so that we can identify a trajectory with the user id and the date    
                            trajectory_user[trajectory_nb] = (prev_user, prev_date)
                            prev_date = line_elements[3]
                        
                        obs_id = line_elements[0]
                        lat = float(line_elements[1])
                        lon = float(line_elements[2])
                        # Store as [ID, lat, lon, speed_ms, prev_bearing]
                        prev_point = [obs_id, lat, lon, 0.0, None]

                # For the other observations in a day        
                elif len(line_elements) == 8:
                    obs_id = line_elements[0]
                    # If the observation is unobscured and distance is valid
                    if prev_date == line_elements[3] and not obs_id.startswith("iN-o") and line_elements[6] != ' N/A':
                        lat = float(line_elements[1])
                        lon = float(line_elements[2])
                        elapsed_time = float(line_elements[5].replace("s", ""))
                        distance = float(line_elements[6].replace("m", ""))
                        if elapsed_time > 0:
                            speed_kmh = (distance / elapsed_time) * 3.6
                        else:
                            speed_kmh = float(line_elements[7].replace("km/h", ""))
                            # Handle sentinel value -1.0km/h for 0s elapsed time
                            if speed_kmh < 0:
                                speed_kmh = 0.0
                        
                        if prev_point is not None:
                            prev_id, prev_lat, prev_lon, prev_speed_ms, prev_bearing = prev_point
                            
                            speed_ms = speed_kmh / 3.6
                            acceleration = (speed_ms - prev_speed_ms) / elapsed_time if elapsed_time > 0 else 0.0
                            
                            current_bearing = compute_bearing(prev_lat, prev_lon, lat, lon)
                            if prev_bearing is None:
                                bearing_change = 0.0
                            else:
                                bearing_change = current_bearing - prev_bearing
                                # Normalize bearing change to be between -pi and pi
                                bearing_change = (bearing_change + math.pi) % (2 * math.pi) - math.pi
                            
                            plausibility_reason = None
                            # If speed is above 900 Km/h (maximum commercial flight speed), then this state transition is impossible
                            if speed_kmh > 900.0:
                                transition_plausibility = 0 # 0 = this transition is impossible
                                plausibility_reason = "Too high speed"
                            elif elapsed_time > 86400.0:
                                transition_plausibility = 0
                                plausibility_reason = "Elapsed time exceeds 24 hours"
                            elif distance > 10000000.0:
                                transition_plausibility = 0
                                plausibility_reason = "Distance exceeds 10,000 km"
                            elif abs(acceleration) > 50.0:
                                transition_plausibility = 0
                                plausibility_reason = "Acceleration exceeds 50 m/s^2"
                            elif (speed_kmh > 30.0 and elapsed_time <= 30) or (speed_kmh > 200.0 and elapsed_time <= 300) or (speed_kmh > 300.0 and elapsed_time <= 3600):
                                transition_plausibility = 0
                                plausibility_reason = "This speed cannot be reached with any vehicle during such a small amount of time"
                            else:
                                transition_plausibility = 1 # 1 = this transition is plausible

                            # TrajectoryPoint: [speed, elapsed_time (s), distance (m), acceleration (m/s^2), bearing_change (rad)]
                            transition = [speed_kmh, elapsed_time, distance, acceleration, bearing_change]
                            transitions.append(transition)
                            # Trajectory_id: [transition_id, observation1_id, observation2_id, date, user_id, transition_plausibility, plausibility_reason]
                            transitions_id.append((transition_nb, prev_id, obs_id, prev_date, prev_user, transition_plausibility, plausibility_reason))

                            trajectory.append(transition)
                            trajectory_id.append(transition_nb)
                            transition_nb += 1
                            
                            # Update prev point for next iteration (pass current_bearing onwards)
                            prev_point = [obs_id, lat, lon, speed_ms, current_bearing]

            # Add the last sequence if any
            if trajectory != []:
                if len(trajectory) >= 2:
                    trajectories.append(trajectory)
                    trajectories_id.append(trajectory_id)

        f.close()

        # Before normalizing data for ML algorithms, we save the original values for the interactive visualisation
        for i in range(len(transitions)):
            transition_id = transitions_id[i]
            reason = transition_id[6] if len(transition_id) > 6 else None
            transitions_list.append({"transition_id": transition_id[0], "user_id": transition_id[4], "observation_id1": transition_id[1], "observation_id2": transition_id[2], "date": transition_id[3], "speed": transitions[i][0], "elapsed_time": transitions[i][1], "distance": transitions[i][2], "acceleration": transitions[i][3], "bearing_change": transitions[i][4], "transition_plausibility": transition_id[5], "plausibility_reason": reason})
        for t in range(len(trajectories)):
            first_transition_id = trajectories_id[t][1]
            user_id = transitions_id[first_transition_id][4]
            date = transitions_id[first_transition_id][3]
            trajectories_list.append({"trajectory_id": trajectories_id[t][0], "user_id": user_id, "date": date, "transitions": trajectories_id[t][1:]})

        # Normalize every parameter in transitions and trajectories
        transitions, trajectories, unnormalized_trajectories = normalize_transitions_and_trajectories(transitions, trajectories)

        
    return transitions, transitions_id, transitions_list, trajectories, trajectories_id, trajectories_list, unnormalized_trajectories

def location_point_preprocessing_and_export():
    location_points, location_id, users_list, species_list, obscured_observations, observations = location_point_preprocessing("Observations/")
    # Export data made for ML algorithms in .pkl format
    with open('Preprocessed_Dataset/location_points.pkl', 'wb') as f:
        pickle.dump(location_points, f)
    with open('Preprocessed_Dataset/location_points_id.pkl', 'wb') as f:
        pickle.dump(location_id, f)

    # Export data made for Interactive Visualization in .json format
    with open("Preprocessed_Dataset/users_list.json", "w") as f:
        f.write(json.dumps(users_list, indent=4))
    with open("Preprocessed_Dataset/species_list.json", "w") as f:
        f.write(json.dumps(species_list, indent=4))
    with open("Preprocessed_Dataset/obscured_observations.json", "w") as f:
        f.write(json.dumps(obscured_observations, indent=4))
    with open("Preprocessed_Dataset/observations.json", "w") as f:
        f.write(json.dumps(observations, indent=4))

def transitions_and_trajectory_preprocessing_and_export():
    transitions, transitions_id, transitions_list, trajectories, trajectories_id, trajectories_list, unnormalized_trajectories = transitions_and_trajectory_preprocessing()
    
    # CREATE A FUNCTION TO AUTOMATE THE PLOTS CALCULATION
    plt.hist(transitions[:, 1], bins=50, range=(0,1), edgecolor='black')
    plt.title("Distribution of Normalized Speed")
    plt.xlabel("Normalized Speed")
    plt.ylabel("Frequency")
    plt.show()
    # END OF THE FUNCTION

    # Convert data made for ML algorithms in .pkl format
    with open('Preprocessed_Dataset/transitions.pkl', 'wb') as f:
        pickle.dump(transitions, f)
    with open('Preprocessed_Dataset/transitions_id.pkl', 'wb') as f:
        pickle.dump(transitions_id, f)
    
    with open('Preprocessed_Dataset/trajectories.pkl', 'wb') as f:
        pickle.dump(trajectories, f)
    with open('Preprocessed_Dataset/trajectories_id.pkl', 'wb') as f:
        pickle.dump(trajectories_id, f)
    with open('Preprocessed_Dataset/unnormalized_trajectories.pkl', 'wb') as f:
        pickle.dump(unnormalized_trajectories, f)

    # Convert data made for Interactive Visualization in .json format
    with open("Preprocessed_Dataset/transitions_list.json", "w") as f:
        f.write(json.dumps(transitions_list, indent=4))
    with open("Preprocessed_Dataset/trajectories_list.json", "w") as f:
        f.write(json.dumps(trajectories_list, indent=4))

if __name__ == '__main__':
    if preprocess_location_points:
        location_point_preprocessing_and_export()
    if preprocess_transitions:
        transitions_and_trajectory_preprocessing_and_export()