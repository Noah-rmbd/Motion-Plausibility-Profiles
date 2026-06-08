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
preprocess_location_points = False #True
preprocess_trips = True

# Takes a directory as input and preprocess location points in it
def location_point_preprocessing(directory: str) -> [dict, dict, list, list, list] :
    location_points = {} # Store and cluster observations by taxon_family_name (used for ML algorithms)
    location_points_id = {} # Store observation ids that are located at the same index than the id in location_points (used for ML algorithms) 
    users_dictionary = {} # Temporarily store the number of observations posted by every user_id (then converted to users_list)
    users_list = [] # List of every user_id, associated with its number of observations (converted to .json for interactive visualisation)
    species_list = [] # List of every taxon_family_name, associated with the list of every observation of this species (converted to .json for interactive visualisation)
    observations = [] # List of unobscured observations
    obscured_observations = [] # List of obscured observations

    if os.path.exists("gowalla_frequent_poster.log"):
        with open("gowalla_frequent_poster.log", "r") as f:
            current_user = ""
            for line in f:
                line = line.strip()
                if line.startswith("===="): continue
                parts = line.split(",")
                if len(parts) == 2:
                    current_user = parts[0].strip()
                elif len(parts) >= 6:
                    entry_id = parts[0].strip()
                    lat = float(parts[1])
                    lon = float(parts[2])
                    date = parts[3].strip()
                    
                    observations.append({
                        "user_id": current_user,
                        "observation_id": entry_id,
                        "date": date,
                        "lat": lat,
                        "long": lon
                    })
                    
                    if current_user in users_dictionary:
                        users_dictionary[current_user] += 1
                    else:
                        users_dictionary[current_user] = 1
                    
                    taxon = "Gowalla"
                    if taxon in location_points:
                        location_points[taxon].append((entry_id, lat, lon))
                        location_points_id[taxon].append(entry_id)
                    else:
                        location_points[taxon] = [(entry_id, lat, lon)]
                        location_points_id[taxon] = [entry_id]

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

def normalize_trips_and_trajectories(trips, trajectories):
    # Normalize each parameter of a trip (speed, elapsed time, distance, acceleration, bearing change)
    trips = np.array(trips, dtype=object)
    trips_size = np.shape(trips)[0]

    speed_scaler = MinMaxScaler()
    elapsed_time_scaler = MinMaxScaler()
    distance_scaler = MinMaxScaler()
    acceleration_scaler = MinMaxScaler()
    # Bearing change is converted to Sine and Cosine (which are already correctly bounded [-1, 1])

    # Extract speeds to fit the scaler without taking >900 values into account
    speeds = trips[:,0].astype(float)
    valid_speeds = speeds[speeds <= 900.0]
    if len(valid_speeds) > 0:
        speed_scaler.fit(np.log1p(valid_speeds).reshape(-1,1))
    else:
        speed_scaler.fit(np.log1p(speeds).reshape(-1,1))

    # Create new trips array with 6 columns
    new_trips = np.zeros((trips_size, 6), dtype=float)
    new_trips[:,0] = speed_scaler.transform(np.log1p(speeds).reshape(-1,1)).reshape(trips_size,)
    new_trips[:,1] = elapsed_time_scaler.fit_transform(trips[:,1].reshape(-1,1)).reshape(trips_size,)
    new_trips[:,2] = distance_scaler.fit_transform(trips[:,2].reshape(-1,1)).reshape(trips_size,)
    new_trips[:,3] = acceleration_scaler.fit_transform(trips[:,3].reshape(-1,1)).reshape(trips_size,)
    new_trips[:,4] = (np.sin(trips[:,4].astype(float)) + 1.0) / 2.0
    new_trips[:,5] = (np.cos(trips[:,4].astype(float)) + 1.0) / 2.0
    
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
    
    return new_trips, trajectories, unnormalized_trajectories

def compute_unnormalized_statistics(trips, trajectories_id, title_prefix="", plot_path="vis/gowalla_unnormalized_distributions.png"):
    print(f"Computing statistics on unnormalized Gowalla trajectories... {title_prefix}")
    trips_arr = np.array(trips, dtype=float)
    
    # 1. Average number of transitions in trajectories
    transitions_per_trajectory = [len(t) - 1 for t in trajectories_id]
    avg_transitions = np.mean(transitions_per_trajectory) if transitions_per_trajectory else 0
    print(f"Average number of transitions in trajectories: {avg_transitions:.4f}")
    
    # 2. & 3. Average total distance and average speed per trajectory
    total_distances = []
    avg_speeds = []
    
    for traj_info in trajectories_id:
        trip_ids = traj_info[1:]
        traj_dist = np.sum(trips_arr[trip_ids, 2])
        # Average speed of the trajectory (mean of transition speeds inside the trajectory)
        traj_speed = np.mean(trips_arr[trip_ids, 0])
        total_distances.append(traj_dist)
        avg_speeds.append(traj_speed)
        
    # We display both Mean and Median to show the effect of outliers
    mean_total_distance = np.mean(total_distances) if total_distances else 0
    median_total_distance = np.median(total_distances) if total_distances else 0
    mean_avg_speed = np.mean(avg_speeds) if avg_speeds else 0
    median_avg_speed = np.median(avg_speeds) if avg_speeds else 0
    
    print(f"Trajectory Distance:")
    print(f"  - Mean total distance: {mean_total_distance:.4f} m")
    print(f"  - Median total distance: {median_total_distance:.4f} m")
    print(f"Trajectory Speed:")
    print(f"  - Mean average speed: {mean_avg_speed:.4f} km/h")
    print(f"  - Median average speed: {median_avg_speed:.4f} km/h")

    # 4. Extract active transitions to compute transition-level stats and plots
    valid_trip_ids = []
    for traj_info in trajectories_id:
        valid_trip_ids.extend(traj_info[1:])
    active_trips_arr = trips_arr[valid_trip_ids]

    # Maximum value in each category for active transitions
    # Columns of active_trips_arr: [0: speed, 1: elapsed_time, 2: distance, 3: acceleration, 4: bearing_change]
    features = ['Speed (km/h)', 'Elapsed Time (s)', 'Distance (m)', 'Acceleration (m/s²)', 'Bearing Change (rad)']
    print("\nMaximum values per transition category:")
    for i in range(5):
        max_val = np.max(np.abs(active_trips_arr[:, i])) if i == 4 else np.max(active_trips_arr[:, i])
        print(f"  - Max {features[i]}: {max_val:.4f}")

    # 5. Number of transitions above thresholds
    print("\nTransitions above logical thresholds:")
    total_transitions = len(active_trips_arr)
    
    # Threshold definitions (threshold, display_name, index, is_absolute)
    thresholds = [
        (130.0, "Speed > 130 km/h (highway speed limit)", 0, False),
        (900.0, "Speed > 900 km/h (airplane cruise speed)", 0, False),
        (3600.0, "Elapsed Time > 1 hour (3600s)", 1, False),
        (86400.0, "Elapsed Time > 24 hours (86400s)", 1, False),
        (10000.0, "Distance > 10 km (10000m)", 2, False),
        (100000.0, "Distance > 100 km (100000m)", 2, False),
        (10.0, "Acceleration absolute > 10 m/s² (~1G gravity)", 3, True),
        (50.0, "Acceleration absolute > 50 m/s²", 3, True)
    ]
    
    for item in thresholds:
        if item[3]:
            count = np.sum(np.abs(active_trips_arr[:, item[2]]) > item[0])
        else:
            count = np.sum(active_trips_arr[:, item[2]] > item[0])
        percentage = (count / total_transitions) * 100 if total_transitions > 0 else 0
        print(f"  - {item[1]}: {count} transitions ({percentage:.3f}%)")
    print("")

    # 6. Number of trajectories with at least one transition above thresholds
    print("Trajectories with at least one transition above thresholds:")
    total_trajectories = len(trajectories_id)
    
    for item in thresholds:
        count = 0
        for traj_info in trajectories_id:
            trip_ids = traj_info[1:]
            if len(trip_ids) == 0:
                continue
            values = trips_arr[trip_ids, item[2]]
            if item[3]:
                exceeds = np.any(np.abs(values) > item[0])
            else:
                exceeds = np.any(values > item[0])
            if exceeds:
                count += 1
        percentage = (count / total_trajectories) * 100 if total_trajectories > 0 else 0
        print(f"  - {item[1]}: {count} trajectories ({percentage:.3f}%)")
    print("")

    # Plot distributions
    print(f"Plotting distributions for unnormalized Gowalla transitions... {title_prefix}")
    os.makedirs('vis', exist_ok=True)
    
    features = ['Speed (km/h)', 'Elapsed Time (s)', 'Distance (m)', 'Acceleration (m/s²)', 'Bearing Change (rad)']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(5):
        data = active_trips_arr[:, i]
        
        # Speed, Elapsed Time, Distance, and Acceleration have heavy-tailed distributions (extreme outliers)
        # We must use logarithmic bins and axes so the plot doesn't just show a single bar at 0.
        if i < 4: 
            positive_data = data[data > 0]
            if len(positive_data) > 0:
                bins = np.logspace(np.log10(positive_data.min()), np.log10(positive_data.max()), 100)
                axes[i].hist(positive_data, bins=bins, edgecolor='black', alpha=0.7)
                axes[i].set_xscale('log')
                axes[i].set_yscale('log')
            else:
                axes[i].hist(data, bins=100, edgecolor='black', alpha=0.7)
        else:
            # Bearing change is bound between -pi and pi, linear scale is fine
            axes[i].hist(data, bins=100, edgecolor='black', alpha=0.7)
            
        axes[i].set_title(f'Distribution of {features[i]}')
        axes[i].set_xlabel(features[i])
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    # Remove the empty 6th subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Distributions saved to {plot_path}")

def trips_and_trajectory_preprocessing():
    # Computes valid transitions and trajectories between valid observations
    trip_nb = 0 # Variable to determine the id of a trip
    trips = [] # List of every trips between valid observations in the database
    trips_id = [] # List of trip_id that correspond to the trip located at the same index in trips[], store in addition the ids of its observations
    transitions_list = [] # List of transitions for the interactive visualization

    trajectory_nb = 0 # Variable to determine the id of a trajectory
    trajectory = [] # Store temporarily a new trajectory at each iteration
    trajectories = [] # List of every valid trajectory in the dataset
    trajectories_id = [] # List of trajectory_id that correspond to the trajectory located at the same index in trajectories[], store in addition the id of its inner trips
    trajectories_list = [] # List of trajectories for the interactive visualization

    trajectory_user = {} # Dictionary that takes the trajectory_id and returns its corresponding user

    # We use data from the gowalla_frequent_poster.log file
    if os.path.exists("gowalla_frequent_poster.log"):
        with open("gowalla_frequent_poster.log", "r") as f:
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
                                # A trajectory should contain at least 2 trips (3 valid observations)
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
                    # If the observation is unobscured
                    if prev_date == line_elements[3] and not obs_id.startswith("iN-o"):
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

                            # TrajectoryPoint: [speed, elapsed_time, distance, acceleration, bearing_change]
                            trip = [speed_kmh, elapsed_time, distance, acceleration, bearing_change]
                            trips.append(trip)
                            # Trajectory_id: [trip_id, observation1_id, observation2_id, date, user_id, transition_plausibility, plausibility_reason]
                            trips_id.append((trip_nb, prev_id, obs_id, prev_date, prev_user, transition_plausibility, plausibility_reason))

                            trajectory.append(trip)
                            trajectory_id.append(trip_nb)
                            trip_nb += 1
                            
                            # Update prev point for next iteration (pass current_bearing onwards)
                            prev_point = [obs_id, lat, lon, speed_ms, current_bearing]

            # Add the last sequence if any
            if trajectory != []:
                if len(trajectory) >= 2:
                    trajectories.append(trajectory)
                    trajectories_id.append(trajectory_id)

        f.close()

        # Before normalizing data for ML algorithms, we save the original values for the interactive visualisation
        for i in range(len(trips)):
            transition_id = trips_id[i]
            reason = transition_id[6] if len(transition_id) > 6 else None
            transitions_list.append({"transition_id": transition_id[0], "user_id": transition_id[4], "observation_id1": transition_id[1], "observation_id2": transition_id[2], "date": transition_id[3], "speed": trips[i][0], "elapsed_time": trips[i][1], "distance": trips[i][2], "acceleration": trips[i][3], "bearing_change": trips[i][4], "transition_plausibility": transition_id[5], "plausibility_reason": reason})
        for t in range(len(trajectories)):
            first_trip_id = trajectories_id[t][1]
            user_id = trips_id[first_trip_id][4]
            date = trips_id[first_trip_id][3]
            trajectories_list.append({"trajectory_id": trajectories_id[t][0], "user_id": user_id, "date": date, "transitions": trajectories_id[t][1:]})

        # Compute statistics on unnormalized real values
        compute_unnormalized_statistics(trips, trajectories_id)

        # Save unnormalized trips for later anomaly analysis
        with open('Preprocessed_Dataset/gowalla_unnormalized_trips.pkl', 'wb') as f:
            pickle.dump(trips, f)

        # Normalize every parameter in trips and trajectories
        trips, trajectories, unnormalized_trajectories = normalize_trips_and_trajectories(trips, trajectories)

        
    return trips, trips_id, transitions_list, trajectories, trajectories_id, trajectories_list, unnormalized_trajectories

def location_point_preprocessing_and_export():
    location_points, location_id, users_list, species_list, obscured_observations, observations = location_point_preprocessing("Observations/")
    # Export data made for ML algorithms in .pkl format
    with open('Preprocessed_Dataset/gowalla_location_points.pkl', 'wb') as f:
        pickle.dump(location_points, f)
    with open('Preprocessed_Dataset/gowalla_location_points_id.pkl', 'wb') as f:
        pickle.dump(location_id, f)

    # Sort users and keep top 1000 for the visualization to prevent browser memory crashes
    users_list = sorted(users_list, key=lambda x: x["nb_observations"], reverse=True)[:1000]
    top_user_ids = set([u["username"] for u in users_list])
    
    # Filter observations for the visualization
    vis_observations = [o for o in observations if o["user_id"] in top_user_ids]

    # Export data made for Interactive Visualization in .json format
    with open("Preprocessed_Dataset/gowalla_users_list.json", "w") as f:
        f.write(json.dumps(users_list, indent=4))
    with open("Preprocessed_Dataset/gowalla_observations.json", "w") as f:
        f.write(json.dumps(vis_observations, indent=4))

def trips_and_trajectory_preprocessing_and_export():
    trips, trips_id, transitions_list, trajectories, trajectories_id, trajectories_list, unnormalized_trajectories = trips_and_trajectory_preprocessing()
    
    # CREATE A FUNCTION TO AUTOMATE THE PLOTS CALCULATION
    '''plt.hist(trips[:, 1], bins=50, range=(0,1), edgecolor='black')
    plt.title("Distribution of Normalized Speed")
    plt.xlabel("Normalized Speed")
    plt.ylabel("Frequency")
    plt.show()'''
    # END OF THE FUNCTION

    # Convert data made for ML algorithms in .pkl format
    with open('Preprocessed_Dataset/gowalla_trips.pkl', 'wb') as f:
        pickle.dump(trips, f)
    with open('Preprocessed_Dataset/gowalla_trips_id.pkl', 'wb') as f:
        pickle.dump(trips_id, f)
    
    with open('Preprocessed_Dataset/gowalla_trajectories.pkl', 'wb') as f:
        pickle.dump(trajectories, f)
    with open('Preprocessed_Dataset/gowalla_trajectories_id.pkl', 'wb') as f:
        pickle.dump(trajectories_id, f)
    with open('Preprocessed_Dataset/gowalla_unnormalized_trajectories.pkl', 'wb') as f:
        pickle.dump(unnormalized_trajectories, f)

    # Filter visualization data to prevent browser out-of-memory errors
    try:
        with open("Preprocessed_Dataset/gowalla_users_list.json", "r") as f:
            top_users = json.load(f)
            top_user_ids = set([u["username"] for u in top_users])
        transitions_list = [t for t in transitions_list if t["user_id"] in top_user_ids]
        trajectories_list = [t for t in trajectories_list if t["user_id"] in top_user_ids]
    except Exception as e:
        print(f"Could not filter JSON data for visualization: {e}")

    # Export data made for Interactive Visualization in .json format
    with open("Preprocessed_Dataset/gowalla_transitions_list.json", "w") as f:
        f.write(json.dumps(transitions_list, indent=4))
    with open("Preprocessed_Dataset/gowalla_trajectories_list.json", "w") as f:
        f.write(json.dumps(trajectories_list, indent=4))

if __name__ == '__main__':
    if preprocess_location_points:
        location_point_preprocessing_and_export()
    if preprocess_trips:
        trips_and_trajectory_preprocessing_and_export()