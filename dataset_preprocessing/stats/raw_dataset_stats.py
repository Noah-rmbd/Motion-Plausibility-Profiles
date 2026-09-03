import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import pickle

# Select what metrics you want to compute
compute_most_active_users = True
compute_count_valid_sequences = True
compute_plot_speed_distribution = True
compute_iNaturalist_stats = False
compute_Gowalla_stats = True

#########################################################
# Return the list of the most active users in the dataset
#########################################################
def most_active_users():
    familyFiles = [ "droseraceae", "nepenthaceae", "sarraceniaceae", "roridulaceae", "byblidaceae", "lentibulariaceae", "cephalotaceae", "drosophyllaceae" ]
    usersList = []
    usersContributions = {}

    for file in familyFiles:
        path = "Observations/" + file + ".csv"
        df = pd.read_csv(path)
        for login, uid in zip(df["user_login"], df["user_id"]):
            user_key = f"{uid}-{login}"
            if user_key not in usersList:
                usersList.append(user_key)
                usersContributions[user_key] = 1
            else:
                usersContributions[user_key] += 1

    dsc = [(k, v) for k, v in sorted(usersContributions.items(), key=lambda item: item[1], reverse=True)]
    return dsc


#######################################################################
# Computes the number of valid sequences of observations in the dataset
#######################################################################
def count_valid_sequences():
    # A valid sequence is a sequence of unobscured observations that were posted by the same user, the same day.
    
    valid_sequences_count = 0
    sequence_length_distribution = {}

    current_day = None
    unobscured_obs_in_day = 0

    if os.path.exists("frequent-poster.log"):
        with open("frequent-poster.log", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("==="):
                    continue
                if "active from" in line:
                    if unobscured_obs_in_day > 2:
                        valid_sequences_count += 1
                        sequence_length_distribution[unobscured_obs_in_day] = sequence_length_distribution.get(unobscured_obs_in_day, 0) + 1
                    current_day = None
                    unobscured_obs_in_day = 0
                    continue
                
                parts = line.split(",")
                if len(parts) >= 4:
                    obs_id = parts[0].strip()
                    date = parts[3].strip()
                    
                    is_obscured = obs_id.startswith("iN-o")
                    
                    if date != current_day:
                        if unobscured_obs_in_day > 2:
                            valid_sequences_count += 1
                            sequence_length_distribution[unobscured_obs_in_day] = sequence_length_distribution.get(unobscured_obs_in_day, 0) + 1
                        current_day = date
                        unobscured_obs_in_day = 0
                    
                    if not is_obscured:
                        unobscured_obs_in_day += 1
        f.close()

        # Check the last day processed
        if unobscured_obs_in_day > 2:
            valid_sequences_count += 1
            sequence_length_distribution[unobscured_obs_in_day] = sequence_length_distribution.get(unobscured_obs_in_day, 0) + 1

        print(f"Number of valid sequences (days with > 2 unobscured observations): {valid_sequences_count}")
        print("Distribution of valid observations per valid sequence:")
        length_list = []
        for length in sorted(sequence_length_distribution.keys()):
            print(f"  {length} observations: {sequence_length_distribution[length]} sequences")
            length_list.append(sequence_length_distribution[length])

        sizes_arr = np.array(sorted(sequence_length_distribution.keys()))
        length_arr = np.array(length_list)
        total_valid_observations = np.sum(sizes_arr * length_arr)
        average_length = total_valid_observations / valid_sequences_count
        print(f"Average length of a valid sequence: {average_length:.2f} observations")

        # Create a plot to visualize valid observations sequences length
        plt.plot(sizes_arr, length_arr)
        plt.yscale('log')
        plt.xlabel('Number of valid observations')
        plt.ylabel('Number of sequences (log scale)')
        plt.title('Distribution of Valid Observation Sequences')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig('vis/0_sequences_length.png')
    else:
        print("frequent-poster.log not found.")


#######################################################################################
# Shows how the speeds and distances between points in a valid sequence are distributed
#######################################################################################
def plot_speed_distribution(file_path):
    speeds = []
    distances = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(", ")
            if len(parts) == 8 and parts[7].endswith("km/h"):
                try:
                    speed = float(parts[7].replace("km/h", ""))
                    distance = float(parts[6].replace("m", ""))
                    # Filter strictly positive values (powerlaw needs values > 0)
                    if speed > 0: 
                        speeds.append(speed)
                    if distance > 0:
                        distances.append(distance)
                except ValueError:
                    pass
    f.close()

    print(f"Found {len(speeds)} strictly positive speed measurements.")
    speeds_arr = np.array(speeds)
    clean_speeds = speeds_arr[(speeds_arr>0.3)]

    if len(speeds) > 0:
        os.makedirs('vis', exist_ok=True)
        
        # -- SPEED PLOT --
        plt.figure(figsize=(10, 6))
        plt.hist(clean_speeds, bins=100, color='b', alpha=0.7, density=True)
        plt.title('Distribution of Speeds')
        plt.xlabel('Speed (km/h)')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.savefig('vis/0_speed_distribution.png', bbox_inches='tight')
        plt.close()
        print("Saved plot to vis/0_speed_distribution.png")

        # -- DISTANCE PLOT --
        distances_arr = np.array(distances)
        clean_distances = distances_arr[distances_arr > 0]
        plt.figure(figsize=(10, 6))
        plt.hist(clean_distances, bins=100, color='b', alpha=0.7, density=True)
        plt.title('Distribution of Distances')
        plt.xlabel('Distance (m)')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.savefig('vis/0_distance_distribution.png', bbox_inches='tight')
        plt.close()
        print("Saved plot to vis/0_distance_distribution.png")

    else:
        print("No valid speed data found.")

if __name__ == "__main__":
    if compute_iNaturalist_stats:
        if compute_most_active_users:
            most_active_users()
        if compute_count_valid_sequences:
            count_valid_sequences()
    if compute_plot_speed_distribution:
        plot_speed_distribution("frequent-poster.log" if compute_iNaturalist_stats else "gowalla_frequent_poster.log")