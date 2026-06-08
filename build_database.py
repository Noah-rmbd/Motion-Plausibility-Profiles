import sqlite3
import json
import pickle
import os
import glob
import gc
import argparse

DB_PATH = "plausibility.db"

def stream_json_array(file_path, chunk_size=65536):
    decoder = json.JSONDecoder()
    buffer = ""
    with open(file_path, 'r', encoding='utf-8') as f:
        # Find the starting '['
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                return
            buffer += chunk
            first_bracket = buffer.find('[')
            if first_bracket != -1:
                buffer = buffer[first_bracket + 1:]
                break
        
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(']'):
                break
            if buffer.startswith(','):
                buffer = buffer[1:].lstrip()
            
            try:
                obj, index = decoder.raw_decode(buffer)
                yield obj
                buffer = buffer[index:]
            except json.JSONDecodeError:
                chunk = f.read(chunk_size)
                if not chunk:
                    buffer = buffer.strip()
                    if buffer == ']' or not buffer:
                        break
                    raise ValueError(f"Incomplete JSON or decode error in {file_path}")
                buffer += chunk

def batch_generator(generator, batch_size=10000):
    batch = []
    for item in generator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def init_db(recreate=False):
    if recreate and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = OFF;")
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA cache_size = 2000;")
    
    if recreate:
        c = conn.cursor()
        c.executescript("""
            CREATE TABLE users (
                dataset TEXT,
                username TEXT,
                nb_observations INTEGER,
                PRIMARY KEY (dataset, username)
            );

            CREATE TABLE observations (
                observation_id TEXT PRIMARY KEY,
                dataset TEXT,
                user_id TEXT,
                date TEXT,
                lat REAL,
                lon REAL
            );

            CREATE TABLE transitions (
                transition_id INTEGER,
                dataset TEXT,
                user_id TEXT,
                observation_id1 TEXT,
                observation_id2 TEXT,
                date TEXT,
                speed REAL,
                elapsed_time REAL,
                distance REAL,
                acceleration REAL,
                bearing_change REAL,
                transition_plausibility REAL,
                plausibility_reason TEXT,
                PRIMARY KEY (dataset, transition_id)
            );

            CREATE TABLE trajectories (
                trajectory_id INTEGER,
                dataset TEXT,
                user_id TEXT,
                date TEXT,
                transitions_json TEXT,
                PRIMARY KEY (dataset, user_id, trajectory_id)
            );

            CREATE TABLE models (
                model_id TEXT PRIMARY KEY,
                name TEXT,
                dataset TEXT,
                percentile INTEGER,
                threshold REAL
            );

            CREATE TABLE model_trajectory_scores (
                model_id TEXT,
                trajectory_id INTEGER,
                is_unplausible BOOLEAN,
                reconstruction_error REAL,
                least_plausible_transition_id INTEGER,
                plausibility_reason TEXT,
                PRIMARY KEY (model_id, trajectory_id)
            );

            CREATE TABLE model_transition_scores (
                model_id TEXT,
                transition_id INTEGER,
                mse REAL,
                feature_error_speed REAL,
                feature_error_time REAL,
                feature_error_dist REAL,
                feature_error_accel REAL,
                feature_error_bear_sin REAL,
                feature_error_bear_cos REAL,
                PRIMARY KEY (model_id, transition_id)
            );
            
            CREATE INDEX idx_obs_user ON observations(dataset, user_id);
            CREATE INDEX idx_trans_user ON transitions(dataset, user_id);
            CREATE INDEX idx_traj_user ON trajectories(dataset, user_id);
        """)
        conn.commit()
    return conn

def load_dataset_base(conn, dataset_name, prefix=""):
    print(f"Loading {dataset_name} base data...")
    c = conn.cursor()
    
    # 1. Users
    users_file = f"Preprocessed_Dataset/{prefix}users_list.json"
    if os.path.exists(users_file):
        records_gen = (
            (dataset_name, str(u['username']), u['nb_observations'])
            for u in stream_json_array(users_file)
        )
        for batch in batch_generator(records_gen, 50000):
            c.executemany("INSERT INTO users VALUES (?, ?, ?)", batch)
            conn.commit()
            
    # 2. Observations
    obs_file = f"Preprocessed_Dataset/{prefix}observations.json"
    if os.path.exists(obs_file):
        records_gen = (
            (str(o.get('observation_id', o.get('id'))), dataset_name, str(o.get('user_id', '')), o.get('date', ''), float(o['lat']), float(o['long']))
            for o in stream_json_array(obs_file)
        )
        for batch in batch_generator(records_gen, 50000):
            c.executemany("INSERT OR IGNORE INTO observations VALUES (?, ?, ?, ?, ?, ?)", batch)
            conn.commit()

    # 3. Transitions
    trans_file = f"Preprocessed_Dataset/{prefix}transitions_list.json"
    if os.path.exists(trans_file):
        records_gen = (
            (
                int(t['transition_id']), dataset_name, str(t['user_id']), str(t['observation_id1']), str(t['observation_id2']),
                t['date'], float(t['speed']), float(t['elapsed_time']), float(t['distance']),
                float(t.get('acceleration', 0.0)), float(t.get('bearing_change', 0.0)),
                float(t.get('transition_plausibility', 1.0)), t.get('plausibility_reason')
            )
            for t in stream_json_array(trans_file)
        )
        for batch in batch_generator(records_gen, 50000):
            c.executemany("INSERT INTO transitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
            conn.commit()

    # 4. Trajectories
    traj_file = f"Preprocessed_Dataset/{prefix}trajectories_list.json"
    if os.path.exists(traj_file):
        records_gen = (
            (int(t['trajectory_id']), dataset_name, str(t['user_id']), t['date'], json.dumps(t['transitions']))
            for t in stream_json_array(traj_file)
        )
        for batch in batch_generator(records_gen, 50000):
            c.executemany("INSERT INTO trajectories VALUES (?, ?, ?, ?, ?)", batch)
            conn.commit()
            
    conn.commit()
    gc.collect()

def load_single_model_score(conn, file_path):
    print(f"Loading model scores from {file_path}...")
    c = conn.cursor()
    
    filename = os.path.basename(file_path)
    # Format: anomaly_scores_{dataset}_{model_name}_{percentile}.jsonl
    # Or older format: anomaly_scores_{dataset}_{model_name}.jsonl
    
    # Remove 'anomaly_scores_' and '.jsonl' or '.pkl'
    name_parts = filename.replace("anomaly_scores_", "").replace(".jsonl", "").replace(".pkl", "").split("_")
    
    dataset = name_parts[0] # gowalla, inat, synthetic, etc.
    
    # Try to parse percentile from the end
    try:
        val = int(name_parts[-1])
        if 0 < val < 100:
            percentile = val
            model_name = "_".join(name_parts[1:-1])
        else:
            percentile = 0
            model_name = "_".join(name_parts[1:])
    except ValueError:
        percentile = 0
        model_name = "_".join(name_parts[1:])
        
    model_id = filename.replace(".jsonl", "").replace(".pkl", "")
    
    print(f"  -> Processing {filename} (Dataset: {dataset}, Model: {model_name}, Percentile: {percentile})")
    
    threshold = 0.0
    c.execute("INSERT OR REPLACE INTO models VALUES (?, ?, ?, ?, ?)", (model_id, model_name, dataset, percentile, threshold))
    
    traj_batch = []
    trans_batch = []
    uncommitted_transitions = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            obj = json.loads(line)
            if 'metadata' in obj:
                metadata = obj['metadata']
                threshold = metadata.get('threshold', threshold)
                if percentile == 0:
                    percentile = metadata.get('percentile', percentile)
                c.execute("INSERT OR REPLACE INTO models VALUES (?, ?, ?, ?, ?)", (model_id, model_name, dataset, percentile, threshold))
                continue
                
            t_id_val = obj['trajectory_id']
            t_id = int(t_id_val) if isinstance(t_id_val, int) else int(t_id_val[0])
            is_unplausible = obj['is_unplausible']
            mse = float(obj['reconstruction_error'])
            least_plausible = obj.get('least_plausible_transition_id')
            least_plausible = int(least_plausible) if least_plausible is not None else None
            reason = obj.get('plausibility_reason')
            
            traj_batch.append((model_id, t_id, is_unplausible, mse, least_plausible, reason))
            
            for trans_id_str, trans_mse in obj['transition_errors'].items():
                tid = int(trans_id_str)
                features = obj['transition_feature_errors'].get(trans_id_str, [0.0]*6)
                trans_batch.append((
                    model_id, tid, float(trans_mse),
                    float(features[0]), float(features[1]), float(features[2]), 
                    float(features[3]), float(features[4]), float(features[5])
                ))
                
            if len(traj_batch) >= 10000:
                c.executemany("INSERT OR REPLACE INTO model_trajectory_scores VALUES (?, ?, ?, ?, ?, ?)", traj_batch)
                traj_batch.clear()
                
            if len(trans_batch) >= 50000:
                c.executemany("INSERT OR REPLACE INTO model_transition_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", trans_batch)
                uncommitted_transitions += len(trans_batch)
                trans_batch.clear()
                
                # Periodically commit to keep SQLite transaction memory small
                if uncommitted_transitions >= 250000:
                    conn.commit()
                    uncommitted_transitions = 0
                    
    if traj_batch:
        c.executemany("INSERT OR REPLACE INTO model_trajectory_scores VALUES (?, ?, ?, ?, ?, ?)", traj_batch)
        traj_batch.clear()
    if trans_batch:
        c.executemany("INSERT OR REPLACE INTO model_transition_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", trans_batch)
        trans_batch.clear()
        
    conn.commit()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Plausibility Database in chunks to avoid OOM.")
    parser.add_argument("--init", action="store_true", help="Initialize database (Drops existing!)")
    parser.add_argument("--dataset", type=str, help="Load base data for dataset (e.g. gowalla, inat, synthetic)")
    parser.add_argument("--model-score", type=str, help="Path to a single pickle file to load")
    
    args = parser.parse_args()
    
    if not any([args.init, args.dataset, args.model_score]):
        parser.print_help()
        exit(1)
        
    conn = init_db(recreate=args.init)
    
    if args.dataset:
        prefix = ""
        if args.dataset != "inat":
            prefix = f"{args.dataset}_"
        load_dataset_base(conn, args.dataset, prefix)
        
    if args.model_score:
        load_single_model_score(conn, args.model_score)
        
    conn.close()
    print("Task complete!")
