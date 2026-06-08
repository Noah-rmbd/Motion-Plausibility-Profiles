import os
import http.server
import socketserver
import webbrowser
import json
import urllib.parse
import sqlite3
import numpy as np

DB_PATH = "plausibility.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        query = urllib.parse.parse_qs(parsed_path.query)
        
        if path == '/api/models':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            if not os.path.exists(DB_PATH):
                self.wfile.write(b"[]")
                return
                
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("SELECT * FROM models")
            models = [dict(row) for row in c.fetchall()]
            conn.close()
            
            self.wfile.write(json.dumps(models).encode('utf-8'))
            return
            
        elif path == '/api/users':
            dataset = query.get('dataset', [''])[0]
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            if not os.path.exists(DB_PATH):
                self.wfile.write(b"[]")
                return
                
            conn = get_db_connection()
            c = conn.cursor()
            if dataset:
                c.execute("SELECT username, nb_observations FROM users WHERE dataset=? ORDER BY nb_observations DESC LIMIT 1000", (dataset,))
            else:
                c.execute("SELECT username, nb_observations FROM users ORDER BY nb_observations DESC LIMIT 1000")
                
            users = [dict(row) for row in c.fetchall()]
            conn.close()
            
            self.wfile.write(json.dumps(users).encode('utf-8'))
            return
            
        elif path == '/api/user_data':
            user_id = query.get('user_id', [''])[0]
            dataset = query.get('dataset', [''])[0]
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            if not os.path.exists(DB_PATH) or not user_id or not dataset:
                self.wfile.write(b"{}")
                return
                
            conn = get_db_connection()
            c = conn.cursor()
            
            # Observations
            c.execute("SELECT * FROM observations WHERE dataset=? AND user_id=?", (dataset, user_id))
            observations = [dict(row) for row in c.fetchall()]
            
            # Transitions
            c.execute("SELECT * FROM transitions WHERE dataset=? AND user_id=?", (dataset, user_id))
            transitions = [dict(row) for row in c.fetchall()]
            
            # Trajectories
            c.execute("SELECT * FROM trajectories WHERE dataset=? AND user_id=?", (dataset, user_id))
            trajectories = []
            for row in c.fetchall():
                r = dict(row)
                r['transitions'] = json.loads(r['transitions_json'])
                del r['transitions_json']
                trajectories.append(r)
                
            conn.close()
            
            self.wfile.write(json.dumps({
                "observations": observations,
                "transitions": transitions,
                "trajectories": trajectories
            }).encode('utf-8'))
            return

        super().do_GET()
        
    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == '/api/scores':
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_response(400)
                self.end_headers()
                return
                
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode('utf-8'))
            
            model_id = req.get('model_id')
            percentile = req.get('percentile', 97)
            transition_ids = req.get('transition_ids', [])
            trajectory_ids = req.get('trajectory_ids', [])
            
            if not os.path.exists(DB_PATH) or not model_id:
                self.send_response(404)
                self.end_headers()
                return
                
            conn = get_db_connection()
            c = conn.cursor()
            
            # Suffix/prefix matching to handle model_id configurations dynamically
            c.execute("SELECT model_id, percentile FROM models WHERE model_id = ? OR model_id LIKE ?", (model_id, f"{model_id}%"))
            model_row = c.fetchone()
            db_model_id = model_row[0] if model_row else model_id
            use_stored_flags = bool(model_row and model_row["percentile"] == 0)
            
            res_trans = {}
            res_traj = {}
            unplausible_trans_ids = set()
            
            # Calculate percentile threshold dynamically
            c.execute("SELECT reconstruction_error FROM model_trajectory_scores WHERE model_id = ? AND reconstruction_error != -1.0", (db_model_id,))
            all_errors = [row[0] for row in c.fetchall()]
            if all_errors:
                threshold = float(np.percentile(all_errors, percentile))
            else:
                threshold = 0.0191
            
            if trajectory_ids:
                placeholders = ','.join('?' * len(trajectory_ids))
                c.execute(f"SELECT * FROM model_trajectory_scores WHERE model_id=? AND trajectory_id IN ({placeholders})", [db_model_id] + trajectory_ids)
                for row in c.fetchall():
                    row_dict = dict(row)
                    rec_err = row_dict['reconstruction_error']
                    if use_stored_flags:
                        is_unplausible = bool(row_dict['is_unplausible'])
                    else:
                        is_unplausible = (rec_err == -1.0) or (rec_err > threshold)
                    row_dict['is_unplausible'] = is_unplausible
                    
                    if is_unplausible and row_dict['least_plausible_transition_id'] is not None:
                        unplausible_trans_ids.add(row_dict['least_plausible_transition_id'])
                    
                    res_traj[str(row_dict['trajectory_id'])] = row_dict
            
            if transition_ids:
                placeholders = ','.join('?' * len(transition_ids))
                c.execute(f"SELECT * FROM model_transition_scores WHERE model_id=? AND transition_id IN ({placeholders})", [db_model_id] + transition_ids)
                for row in c.fetchall():
                    d = dict(row)
                    d['features'] = [
                        d['feature_error_speed'], d['feature_error_time'], d['feature_error_dist'], 
                        d['feature_error_accel'], d['feature_error_bear_sin'], d['feature_error_bear_cos']
                    ]
                    d['is_unplausible'] = d['transition_id'] in unplausible_trans_ids
                    res_trans[str(d['transition_id'])] = d
                    
            conn.close()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"transitions": res_trans, "trajectories": res_traj}).encode('utf-8'))
            return
            
        self.send_response(404)
        self.end_headers()

PORT = int(os.environ.get("PORT", "8001"))
Handler = CustomHandler

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"WARNING: {DB_PATH} not found. Please run build_database.py first.")
        
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at port {PORT}")
        url = "http://localhost:" + str(PORT) + "/Interactive_Visualisation/main.html"
        webbrowser.open(url)
        httpd.serve_forever()
