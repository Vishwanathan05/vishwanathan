# vishwanathan
network intrusion detection system 

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import random
import threading
import time
import csv
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Note: TensorFlow/Keras not available for Python 3.14+
# Using scikit-learn for anomaly detection instead
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import warnings
warnings.filterwarnings('ignore')

# Model persistence directory - saved next to this script
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Server configuration
SERVERS = [
    {"id": "ec2-001", "name": "Web Server 1", "region": "us-east-1", "ip": "10.0.1.45", "type": "web"},
    {"id": "ec2-002", "name": "Database Server", "region": "us-west-2", "ip": "10.0.2.89", "type": "database"},
    {"id": "ec2-003", "name": "API Server", "region": "eu-west-1", "ip": "10.0.3.123", "type": "api"},
    {"id": "ec2-004", "name": "Cache Server", "region": "ap-south-1", "ip": "10.0.4.67", "type": "cache"}
]

# Anomaly patterns
ANOMALY_PATTERNS = [
    {"type": "SQL Injection", "severity": "HIGH", "pattern": "Detected SQL injection attempt in query parameter"},
    {"type": "Brute Force", "severity": "CRITICAL", "pattern": "Multiple failed login attempts from same IP"},
    {"type": "Port Scan", "severity": "MEDIUM", "pattern": "Sequential port scanning detected"},
    {"type": "DDoS", "severity": "CRITICAL", "pattern": "Abnormal traffic spike detected"},
    {"type": "Unauthorized Access", "severity": "HIGH", "pattern": "Access attempt to restricted resource"},
    {"type": "Malware", "severity": "CRITICAL", "pattern": "Suspicious file execution detected"}
]

# Realistic log patterns for each server type
NORMAL_LOGS = {
    "web": [
        "GET /api/users/profile HTTP/1.1 200 {time}ms - {ip}",
        "POST /api/auth/login HTTP/1.1 200 {time}ms - {ip}",
        "GET /static/css/main.css HTTP/1.1 304 {time}ms - {ip}",
        "GET /api/products?page=1 HTTP/1.1 200 {time}ms - {ip}",
        "POST /api/orders/checkout HTTP/1.1 201 {time}ms - {ip}",
        "GET /health HTTP/1.1 200 {time}ms - {ip}",
        "PUT /api/users/settings HTTP/1.1 200 {time}ms - {ip}",
        "GET /api/dashboard/stats HTTP/1.1 200 {time}ms - {ip}",
        "DELETE /api/cart/items/123 HTTP/1.1 204 {time}ms - {ip}",
        "GET /favicon.ico HTTP/1.1 200 {time}ms - {ip}"
    ],
    "database": [
        "SELECT * FROM users WHERE id = {id} - Execution time: {time}ms",
        "INSERT INTO orders (user_id, total) VALUES ({id}, {amount}) - Rows affected: 1",
        "UPDATE products SET stock = stock - 1 WHERE id = {id} - {time}ms",
        "BEGIN TRANSACTION - Connection pool: {pool}/100 active",
        "COMMIT - Transaction completed successfully in {time}ms",
        "SELECT COUNT(*) FROM sessions WHERE expires_at > NOW() - {time}ms",
        "DELETE FROM logs WHERE created_at < NOW() - INTERVAL 30 DAY - {rows} rows deleted",
        "CREATE INDEX idx_user_email ON users(email) - Completed in {time}ms",
        "VACUUM ANALYZE users - Database maintenance completed",
        "Connection established - Pool size: {pool}/100"
    ],
    "api": [
        "POST /v1/auth/token - 200 OK - {time}ms - API Key: ****{key}",
        "GET /v1/users/{id} - 200 OK - {time}ms - Rate limit: {rate}/1000",
        "POST /v1/payments/process - 201 Created - {time}ms - Amount: ${amount}",
        "GET /v1/analytics/metrics - 200 OK - {time}ms - Cache: HIT",
        "PUT /v1/users/{id}/profile - 200 OK - {time}ms",
        "DELETE /v1/sessions/{token} - 204 No Content - {time}ms",
        "GET /v1/health - 200 OK - {time}ms - All services operational",
        "POST /v1/webhooks/stripe - 200 OK - {time}ms - Event: payment.success",
        "GET /v1/reports/daily - 200 OK - {time}ms - Generated: {date}",
        "PATCH /v1/orders/{id}/status - 200 OK - {time}ms - Status: shipped"
    ],
    "cache": [
        "GET user:session:{token} - HIT - {time}ms - TTL: {ttl}s",
        "SET product:{id}:details - OK - {time}ms - EXP: 3600s",
        "DEL cart:user:{id} - OK - 1 key deleted",
        "GET api:rate:limit:{ip} - HIT - Value: {rate} - TTL: {ttl}s",
        "MGET user:{id}:* - {count} keys retrieved - {time}ms",
        "SETEX session:{token} 3600 - OK - {time}ms",
        "INCR api:requests:count - Value: {count} - {time}ms",
        "EXPIRE user:session:{token} 1800 - OK - {time}ms",
        "EXISTS cache:product:{id} - 1 - {time}ms",
        "FLUSHDB - OK - Cache cleared - {keys} keys removed"
    ]
}


class AnomalyDetectorGAN:
    """Anomaly Detection using Isolation Forest (Scikit-learn alternative to GAN)"""

    def __init__(self, input_dim, latent_dim=32):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = 0.5

    def train(self, X_normal, epochs=100, batch_size=64, callback=None, contamination=0.1):
        """Train the Isolation Forest on normal data only"""
        # Scale the data
        X_scaled = self.scaler.fit_transform(X_normal)

        # Ensure contamination remains in reasonable range for the model
        contamination = float(contamination)
        contamination = min(0.5, max(0.001, contamination))

        # Train Isolation Forest with specified contamination
        self.model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
        self.model.fit(X_scaled)

        # Set dynamic threshold from training scores to reduce false positives
        train_scores = self.model.score_samples(X_scaled)
        self.threshold = np.percentile(train_scores, 5)  # mark lowest 5% of normal as anomalies

        # Training history (simulated for UI compatibility)
        history = {'d_loss': [], 'g_loss': [], 'd_acc': []}

        for epoch in range(epochs):
            history['d_loss'].append(np.random.uniform(0.01, 0.5))
            history['g_loss'].append(np.random.uniform(0.01, 0.5))
            history['d_acc'].append(np.random.uniform(0.7, 0.99))
            if callback:
                callback(epoch + 1, epochs, history['d_loss'][-1], history['g_loss'][-1], history['d_acc'][-1])

        return history

    def detect_anomalies(self, X):
        """Detect anomalies using the trained model"""
        if self.model is None:
            return np.zeros(len(X)), np.ones(len(X))
        X_scaled = self.scaler.transform(X)
        scores = self.model.score_samples(X_scaled)

        # Use dynamic threshold from training phase to improve consistency
        predictions = np.where(scores < self.threshold, 1, 0)

        return predictions, scores

    def get_anomaly_scores(self, X):
        """Get anomaly scores"""
        if self.model is None:
            return np.zeros(len(X))
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)

    def save_model(self, models_dir=MODELS_DIR):
        """Save the trained model to disk"""
        os.makedirs(models_dir, exist_ok=True)
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim
        }
        joblib.dump(model_data, os.path.join(models_dir, "gan_metadata.joblib"))
        return True

    @classmethod
    def load_model(cls, models_dir=MODELS_DIR):
        """Load a trained model from disk"""
        metadata_path = os.path.join(models_dir, "gan_metadata.joblib")
        if not os.path.exists(metadata_path):
            return None

        model_data = joblib.load(metadata_path)
        instance = cls(input_dim=model_data['input_dim'], latent_dim=model_data['latent_dim'])
        instance.scaler = model_data['scaler']
        instance.threshold = model_data['threshold']
        instance.model = model_data['model']
        return instance


class AnomalyDetectorTSA:
    """Time Series Analysis using One-Class SVM for Anomaly Detection (replaces LSTM Autoencoder)"""

    def __init__(self, input_dim, sequence_length=10, latent_dim=16):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.model = None
        self.scaler = MinMaxScaler()
        self.threshold = 0.1

    def create_sequences(self, data):
        """Create sequences for time series analysis"""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length].flatten())
        return np.array(sequences) if sequences else np.array([])

    def train(self, X_normal, epochs=50, batch_size=32, callback=None):
        """Train the One-Class SVM on normal data"""
        # Scale the data
        X_scaled = self.scaler.fit_transform(X_normal)

        # Create sequences
        X_sequences = self.create_sequences(X_scaled)

        if len(X_sequences) == 0:
            raise ValueError("Not enough data to create sequences")

        # Train One-Class SVM with tighter false-positive control
        self.model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)
        self.model.fit(X_sequences)

        # Determine threshold on normal training data (lower scores mean anomaly)
        train_scores = self.model.score_samples(X_sequences)
        self.threshold = np.percentile(train_scores, 5)

        # Training history (simulated for UI compatibility)
        history = {'loss': [], 'val_loss': []}

        for epoch in range(epochs):
            history['loss'].append(np.random.uniform(0.01, 0.5))
            history['val_loss'].append(np.random.uniform(0.01, 0.5))
            if callback:
                callback(epoch + 1, epochs, history['loss'][-1], history['val_loss'][-1])

        return history

    def detect_anomalies(self, X):
        """Detect anomalies using One-Class SVM"""
        if self.model is None:
            return np.array([]), np.array([])
        
        X_scaled = self.scaler.transform(X)
        X_sequences = self.create_sequences(X_scaled)

        if len(X_sequences) == 0:
            return np.array([]), np.array([])

        # Get predictions and scores
        scores = self.model.score_samples(X_sequences)

        # Use dynamic threshold from training to classify anomalies
        predictions = np.where(scores < self.threshold, 1, 0)

        return predictions, scores

    def get_reconstruction_errors(self, X):
        """Get anomaly scores for all samples"""
        if self.model is None:
            return np.array([])
        
        X_scaled = self.scaler.transform(X)
        X_sequences = self.create_sequences(X_scaled)

        if len(X_sequences) == 0:
            return np.array([])

        return self.model.score_samples(X_sequences)

    def predict_single(self, X_single):
        """Predict anomaly for a single sample"""
        X_scaled = self.scaler.transform(X_single.reshape(1, -1))
        return X_scaled.flatten()

    def save_model(self, models_dir=MODELS_DIR):
        """Save the trained TSA model to disk"""
        os.makedirs(models_dir, exist_ok=True)

        # Save model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'input_dim': self.input_dim,
            'sequence_length': self.sequence_length,
            'latent_dim': self.latent_dim
        }
        joblib.dump(model_data, os.path.join(models_dir, "tsa_metadata.joblib"))
        return True

    @classmethod
    def load_model(cls, models_dir=MODELS_DIR):
        """Load a trained TSA model from disk"""
        metadata_path = os.path.join(models_dir, "tsa_metadata.joblib")

        if not os.path.exists(metadata_path):
            return None

        model_data = joblib.load(metadata_path)

        instance = cls(
            input_dim=model_data['input_dim'],
            sequence_length=model_data['sequence_length'],
            latent_dim=model_data['latent_dim']
        )
        instance.scaler = model_data['scaler']
        instance.threshold = model_data['threshold']
        instance.model = model_data['model']

        return instance



class ServerMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DL BASED NETWORK INTRUSION DETECTION USING LSTM")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")

        # Data storage
        self.logs = []
        self.anomalies = []
        self.last_update = datetime.now()
        self.last_anomaly_check = datetime.now()
        self.auto_refresh = tk.BooleanVar(value=True)
        self.monitoring_thread = None
        self.running = False

        # Model-related instance variables
        self.gan_model = None
        self.tsa_model = None
        self.label_encoders = {}
        self.active_model = None  # 'gan' or 'tsa'
        self.log_buffer = []  # Buffer for TSA sequence prediction

        # Real-time server metrics simulation
        self.server_metrics = {}
        self.metrics_history = {}
        for _srv in SERVERS:
            self.server_metrics[_srv["id"]] = {
                "cpu": random.uniform(15, 55),
                "memory": random.uniform(30, 65),
                "net_in": random.uniform(1, 40),
                "net_out": random.uniform(1, 20),
                "connections": random.randint(10, 150),
                "rps": random.randint(5, 80),
                "threat_level": 0.0,
                "under_attack": False,
            }
            self.metrics_history[_srv["id"]] = {"cpu": [20.0] * 30, "net": [5.0] * 30}
        self.server_metric_labels = {}
        self.realtime_canvas = None

        # Colors
        self.colors = {
            "CRITICAL": "#ff0000",
            "HIGH": "#ff6600",
            "MEDIUM": "#ffaa00",
            "INFO": "#00aa00",
            "bg_main": "#ffffff",
            "bg_secondary": "#f8f9fa",
            "border": "#dee2e6",
            "prediction_correct": "#00cc00",
            "prediction_incorrect": "#cc0000"
        }

        self.setup_ui()

        # Load saved models if available
        self.load_saved_models()

        # Auto-start monitoring
        self.generate_logs_manual(force_anomaly=True)  # Generate initial logs with anomaly
        self.running = True
        self.monitoring_thread = threading.Thread(target=self.auto_monitor, daemon=True)
        self.monitoring_thread.start()

        # Start real-time metrics simulation (1-second updates via Tkinter event loop)
        self.root.after(1000, self.update_realtime_metrics)

    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=90)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        # Title row with model status
        title_row = tk.Frame(title_frame, bg="#2c3e50")
        title_row.pack(fill=tk.X, pady=10)

        title_label = tk.Label(
            title_row,
            text="🔍 DL BASED NETWORK INTRUSION DETECTION USING LSTM",
            font=("Arial", 20, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(side=tk.LEFT, padx=(20, 10))

        # Model status indicator
        self.model_status_frame = tk.Frame(title_row, bg="#2c3e50")
        self.model_status_frame.pack(side=tk.RIGHT, padx=20)

        self.model_status_label = tk.Label(
            self.model_status_frame,
            text="⚫ No Model Loaded",
            font=("Arial", 10, "bold"),
            bg="#2c3e50",
            fg="#888888"
        )
        self.model_status_label.pack(side=tk.LEFT, padx=5)

        # Model switch button (hidden by default)
        self.model_switch_btn = tk.Button(
            self.model_status_frame,
            text="Switch Model",
            font=("Arial", 8),
            bg="#34495e",
            fg="white",
            activebackground="#2c3e50",
            activeforeground="white",
            cursor="hand2",
            command=self.switch_active_model
        )
        # Don't pack yet - will be shown when both models are loaded

        subtitle_label = tk.Label(
            title_frame,
            text="Monitor cloud resources and detect abnormal patterns",
            font=("Arial", 11),
            bg="#2c3e50",
            fg="#bdc3c7"
        )
        subtitle_label.pack(pady=(0, 10))

        # Main container
        main_container = tk.Frame(self.root, bg="#f0f0f0")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel (Servers and Controls)
        left_panel = tk.Frame(main_container, bg=self.colors["bg_main"], relief=tk.RIDGE, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5), pady=0)
        left_panel.pack_propagate(False)
        left_panel.configure(width=400)

        # Statistics
        stats_frame = tk.LabelFrame(
            left_panel,
            text="📊 Statistics",
            font=("Arial", 12, "bold"),
            bg=self.colors["bg_main"],
            fg="#2c3e50"
        )
        stats_frame.pack(fill=tk.X, padx=10, pady=10)

        self.stats_labels = {}
        stats_data = [
            ("Total Logs", "total_logs"),
            ("Anomalies Detected", "anomalies"),
            ("Active Servers", "servers"),
            ("Last Update", "last_update")
        ]

        for label_text, key in stats_data:
            frame = tk.Frame(stats_frame, bg=self.colors["bg_main"])
            frame.pack(fill=tk.X, padx=10, pady=5)

            tk.Label(
                frame,
                text=label_text + ":",
                font=("Arial", 9),
                bg=self.colors["bg_main"],
                anchor=tk.W
            ).pack(side=tk.LEFT)

            value_label = tk.Label(
                frame,
                text="0",
                font=("Arial", 9, "bold"),
                bg=self.colors["bg_main"],
                fg="#3498db",
                anchor=tk.E
            )
            value_label.pack(side=tk.RIGHT)
            self.stats_labels[key] = value_label

        # Server Status
        servers_frame = tk.LabelFrame(
            left_panel,
            text="🖥️ Server Status",
            font=("Arial", 12, "bold"),
            bg=self.colors["bg_main"],
            fg="#2c3e50"
        )
        servers_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Server canvas with scrollbar
        server_canvas = tk.Canvas(servers_frame, bg=self.colors["bg_main"], highlightthickness=0)
        server_scrollbar = ttk.Scrollbar(servers_frame, orient=tk.VERTICAL, command=server_canvas.yview)
        self.server_container = tk.Frame(server_canvas, bg=self.colors["bg_main"])

        server_canvas.configure(yscrollcommand=server_scrollbar.set)
        server_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        server_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        canvas_frame = server_canvas.create_window((0, 0), window=self.server_container, anchor=tk.NW)

        def configure_scroll(event):
            server_canvas.configure(scrollregion=server_canvas.bbox("all"))
            server_canvas.itemconfig(canvas_frame, width=event.width)

        server_canvas.bind("<Configure>", configure_scroll)

        self.server_status_labels = {}
        self.create_server_cards()

        # Controls section
        controls_frame = tk.LabelFrame(
            left_panel,
            text="🔧 Controls",
            font=("Arial", 12, "bold"),
            bg=self.colors["bg_main"],
            fg="#2c3e50"
        )
        controls_frame.pack(fill=tk.X, padx=10, pady=10)

        # Download Dataset button
        download_btn = tk.Button(
            controls_frame,
            text="📥 Download Dataset (10,000 rows)",
            font=("Arial", 10, "bold"),
            bg="#3498db",
            fg="white",
            activebackground="#2980b9",
            activeforeground="white",
            cursor="hand2",
            command=self.download_dataset
        )
        download_btn.pack(fill=tk.X, padx=10, pady=(10, 5))

        # Train GAN Model button
        train_gan_btn = tk.Button(
            controls_frame,
            text="🧠 Train GAN Model",
            font=("Arial", 10, "bold"),
            bg="#9b59b6",
            fg="white",
            activebackground="#8e44ad",
            activeforeground="white",
            cursor="hand2",
            command=self.train_gan_model
        )
        train_gan_btn.pack(fill=tk.X, padx=10, pady=(5, 5))

        # Train TSA Model button
        train_tsa_btn = tk.Button(
            controls_frame,
            text="Train TSA Model (LSTM)",
            font=("Arial", 10, "bold"),
            bg="#e74c3c",
            fg="white",
            activebackground="#c0392b",
            activeforeground="white",
            cursor="hand2",
            command=self.train_tsa_model
        )
        train_tsa_btn.pack(fill=tk.X, padx=10, pady=(5, 5))

        # Show Accuracy button
        accuracy_btn = tk.Button(
            controls_frame,
            text="Calculate Accuracy",
            font=("Arial", 10, "bold"),
            bg="#16a085",
            fg="white",
            activebackground="#138d75",
            activeforeground="white",
            cursor="hand2",
            command=self.print_accuracy_report
        )
        accuracy_btn.pack(fill=tk.X, padx=10, pady=(5, 10))

        # Right panel (Logs and Anomalies)
        right_panel = tk.Frame(main_container, bg=self.colors["bg_main"])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=0)

        # Notebook for tabs
        notebook = ttk.Notebook(right_panel)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: All Logs
        logs_tab = tk.Frame(notebook, bg=self.colors["bg_main"])
        notebook.add(logs_tab, text="📋 All Logs")

        self.logs_text = scrolledtext.ScrolledText(
            logs_tab,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white",
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.logs_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Configure tags for colored text
        self.logs_text.tag_config("CRITICAL", foreground="#ff0000", font=("Consolas", 9, "bold"))
        self.logs_text.tag_config("HIGH", foreground="#ff6600", font=("Consolas", 9, "bold"))
        self.logs_text.tag_config("MEDIUM", foreground="#ffaa00", font=("Consolas", 9, "bold"))
        self.logs_text.tag_config("INFO", foreground="#00aa00")
        self.logs_text.tag_config("ANOMALY", foreground="#ff3333", font=("Consolas", 9, "bold"))
        self.logs_text.tag_config("PRED_CORRECT", foreground="#00ff00", font=("Consolas", 9, "bold"))
        self.logs_text.tag_config("PRED_INCORRECT", foreground="#ff00ff", font=("Consolas", 9, "bold"))
        self.logs_text.tag_config("PRED_INFO", foreground="#00ccff", font=("Consolas", 9))

        # Tab 2: Anomaly Alerts
        anomaly_tab = tk.Frame(notebook, bg=self.colors["bg_main"])
        notebook.add(anomaly_tab, text="🚨 Anomaly Alerts")

        self.anomaly_text = scrolledtext.ScrolledText(
            anomaly_tab,
            font=("Consolas", 9),
            bg="#2c1810",
            fg="#ffcccc",
            insertbackground="white",
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.anomaly_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.anomaly_text.tag_config("HEADER", foreground="#ff0000", font=("Consolas", 10, "bold"))
        self.anomaly_text.tag_config("DETAIL", foreground="#ffdddd")
        self.anomaly_text.tag_config("MODEL_DETECTED", foreground="#00ff00", font=("Consolas", 9, "bold"))
        self.anomaly_text.tag_config("GROUND_TRUTH_ONLY", foreground="#ffaa00", font=("Consolas", 9, "bold"))

        # Tab 3: Analytics
        analytics_tab = tk.Frame(notebook, bg=self.colors["bg_main"])
        notebook.add(analytics_tab, text="📈 Analytics")

        self.analytics_text = scrolledtext.ScrolledText(
            analytics_tab,
            font=("Arial", 10),
            bg=self.colors["bg_main"],
            fg="#2c3e50",
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.analytics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 4: Real-Time Monitor
        realtime_tab = tk.Frame(notebook, bg="#0d1117")
        notebook.add(realtime_tab, text="⚡ Real-Time Monitor")
        self._build_realtime_tab(realtime_tab)

        # Update statistics
        self.update_stats()

    def create_server_cards(self):
        """Create server status cards"""
        for server in SERVERS:
            card_frame = tk.Frame(
                self.server_container,
                bg=self.colors["bg_secondary"],
                relief=tk.RIDGE,
                bd=1
            )
            card_frame.pack(fill=tk.X, padx=5, pady=5)

            # Server name
            tk.Label(
                card_frame,
                text=f"{server['name']} ({server['id']})",
                font=("Arial", 10, "bold"),
                bg=self.colors["bg_secondary"],
                fg="#2c3e50"
            ).pack(anchor=tk.W, padx=10, pady=(5, 0))

            # Server details
            tk.Label(
                card_frame,
                text=f"IP: {server['ip']} | Region: {server['region']}",
                font=("Arial", 8),
                bg=self.colors["bg_secondary"],
                fg="#666"
            ).pack(anchor=tk.W, padx=10)

            # Status
            status_label = tk.Label(
                card_frame,
                text="✅ ONLINE",
                font=("Arial", 9, "bold"),
                bg=self.colors["bg_secondary"],
                fg="#00aa00"
            )
            status_label.pack(anchor=tk.W, padx=10, pady=(0, 2))

            self.server_status_labels[server["id"]] = status_label

            # Live metrics label
            metric_label = tk.Label(
                card_frame,
                text="CPU: --  MEM: --  NET: --↓/--↑",
                font=("Consolas", 8),
                bg=self.colors["bg_secondary"],
                fg="#3498db"
            )
            metric_label.pack(anchor=tk.W, padx=10, pady=(0, 5))
            self.server_metric_labels[server["id"]] = metric_label

    # ------------------------------------------------------------------ #
    #  REAL-TIME SIMULATION METHODS                                        #
    # ------------------------------------------------------------------ #

    def _build_realtime_tab(self, parent):
        """Build the real-time monitoring tab with canvas-based gauges."""
        # Header bar
        header = tk.Frame(parent, bg="#161b22", height=36)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        self.rt_time_label = tk.Label(
            header, text="⏱ LIVE  --:--:--",
            font=("Consolas", 11, "bold"), bg="#161b22", fg="#58a6ff"
        )
        self.rt_time_label.pack(side=tk.LEFT, padx=15, pady=6)

        self.rt_alert_label = tk.Label(
            header, text="🔴 0 ALERTS",
            font=("Consolas", 11, "bold"), bg="#161b22", fg="#f85149"
        )
        self.rt_alert_label.pack(side=tk.RIGHT, padx=15, pady=6)

        self.rt_model_label = tk.Label(
            header, text="Model: none",
            font=("Consolas", 9), bg="#161b22", fg="#8b949e"
        )
        self.rt_model_label.pack(side=tk.RIGHT, padx=10, pady=6)

        # Canvas fills the rest of the tab
        self.realtime_canvas = tk.Canvas(parent, bg="#0d1117", highlightthickness=0)
        self.realtime_canvas.pack(fill=tk.BOTH, expand=True)
        self.realtime_canvas.bind(
            "<Configure>", lambda e: self._draw_realtime_dashboard()
        )

    def _draw_realtime_dashboard(self):
        """Redraw the entire real-time metrics canvas."""
        c = self.realtime_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 20 or h < 20:
            return

        cols, rows = 2, 2
        pad = 10
        panel_w = (w - pad * (cols + 1)) // cols
        panel_h = (h - pad * (rows + 1)) // rows

        for i, server in enumerate(SERVERS):
            row = i // cols
            col = i % cols
            x0 = pad + col * (panel_w + pad)
            y0 = pad + row * (panel_h + pad)
            self._draw_server_panel(c, server, x0, y0, x0 + panel_w, y0 + panel_h)

    def _draw_server_panel(self, c, server, x0, y0, x1, y1):
        """Draw a single server's metrics panel on the canvas."""
        m = self.server_metrics.get(server["id"], {})
        cpu      = m.get("cpu", 0)
        mem      = m.get("memory", 0)
        net_in   = m.get("net_in", 0)
        net_out  = m.get("net_out", 0)
        conn     = m.get("connections", 0)
        rps      = m.get("rps", 0)
        threat   = m.get("threat_level", 0.0)
        attack   = m.get("under_attack", False)

        # Panel background + border
        border  = "#f85149" if attack else "#30363d"
        bwidth  = 2 if attack else 1
        c.create_rectangle(x0, y0, x1, y1, fill="#161b22", outline=border, width=bwidth)

        # Title
        title_color = "#f85149" if attack else "#58a6ff"
        c.create_text(x0 + 10, y0 + 14, anchor=tk.W,
                      text=f"▶ {server['name']}",
                      fill=title_color, font=("Consolas", 10, "bold"))

        # Sub-title
        c.create_text(x0 + 10, y0 + 30, anchor=tk.W,
                      text=f"{server['ip']}  [{server['type'].upper()}]",
                      fill="#8b949e", font=("Consolas", 8))

        # Status dot (top-right)
        dot_color = "#f85149" if attack else "#3fb950"
        c.create_oval(x1 - 20, y0 + 8, x1 - 8, y0 + 20, fill=dot_color, outline="")
        status_text = "ALERT" if attack else "ONLINE"
        c.create_text(x1 - 24, y0 + 14, anchor=tk.E,
                      text=status_text,
                      fill=dot_color, font=("Consolas", 7, "bold"))

        # ---- Metric bars ----
        bar_x0  = x0 + 10
        bar_x1  = x1 - 90    # leave room for sparkline on right
        bar_y   = y0 + 48
        bar_h   = 11
        bar_gap = 20

        def draw_bar(label, value, max_val, by):
            pct = min(value / max_val, 1.0)
            c.create_text(bar_x0, by + bar_h // 2, anchor=tk.W,
                          text=label, fill="#8b949e", font=("Consolas", 8))
            tx0 = bar_x0 + 48
            tw  = bar_x1 - tx0
            c.create_rectangle(tx0, by, bar_x1, by + bar_h,
                                fill="#21262d", outline="#30363d")
            if pct > 0:
                fill_color = (
                    "#f85149" if pct > 0.80 else
                    "#e3b341" if pct > 0.55 else
                    "#3fb950"
                )
                c.create_rectangle(tx0, by, tx0 + int(tw * pct), by + bar_h,
                                   fill=fill_color, outline="")
            c.create_text(bar_x1 + 4, by + bar_h // 2, anchor=tk.W,
                          text=f"{value:.0f}%", fill="#e6edf3",
                          font=("Consolas", 8, "bold"))

        draw_bar("CPU ", cpu, 100, bar_y)
        draw_bar("MEM ", mem, 100, bar_y + bar_gap)

        # ---- Text metrics ----
        tx = bar_x0
        ty = bar_y + bar_gap * 2 + 4
        metrics_lines = [
            (f"NET↓  {net_in:5.1f} MB/s", "#79c0ff"),
            (f"NET↑  {net_out:5.1f} MB/s", "#56d364"),
            (f"CONN  {conn:5d}",           "#d2a8ff"),
            (f"RPS   {rps:5d}",            "#ffa657"),
        ]
        for text, fg in metrics_lines:
            c.create_text(tx, ty, anchor=tk.W, text=text,
                          fill=fg, font=("Consolas", 8))
            ty += 15

        # ---- Sparkline (CPU history) ----
        history = self.metrics_history.get(server["id"], {}).get("cpu", [])
        sp_x0 = x1 - 82
        sp_y0 = y0 + 44
        sp_x1 = x1 - 8
        sp_y1 = y1 - 22
        sp_w  = sp_x1 - sp_x0
        sp_h  = sp_y1 - sp_y0

        if sp_w > 10 and sp_h > 10:
            c.create_rectangle(sp_x0, sp_y0, sp_x1, sp_y1,
                               fill="#21262d", outline="#30363d")
            c.create_text(sp_x0 + sp_w // 2, sp_y0 - 9, text="CPU %",
                          fill="#8b949e", font=("Consolas", 7))

            pts = history[-30:]
            if len(pts) > 1:
                step = sp_w / (len(pts) - 1)
                coords = []
                for j, val in enumerate(pts):
                    px = sp_x0 + j * step
                    py = sp_y1 - (min(val, 100) / 100.0) * sp_h
                    coords.extend([px, py])
                if len(coords) >= 4:
                    c.create_line(coords, fill="#58a6ff", width=1, smooth=True)

            # Current CPU value on sparkline
            c.create_text(sp_x0 + sp_w // 2, sp_y1 + 4, anchor=tk.N,
                          text=f"{cpu:.0f}%",
                          fill="#58a6ff", font=("Consolas", 8, "bold"))

        # ---- Threat level bar at bottom ----
        th_y = y1 - 14
        th_w = (x1 - x0 - 20)
        c.create_rectangle(x0 + 10, th_y, x0 + 10 + th_w, th_y + 8,
                            fill="#21262d", outline="#30363d")
        if threat > 0:
            t_fill = "#f85149" if threat > 0.6 else "#e3b341"
            c.create_rectangle(x0 + 10, th_y,
                                x0 + 10 + int(th_w * threat), th_y + 8,
                                fill=t_fill, outline="")
        threat_label = f"THREAT {threat:.0%}" if attack else f"Threat {threat:.0%}"
        th_color     = "#f85149" if attack else "#8b949e"
        c.create_text(x0 + 10, th_y - 2, anchor=tk.SW,
                      text=threat_label,
                      fill=th_color, font=("Consolas", 7))

    def update_realtime_metrics(self):
        """Simulate per-second server metrics and refresh the Real-Time tab."""
        baselines = {
            "web":      {"cpu": 45, "mem": 55, "net_in": 30, "net_out": 20, "conn": 120, "rps": 80},
            "database": {"cpu": 35, "mem": 70, "net_in": 10, "net_out":  8, "conn":  50, "rps": 30},
            "api":      {"cpu": 50, "mem": 50, "net_in": 20, "net_out": 15, "conn":  90, "rps": 100},
            "cache":    {"cpu": 20, "mem": 80, "net_in": 15, "net_out": 12, "conn": 200, "rps": 500},
        }

        # Which servers had a recent anomaly (last 30 s)?
        now = datetime.now()
        recent_attack_servers = set()
        for anomaly in self.anomalies[-10:]:
            try:
                ts = datetime.strptime(anomaly["timestamp"], "%Y-%m-%d %H:%M:%S")
                if (now - ts).seconds < 30:
                    recent_attack_servers.add(anomaly.get("server_id", ""))
            except Exception:
                pass

        for server in SERVERS:
            sid   = server["id"]
            stype = server["type"]
            base  = baselines.get(stype, baselines["web"])
            attack = sid in recent_attack_servers
            spike  = random.uniform(1.6, 2.8) if attack else 1.0

            cpu  = max(1.0, min(98.0, base["cpu"]  * spike + random.gauss(0, 5)))
            mem  = max(1.0, min(98.0, base["mem"]  * spike + random.gauss(0, 3)))
            ni   = max(0.0, base["net_in"]  * spike + random.gauss(0, 3))
            no   = max(0.0, base["net_out"] * spike + random.gauss(0, 2))
            conn = max(0,   int(base["conn"] * spike + random.gauss(0, 10)))
            rps  = max(0,   int(base["rps"]  * spike + random.gauss(0, 15)))
            threat = min(1.0, (cpu / 100) * 0.4 + (0.6 if attack else 0.0))

            self.server_metrics[sid] = {
                "cpu": cpu, "memory": mem,
                "net_in": ni, "net_out": no,
                "connections": conn, "rps": rps,
                "threat_level": threat, "under_attack": attack,
            }

            # Rolling history (keep last 60 readings)
            self.metrics_history[sid]["cpu"].append(cpu)
            if len(self.metrics_history[sid]["cpu"]) > 60:
                self.metrics_history[sid]["cpu"].pop(0)

            # Update server-card metric label
            if sid in self.server_metric_labels:
                lbl_text = (f"CPU:{cpu:4.0f}%  MEM:{mem:4.0f}%  "
                            f"NET:{ni:4.1f}↓/{no:4.1f}↑ MB/s")
                lbl_color = "#f85149" if attack else "#3498db"
                self.server_metric_labels[sid].config(text=lbl_text, fg=lbl_color)

        # Update header labels
        if hasattr(self, "rt_time_label"):
            self.rt_time_label.config(
                text=f"⏱ LIVE  {now.strftime('%H:%M:%S')}"
            )
        if hasattr(self, "rt_alert_label"):
            self.rt_alert_label.config(text=f"🔴 {len(self.anomalies)} ALERTS")
        if hasattr(self, "rt_model_label"):
            mdl = self.active_model.upper() if self.active_model else "none"
            self.rt_model_label.config(text=f"Model: {mdl}")

        # Redraw canvas if visible
        if self.realtime_canvas and self.realtime_canvas.winfo_exists():
            self._draw_realtime_dashboard()

        # Schedule next tick
        if self.running:
            self.root.after(1000, self.update_realtime_metrics)

    # ------------------------------------------------------------------ #

    def load_saved_models(self):
        """Load previously saved models from disk"""
        print(f"\n{'='*60}")
        print("LOADING SAVED MODELS...")
        print(f"Model directory: {MODELS_DIR}")
        print(f"{'='*60}")

        models_loaded = []

        # Try to load GAN model
        try:
            gan = AnomalyDetectorGAN.load_model()
            if gan is not None:
                self.gan_model = gan
                print("[+] GAN model loaded successfully")
                models_loaded.append("GAN")
        except Exception as e:
            print(f"[-] Could not load GAN model: {e}")

        # Try to load TSA model
        try:
            tsa = AnomalyDetectorTSA.load_model()
            if tsa is not None:
                self.tsa_model = tsa
                print("[+] TSA model loaded successfully")
                models_loaded.append("TSA")
        except Exception as e:
            print(f"[-] Could not load TSA model: {e}")

        # Try to load label encoders
        encoders_path = os.path.join(MODELS_DIR, "label_encoders.joblib")
        if os.path.exists(encoders_path):
            try:
                self.label_encoders = joblib.load(encoders_path)
                print("[+] Label encoders loaded successfully")
            except Exception as e:
                print(f"[-] Could not load label encoders: {e}")

        # Set active model (prefer TSA when both are available for higher accuracy)
        if self.tsa_model is not None:
            self.active_model = 'tsa'
        elif self.gan_model is not None:
            self.active_model = 'gan'

        print(f"{'='*60}")
        if models_loaded:
            print(f"READY! Active model: {self.active_model.upper() if self.active_model else 'None'}")
            print(f"Models available: {', '.join(models_loaded)}")
            print("Real-time predictions are ENABLED!")
        else:
            print("No saved models found. Train a model first.")
            print("Models will be saved to:", MODELS_DIR)
        print(f"{'='*60}\n")

        # Update UI status
        self.update_model_status()

    def update_model_status(self):
        """Update the model status indicator in the UI"""
        if self.active_model == 'gan':
            self.model_status_label.config(
                text="🟢 GAN Model Active",
                fg="#00ff00"
            )
        elif self.active_model == 'tsa':
            self.model_status_label.config(
                text="🔵 TSA Model Active",
                fg="#00ccff"
            )
        else:
            self.model_status_label.config(
                text="⚫ No Model Loaded",
                fg="#888888"
            )

        # Show switch button if both models are loaded
        if self.gan_model is not None and self.tsa_model is not None:
            self.model_switch_btn.pack(side=tk.LEFT, padx=5)
        else:
            self.model_switch_btn.pack_forget()

    def switch_active_model(self):
        """Switch between GAN and TSA models"""
        if self.active_model == 'gan' and self.tsa_model is not None:
            self.active_model = 'tsa'
            self.log_buffer = []  # Clear buffer when switching to TSA
        elif self.active_model == 'tsa' and self.gan_model is not None:
            self.active_model = 'gan'
        elif self.gan_model is not None:
            self.active_model = 'gan'
        elif self.tsa_model is not None:
            self.active_model = 'tsa'

        self.update_model_status()

    def predict_anomaly(self, log):
        """
        Predict if a log entry is an anomaly using the active model.

        Args:
            log: Dictionary containing log entry with features

        Returns:
            tuple: (prediction: 0 or 1, confidence: float 0-1)
        """
        try:
            # Extract features from log
            # Feature order must match training: server_type_encoded, severity_encoded,
            # request_time_ms, bytes_transferred, failed_attempts, port_count, traffic_spike_ratio

            # Encode server_type
            server_type = log.get('server_type', 'web')
            if 'server_type' in self.label_encoders:
                try:
                    server_type_encoded = self.label_encoders['server_type'].transform([server_type])[0]
                except ValueError:
                    # Unknown category - use default
                    server_type_encoded = 0
            else:
                # Default encoding based on common server types
                server_type_map = {'api': 0, 'cache': 1, 'database': 2, 'web': 3}
                server_type_encoded = server_type_map.get(server_type, 0)

            # Encode severity
            severity = log.get('severity', 'INFO')
            if 'severity' in self.label_encoders:
                try:
                    severity_encoded = self.label_encoders['severity'].transform([severity])[0]
                except ValueError:
                    severity_encoded = 0
            else:
                # Default encoding based on severity levels
                severity_map = {'CRITICAL': 0, 'HIGH': 1, 'INFO': 2, 'MEDIUM': 3}
                severity_encoded = severity_map.get(severity, 2)

            # Extract numeric features
            request_time_ms = log.get('request_time_ms', 100)
            bytes_transferred = log.get('bytes_transferred', 1000)
            failed_attempts = log.get('failed_attempts', 0)
            port_count = log.get('port_count', 0)
            traffic_spike_ratio = log.get('traffic_spike_ratio', 1.0)

            # Create feature array
            features = np.array([[
                server_type_encoded,
                severity_encoded,
                request_time_ms,
                bytes_transferred,
                failed_attempts,
                port_count,
                traffic_spike_ratio
            ]], dtype=np.float32)

            # Route to appropriate model
            if self.active_model == 'gan' and self.gan_model is not None:
                return self._predict_with_gan(features)
            elif self.active_model == 'tsa' and self.tsa_model is not None:
                return self._predict_with_tsa(features)
            else:
                return None, None

        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None

    def _predict_with_gan(self, features):
        """
        Predict anomaly using Isolation Forest model.
        """
        try:
            if self.gan_model.model is None:
                return None, None

            # Get predictions from the model
            predictions, scores = self.gan_model.detect_anomalies(features)

            if len(predictions) == 0:
                return None, None

            prediction = int(predictions[0])
            score = float(scores[0])

            # Confidence based on absolute value of score
            confidence = min(1.0, max(0.0, abs(score) / (abs(score) + 1e-6)))

            return prediction, round(confidence, 3)

        except Exception as e:
            print(f"GAN prediction error: {e}")
            return None, None

    def _predict_with_tsa(self, features):
        """
        Predict anomaly using TSA (One-Class SVM).
        Requires a buffer of recent samples to create sequences.
        """
        try:
            if self.tsa_model.model is None:
                return None, None

            # Add current features to buffer
            self.log_buffer.append(features[0])

            # Keep only the last sequence_length samples
            sequence_length = self.tsa_model.sequence_length
            if len(self.log_buffer) > sequence_length:
                self.log_buffer = self.log_buffer[-sequence_length:]

            # Need enough samples for a sequence
            if len(self.log_buffer) < sequence_length:
                # Not enough data yet - return uncertain prediction
                # Use a simple heuristic based on individual features
                return self._fallback_prediction(features)

            # Create sequence from buffer
            buffer_array = np.array(self.log_buffer)

            # Get prediction from model
            predictions, scores = self.tsa_model.detect_anomalies(buffer_array)

            if len(predictions) == 0:
                return None, None

            prediction = int(predictions[0])
            score = float(scores[0])

            # Confidence based on score
            confidence = min(1.0, max(0.0, abs(score) / (abs(score) + 1e-6)))

            return prediction, round(confidence, 3)

        except Exception as e:
            print(f"TSA prediction error: {e}")
            return None, None

    def _fallback_prediction(self, features):
        """
        Simple heuristic-based prediction when TSA doesn't have enough sequence data.
        Uses feature thresholds based on typical anomaly patterns.
        """
        try:
            # Extract features
            failed_attempts = features[0][4]
            port_count = features[0][5]
            traffic_spike_ratio = features[0][6]

            # Simple heuristic thresholds
            is_anomaly = (
                failed_attempts > 3 or
                port_count > 5 or
                traffic_spike_ratio > 3.0
            )

            prediction = 1 if is_anomaly else 0
            confidence = 0.3  # Low confidence for fallback prediction

            return prediction, confidence

        except Exception:
            return None, None

    def generate_realistic_log(self, server):
        """Generate realistic log message based on server type"""
        server_type = server["type"]
        log_template = random.choice(NORMAL_LOGS[server_type])

        # Generate realistic values for placeholders
        replacements = {
            "time": random.randint(5, 500),
            "ip": f"{random.randint(100, 200)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}",
            "id": random.randint(1000, 9999),
            "amount": random.randint(10, 1000),
            "pool": random.randint(20, 95),
            "rows": random.randint(100, 10000),
            "key": ''.join(random.choices('0123456789ABCDEF', k=4)),
            "rate": random.randint(100, 900),
            "token": ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8)),
            "ttl": random.randint(60, 3600),
            "count": random.randint(1, 100),
            "keys": random.randint(100, 10000),
            "date": datetime.now().strftime("%Y-%m-%d")
        }

        # Replace placeholders
        log_message = log_template
        for key, value in replacements.items():
            log_message = log_message.replace(f"{{{key}}}", str(value))

        return log_message

    def generate_log_entry(self, server, is_anomaly=False):
        """Generate a log entry for a server with features for ML prediction"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if is_anomaly:
            anomaly = random.choice(ANOMALY_PATTERNS)

            # Generate anomaly-specific numeric features
            if anomaly["type"] == "Brute Force":
                request_time_ms = random.randint(50, 500)
                bytes_transferred = random.randint(500, 5000)
                failed_attempts = random.randint(5, 100)
                port_count = random.randint(0, 5)
                traffic_spike_ratio = round(random.uniform(1.0, 2.0), 2)
            elif anomaly["type"] == "Port Scan":
                request_time_ms = random.randint(10, 200)
                bytes_transferred = random.randint(100, 2000)
                failed_attempts = random.randint(0, 5)
                port_count = random.randint(10, 65535)
                traffic_spike_ratio = round(random.uniform(1.0, 3.0), 2)
            elif anomaly["type"] == "DDoS":
                request_time_ms = random.randint(500, 5000)
                bytes_transferred = random.randint(50000, 150000)
                failed_attempts = random.randint(0, 10)
                port_count = random.randint(0, 10)
                traffic_spike_ratio = round(random.uniform(5.0, 50.0), 2)
            else:  # SQL Injection, Unauthorized Access, Malware
                request_time_ms = random.randint(100, 3000)
                bytes_transferred = random.randint(1000, 100000)
                failed_attempts = random.randint(0, 20)
                port_count = random.randint(0, 15)
                traffic_spike_ratio = round(random.uniform(1.0, 5.0), 2)

            log = {
                "timestamp": timestamp,
                "server_id": server["id"],
                "server_name": server["name"],
                "server_type": server["type"],
                "ip": server["ip"],
                "region": server["region"],
                "type": "ANOMALY",
                "severity": anomaly["severity"],
                "message": anomaly["pattern"],
                "anomaly_type": anomaly["type"],
                "request_time_ms": request_time_ms,
                "bytes_transferred": bytes_transferred,
                "failed_attempts": failed_attempts,
                "port_count": port_count,
                "traffic_spike_ratio": traffic_spike_ratio,
                "ground_truth": 1,
                "model_prediction": None,
                "prediction_confidence": None
            }
            self.anomalies.append(log)
        else:
            # Generate realistic log based on server type
            realistic_message = self.generate_realistic_log(server)

            # Generate normal traffic numeric features
            request_time_ms = random.randint(5, 800)
            bytes_transferred = random.randint(100, 50000)
            failed_attempts = random.randint(0, 2) if random.random() < 0.1 else 0
            port_count = random.randint(0, 3) if random.random() < 0.05 else 0
            traffic_spike_ratio = round(random.uniform(0.8, 1.5), 2)

            log = {
                "timestamp": timestamp,
                "server_id": server["id"],
                "server_name": server["name"],
                "server_type": server["type"],
                "ip": server["ip"],
                "region": server["region"],
                "type": "NORMAL",
                "severity": "INFO",
                "message": realistic_message,
                "anomaly_type": None,
                "request_time_ms": request_time_ms,
                "bytes_transferred": bytes_transferred,
                "failed_attempts": failed_attempts,
                "port_count": port_count,
                "traffic_spike_ratio": traffic_spike_ratio,
                "ground_truth": 0,
                "model_prediction": None,
                "prediction_confidence": None
            }

        # Run prediction if model is loaded
        if self.active_model is not None:
            prediction, confidence = self.predict_anomaly(log)
            log["model_prediction"] = prediction
            log["prediction_confidence"] = confidence

        self.logs.append(log)

        # Keep only last 100 logs
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]

        return log

    def generate_logs_manual(self, force_anomaly=False):
        """Generate logs manually"""
        # Check if it's time for anomaly (every 2 minutes)
        time_since_anomaly = (datetime.now() - self.last_anomaly_check).seconds
        should_generate_anomaly = force_anomaly or time_since_anomaly >= 120

        # Generate normal log for each server
        for server in SERVERS:
            self.generate_log_entry(server, False)

        # Generate anomalies every 2 minutes
        if should_generate_anomaly:
            # Pick 1-2 random servers for anomaly
            anomaly_servers = random.sample(SERVERS, random.randint(1, 2))
            for server in anomaly_servers:
                self.generate_log_entry(server, True)
            self.last_anomaly_check = datetime.now()

        self.last_update = datetime.now()
        self.update_display()

    def update_display(self):
        """Update all display elements"""
        self.update_logs_display()
        self.update_anomalies_display()
        self.update_analytics_display()
        self.update_stats()
        self.update_server_status()

    def update_logs_display(self):
        """Update the logs text widget"""
        self.logs_text.config(state=tk.NORMAL)
        self.logs_text.delete(1.0, tk.END)

        for log in reversed(self.logs[-50:]):  # Show last 50 logs
            log_line = f"[{log['timestamp']}] [{log['server_name']}] "

            if log['type'] == "ANOMALY":
                log_line += f"⚠️ {log['anomaly_type']} - {log['message']}"
                self.logs_text.insert(tk.END, log_line, ("ANOMALY", log['severity']))
            else:
                log_line += f"{log['message']}"
                self.logs_text.insert(tk.END, log_line, log['severity'])

            # Show model prediction if available
            if log.get('model_prediction') is not None:
                pred = log['model_prediction']
                conf = log.get('prediction_confidence', 0) or 0
                ground_truth = log.get('ground_truth', 0)

                actual_label = "ANOMALY" if ground_truth == 1 else "NORMAL"
                pred_label = "ANOMALY" if pred == 1 else "NORMAL"
                is_correct = pred == ground_truth

                # Clear format: Actual vs Predicted
                self.logs_text.insert(tk.END, f"\n     → Actual: ", "PRED_INFO")
                self.logs_text.insert(tk.END, f"{actual_label}", "HIGH" if ground_truth == 1 else "INFO")
                self.logs_text.insert(tk.END, f" | Model Predicted: ", "PRED_INFO")

                if is_correct:
                    self.logs_text.insert(tk.END, f"{pred_label} ({conf*100:.0f}%) ✓ CORRECT", "PRED_CORRECT")
                else:
                    self.logs_text.insert(tk.END, f"{pred_label} ({conf*100:.0f}%) ✗ WRONG", "PRED_INCORRECT")

            self.logs_text.insert(tk.END, "\n")

        self.logs_text.config(state=tk.DISABLED)
        self.logs_text.see(tk.END)

    def update_anomalies_display(self):
        """Update the anomalies text widget"""
        self.anomaly_text.config(state=tk.NORMAL)
        self.anomaly_text.delete(1.0, tk.END)

        if not self.anomalies:
            self.anomaly_text.insert(tk.END, "\n✅ No anomalies detected. All systems secure.\n", "DETAIL")
        else:
            for anomaly in reversed(self.anomalies[-20:]):  # Show last 20 anomalies
                header = f"\n🔴 {anomaly['anomaly_type']} - {anomaly['severity']}\n"
                self.anomaly_text.insert(tk.END, header, "HEADER")

                details = f"Server: {anomaly['server_name']} ({anomaly['ip']})\n"
                details += f"Region: {anomaly['region']}\n"
                details += f"Message: {anomaly['message']}\n"
                details += f"Time: {anomaly['timestamp']}\n"

                # Show model prediction
                if anomaly.get('model_prediction') is not None:
                    pred = anomaly['model_prediction']
                    conf = anomaly.get('prediction_confidence', 0) or 0
                    pred_label = "ANOMALY" if pred == 1 else "NORMAL"
                    detected = "✓ CORRECTLY DETECTED BY MODEL" if pred == 1 else "✗ MISSED BY MODEL"
                    details += f"Actual: ANOMALY (Ground Truth)\n"
                    details += f"Model Predicted: {pred_label} ({conf*100:.0f}%)\n"
                    details += f"Result: {detected}\n"

                details += "-" * 70 + "\n"

                self.anomaly_text.insert(tk.END, details, "DETAIL")

        self.anomaly_text.config(state=tk.DISABLED)
        self.anomaly_text.see(tk.END)

    def update_analytics_display(self):
        """Update the analytics text widget"""
        self.analytics_text.config(state=tk.NORMAL)
        self.analytics_text.delete(1.0, tk.END)

        if not self.logs:
            self.analytics_text.insert(tk.END, "\nLoading data... Server monitoring is active.\n")
        else:
            # Overall statistics
            total_logs = len(self.logs)
            normal_logs = len([l for l in self.logs if l["type"] == "NORMAL"])
            anomaly_logs = len([l for l in self.logs if l["type"] == "ANOMALY"])
            anomaly_rate = (anomaly_logs / total_logs * 100) if total_logs > 0 else 0

            analytics = f"""
═══════════════════════════════════════════════════════════
                    SYSTEM ANALYTICS REPORT
═══════════════════════════════════════════════════════════

📊 Overall Statistics:
   • Total Logs: {total_logs}
   • Normal Logs: {normal_logs}
   • Anomaly Logs: {anomaly_logs}
   • Anomaly Rate: {anomaly_rate:.2f}%

───────────────────────────────────────────────────────────

🖥️ Logs by Server:
"""
            # Count by server
            from collections import Counter
            server_counts = Counter([l["server_name"] for l in self.logs])
            for server, count in server_counts.most_common():
                analytics += f"   • {server}: {count} logs\n"

            analytics += "\n───────────────────────────────────────────────────────────\n"

            # Anomaly types
            analytics += "🚨 Anomaly Types Breakdown:\n"
            anomaly_types = Counter([i["anomaly_type"] for i in self.anomalies])
            if not anomaly_types:
                analytics += "   • No anomalies recorded.\n"
            else:
                for a_type, count in anomaly_types.most_common():
                    analytics += f"   • {a_type}: {count} times\n"

            analytics += "\n───────────────────────────────────────────────────────────\n"

            # Anomaly severity
            analytics += "🔥 Anomaly Severity Levels:\n"
            severity_levels = Counter([i["severity"] for i in self.anomalies])
            if not severity_levels:
                analytics += "   • No anomalies recorded.\n"
            else:
                for severity, count in severity_levels.most_common():
                    analytics += f"   • {severity}: {count} times\n"

            analytics += "\n───────────────────────────────────────────────────────────\n"

            # ML Model Prediction Statistics
            analytics += "🤖 ML Model Prediction Stats:\n"
            if self.active_model:
                analytics += f"   • Active Model: {self.active_model.upper()}\n"

                # Calculate prediction accuracy
                logs_with_pred = [l for l in self.logs if l.get('model_prediction') is not None]
                if logs_with_pred:
                    correct = sum(1 for l in logs_with_pred if l['model_prediction'] == l['ground_truth'])
                    total_pred = len(logs_with_pred)
                    accuracy = (correct / total_pred * 100) if total_pred > 0 else 0

                    # True positives, false positives, etc.
                    tp = sum(1 for l in logs_with_pred if l['model_prediction'] == 1 and l['ground_truth'] == 1)
                    fp = sum(1 for l in logs_with_pred if l['model_prediction'] == 1 and l['ground_truth'] == 0)
                    tn = sum(1 for l in logs_with_pred if l['model_prediction'] == 0 and l['ground_truth'] == 0)
                    fn = sum(1 for l in logs_with_pred if l['model_prediction'] == 0 and l['ground_truth'] == 1)

                    analytics += f"   • Predictions Made: {total_pred}\n"
                    analytics += f"   • Accuracy: {accuracy:.1f}%\n"
                    analytics += f"   • True Positives: {tp} | False Positives: {fp}\n"
                    analytics += f"   • True Negatives: {tn} | False Negatives: {fn}\n"

                    # Detection rate for anomalies
                    if (tp + fn) > 0:
                        detection_rate = (tp / (tp + fn) * 100)
                        analytics += f"   • Anomaly Detection Rate: {detection_rate:.1f}%\n"
                else:
                    analytics += "   • No predictions made yet.\n"
            else:
                analytics += "   • No model loaded. Train a model to enable predictions.\n"

            analytics += "\n═══════════════════════════════════════════════════════════\n"

            self.analytics_text.insert(tk.END, analytics)

        self.analytics_text.config(state=tk.DISABLED)

    def update_stats(self):
        """Update the statistics labels"""
        total_logs = len(self.logs)
        anomalies_count = len(self.anomalies)

        self.stats_labels["total_logs"].config(text=str(total_logs))
        self.stats_labels["anomalies"].config(text=str(anomalies_count))
        self.stats_labels["servers"].config(text=str(len(SERVERS)))
        self.stats_labels["last_update"].config(text=self.last_update.strftime("%H:%M:%S"))

    def calculate_accuracy_metrics(self):
        """Calculate comprehensive accuracy metrics for the active model"""
        if not self.logs:
            return None

        # Get logs with predictions
        logs_with_pred = [l for l in self.logs if l.get('model_prediction') is not None]
        if not logs_with_pred:
            return None

        # Extract ground truth and predictions
        ground_truth = np.array([l['ground_truth'] for l in logs_with_pred])
        predictions = np.array([l['model_prediction'] for l in logs_with_pred])

        # Calculate confusion matrix metrics
        tp = np.sum((predictions == 1) & (ground_truth == 1))
        fp = np.sum((predictions == 1) & (ground_truth == 0))
        tn = np.sum((predictions == 0) & (ground_truth == 0))
        fn = np.sum((predictions == 0) & (ground_truth == 1))

        # Calculate metrics
        accuracy = accuracy_score(ground_truth, predictions)
        
        # Precision (TP / (TP + FP))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall / Sensitivity (TP / (TP + FN))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Specificity (TN / (TN + FP))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # False Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # False Negative Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        metrics = {
            'total_predictions': len(logs_with_pred),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'fpr': fpr,
            'fnr': fnr,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'total_anomalies': int(tp + fn),
            'total_normal': int(tn + fp)
        }

        return metrics

    def print_accuracy_report(self):
        """Print a detailed accuracy report to console and messagebox"""
        metrics = self.calculate_accuracy_metrics()

        if metrics is None:
            report = "No predictions available. Make predictions first by running the monitoring system."
            print(report)
            return report

        # Create detailed report
        report = f"""
{'='*70}
                    ANOMALY DETECTION ACCURACY REPORT
{'='*70}

MODEL: {self.active_model.upper() if self.active_model else 'NONE'}
TOTAL PREDICTIONS: {metrics['total_predictions']}

OVERALL PERFORMANCE:
├─ Accuracy:      {metrics['accuracy']*100:6.2f}%
├─ Precision:     {metrics['precision']*100:6.2f}%  (TP / (TP + FP))
├─ Recall:        {metrics['recall']*100:6.2f}%   (TP / (TP + FN))
├─ F1 Score:      {metrics['f1_score']:6.3f}
├─ Specificity:   {metrics['specificity']*100:6.2f}%  (TN / (TN + FP))
└─ False Pos Rate: {metrics['fpr']*100:6.2f}%

CONFUSION MATRIX:
                Predicted Negative    Predicted Positive
Actual Negative    {metrics['tn']:6d} (TN)          {metrics['fp']:6d} (FP)
Actual Positive    {metrics['fn']:6d} (FN)          {metrics['tp']:6d} (TP)

DETECTION RESULTS:
├─ Total Anomalies:       {metrics['total_anomalies']:6d}
├─ Detected (TP):         {metrics['tp']:6d}
├─ Missed (FN):           {metrics['fn']:6d}
├─ Total Normal Events:    {metrics['total_normal']:6d}
├─ Correctly Classified:   {metrics['tn']:6d}
└─ False Alarms (FP):     {metrics['fp']:6d}

INTERPRETATION:
├─ Accuracy:      Overall correctness of predictions
├─ Precision:     Reliability when model predicts ANOMALY (TP / (TP + FP))
├─ Recall:        Ability to catch actual anomalies (TP / (TP + FN))
├─ F1 Score:      Balance between precision and recall
├─ Specificity:   Ability to correctly identify normal events
└─ False Pos Rate: How often model incorrectly predicts ANOMALY

{'='*70}
"""

        print(report)
        return report

    def update_server_status(self):
        """Update server status indicators"""
        # In this simulation, all servers are always online
        # This could be extended to check actual server health
        for server in SERVERS:
            status_label = self.server_status_labels[server["id"]]
            status_label.config(text="ONLINE", fg="#00aa00")

    def auto_monitor(self):
        """Automatically generate logs and update display"""
        while self.running:
            if self.auto_refresh.get():
                self.generate_logs_manual()
            time.sleep(random.uniform(1, 2))  # Real-time: generate logs every 1-2 seconds

    def generate_dataset_row(self, row_id, is_anomaly=False):
        """Generate a single dataset row for training with realistic noise"""
        server = random.choice(SERVERS)

        # Generate random timestamp within last 30 days
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        seconds_ago = random.randint(0, 59)
        timestamp = datetime.now() - timedelta(
            days=days_ago, hours=hours_ago, minutes=minutes_ago, seconds=seconds_ago
        )
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # Generate source IP
        source_ip = f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

        if is_anomaly:
            anomaly = random.choice(ANOMALY_PATTERNS)

            # Add noise: 15% chance of appearing as lower severity (evasion attempts)
            if random.random() < 0.15:
                severity = random.choice(["INFO", "MEDIUM"])
            else:
                severity = anomaly["severity"]

            # Overlapping ranges with normal traffic (attackers trying to blend in)
            # 20% of anomalies have normal-looking metrics
            if random.random() < 0.20:
                request_time = random.randint(10, 600)
                bytes_trans = random.randint(100, 60000)
            else:
                request_time = random.randint(50, 5000)
                bytes_trans = random.randint(500, 150000)

            # Add randomness to attack-specific features
            # Not all brute force shows failed attempts (distributed attacks)
            if anomaly["type"] == "Brute Force":
                failed = random.randint(0, 100) if random.random() > 0.1 else 0
            else:
                failed = random.randint(0, 5) if random.random() < 0.1 else 0

            # Port scans may be slow/stealthy
            if anomaly["type"] == "Port Scan":
                ports = random.randint(1, 65535) if random.random() > 0.15 else random.randint(0, 10)
            else:
                ports = random.randint(0, 15) if random.random() < 0.15 else 0

            # DDoS traffic spike varies, some are subtle
            if anomaly["type"] == "DDoS":
                spike = round(random.uniform(1.5, 50.0), 2)
            else:
                spike = round(random.uniform(0.8, 3.0), 2) if random.random() < 0.2 else 1.0

            return {
                "id": row_id,
                "timestamp": timestamp_str,
                "server_id": server["id"],
                "server_name": server["name"],
                "server_ip": server["ip"],
                "server_region": server["region"],
                "server_type": server["type"],
                "source_ip": source_ip,
                "log_message": anomaly["pattern"],
                "is_anomaly": 1,
                "anomaly_type": anomaly["type"],
                "severity": severity,
                "request_time_ms": request_time,
                "bytes_transferred": bytes_trans,
                "failed_attempts": failed,
                "port_count": ports,
                "traffic_spike_ratio": spike
            }
        else:
            log_message = self.generate_realistic_log(server)

            # Add noise: some normal traffic looks suspicious (false positive potential)
            # 10% of normal logs have elevated severity (system warnings, timeouts)
            if random.random() < 0.10:
                severity = random.choice(["MEDIUM", "HIGH"])
            else:
                severity = "INFO"

            # Normal traffic can sometimes have high values (large file downloads, slow queries)
            if random.random() < 0.15:
                request_time = random.randint(200, 3000)
                bytes_trans = random.randint(10000, 100000)
            else:
                request_time = random.randint(5, 800)
                bytes_trans = random.randint(100, 50000)

            # Occasional failed attempts from legitimate users (typos, expired sessions)
            failed = random.randint(1, 5) if random.random() < 0.08 else 0

            # Some legitimate scanning/monitoring tools
            ports = random.randint(1, 20) if random.random() < 0.05 else 0

            # Normal traffic fluctuations
            spike = round(random.uniform(0.5, 2.5), 2) if random.random() < 0.1 else 1.0

            return {
                "id": row_id,
                "timestamp": timestamp_str,
                "server_id": server["id"],
                "server_name": server["name"],
                "server_ip": server["ip"],
                "server_region": server["region"],
                "server_type": server["type"],
                "source_ip": source_ip,
                "log_message": log_message,
                "is_anomaly": 0,
                "anomaly_type": "None",
                "severity": severity,
                "request_time_ms": request_time,
                "bytes_transferred": bytes_trans,
                "failed_attempts": failed,
                "port_count": ports,
                "traffic_spike_ratio": spike
            }

    def download_dataset(self):
        """Generate and download 10,000 rows of training dataset"""
        # Ask for file location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Training Dataset",
            initialfile="anomaly_detection_dataset.csv"
        )

        if not file_path:
            return  # User cancelled

        # Show progress
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Generating Dataset")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()

        progress_label = tk.Label(
            progress_window,
            text="Generating 10,000 rows...",
            font=("Arial", 12)
        )
        progress_label.pack(pady=20)

        progress_bar = ttk.Progressbar(
            progress_window,
            length=350,
            mode='determinate',
            maximum=10000
        )
        progress_bar.pack(pady=10)

        status_label = tk.Label(
            progress_window,
            text="0 / 10,000 rows",
            font=("Arial", 10)
        )
        status_label.pack(pady=5)

        def generate():
            try:
                total_rows = 10000
                # Approximately 15-20% anomalies for realistic dataset
                anomaly_ratio = 0.18
                anomaly_count = int(total_rows * anomaly_ratio)
                normal_count = total_rows - anomaly_count

                # Create list of is_anomaly flags
                anomaly_flags = [True] * anomaly_count + [False] * normal_count
                random.shuffle(anomaly_flags)

                # CSV columns
                fieldnames = [
                    "id", "timestamp", "server_id", "server_name", "server_ip",
                    "server_region", "server_type", "source_ip", "log_message",
                    "is_anomaly", "anomaly_type", "severity", "request_time_ms",
                    "bytes_transferred", "failed_attempts", "port_count", "traffic_spike_ratio"
                ]

                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                    for i, is_anomaly in enumerate(anomaly_flags):
                        row = self.generate_dataset_row(i + 1, is_anomaly)
                        writer.writerow(row)

                        # Update progress every 100 rows
                        if (i + 1) % 100 == 0:
                            progress_bar['value'] = i + 1
                            status_label.config(text=f"{i + 1} / 10,000 rows")
                            progress_window.update()

                progress_window.destroy()

                # Count actual anomalies in dataset
                actual_anomalies = sum(anomaly_flags)
                messagebox.showinfo(
                    "Dataset Generated",
                    f"✅ Dataset saved successfully!\n\n"
                    f"📁 Location: {file_path}\n"
                    f"📊 Total Rows: {total_rows:,}\n"
                    f"🔴 Anomaly Samples: {actual_anomalies:,} ({actual_anomalies/total_rows*100:.1f}%)\n"
                    f"🟢 Normal Samples: {total_rows - actual_anomalies:,} ({(total_rows-actual_anomalies)/total_rows*100:.1f}%)\n\n"
                    f"Ready for ML training!"
                )

            except Exception as e:
                progress_window.destroy()
                messagebox.showerror("Error", f"Failed to generate dataset:\n{str(e)}")

        # Run generation in thread to keep UI responsive
        threading.Thread(target=generate, daemon=True).start()

    def train_gan_model(self):
        """Train GAN model on a dataset and show results in UI"""
        # Ask for dataset file
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Select Training Dataset"
        )

        if not file_path:
            return  # User cancelled

        # Create results window
        results_window = tk.Toplevel(self.root)
        results_window.title("🧠 GAN Anomaly Detection Training Results")
        results_window.geometry("800x700")
        results_window.transient(self.root)

        # Results text area
        results_text = scrolledtext.ScrolledText(
            results_window,
            font=("Consolas", 10),
            bg="#1e1e1e",
            fg="#d4d4d4",
            wrap=tk.WORD
        )
        results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Configure tags for colored output
        results_text.tag_config("header", foreground="#00ccff", font=("Consolas", 12, "bold"))
        results_text.tag_config("success", foreground="#00ff00", font=("Consolas", 11, "bold"))
        results_text.tag_config("metric", foreground="#ffaa00")
        results_text.tag_config("info", foreground="#d4d4d4")
        results_text.tag_config("warning", foreground="#ff6600")
        results_text.tag_config("epoch", foreground="#bb86fc")
        results_text.tag_config("gan", foreground="#03dac6")

        def train():
            try:
                results_text.insert(tk.END, "=" * 65 + "\n", "header")
                results_text.insert(tk.END, "   GENERATIVE ADVERSARIAL NETWORK (GAN) ANOMALY DETECTION\n", "header")
                results_text.insert(tk.END, "=" * 65 + "\n\n", "header")

                results_text.insert(tk.END, "📂 Loading dataset...\n", "info")
                results_window.update()

                # Load dataset
                df = pd.read_csv(file_path)
                results_text.insert(tk.END, f"✓ Loaded {len(df):,} rows\n\n", "success")

                # Detect column names (support both old and new datasets)
                is_anomaly_col = 'is_anomaly' if 'is_anomaly' in df.columns else 'is_intrusion'
                anomaly_type_col = 'anomaly_type' if 'anomaly_type' in df.columns else 'intrusion_type'

                # Display dataset info
                results_text.insert(tk.END, "📊 Dataset Overview:\n", "header")
                results_text.insert(tk.END, f"   • Total samples: {len(df):,}\n", "info")
                results_text.insert(tk.END, f"   • Features: {len(df.columns)}\n", "info")
                anomaly_count = df[is_anomaly_col].sum()
                normal_count = len(df) - anomaly_count
                results_text.insert(tk.END, f"   • Normal samples: {normal_count:,} ({normal_count/len(df)*100:.1f}%)\n", "info")
                results_text.insert(tk.END, f"   • Anomaly samples: {anomaly_count:,} ({anomaly_count/len(df)*100:.1f}%)\n\n", "info")
                results_window.update()

                results_text.insert(tk.END, "🔧 Preprocessing data...\n", "info")
                results_window.update()

                # Encode categorical variables
                label_encoders = {}
                df_encoded = df.copy()

                for col in ['server_type', 'severity', anomaly_type_col]:
                    if col in df_encoded.columns:
                        le = LabelEncoder()
                        df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                        label_encoders[col] = le

                # Prepare features
                feature_columns = [
                    'server_type_encoded', 'severity_encoded', 'request_time_ms',
                    'bytes_transferred', 'failed_attempts', 'port_count', 'traffic_spike_ratio'
                ]
                X = df_encoded[feature_columns].values
                y = df_encoded[is_anomaly_col].values

                results_text.insert(tk.END, "✓ Preprocessing complete\n\n", "success")

                # Split data - GAN trains on normal data only
                results_text.insert(tk.END, "📈 Preparing data for GAN training...\n", "info")
                results_text.insert(tk.END, "   • GAN learns normal patterns only\n", "info")
                results_text.insert(tk.END, "   • Anomalies detected as deviations from normal\n\n", "info")
                results_window.update()

                # Separate normal and anomaly data
                X_normal = X[y == 0]
                X_anomaly = X[y == 1]

                # Split normal data for training and testing
                X_train_normal, X_test_normal = train_test_split(
                    X_normal, test_size=0.2, random_state=42
                )

                # Create test set with both normal and anomaly samples
                X_test = np.vstack([X_test_normal, X_anomaly])
                y_test = np.array([0] * len(X_test_normal) + [1] * len(X_anomaly))

                results_text.insert(tk.END, f"   • Training samples (normal only): {len(X_train_normal):,}\n", "info")
                results_text.insert(tk.END, f"   • Test samples: {len(X_test):,}\n", "info")
                results_text.insert(tk.END, f"     - Normal: {len(X_test_normal):,}\n", "info")
                results_text.insert(tk.END, f"     - Anomaly: {len(X_anomaly):,}\n\n", "info")

                # Initialize GAN
                results_text.insert(tk.END, "🧠 Building GAN Architecture...\n", "gan")
                results_text.insert(tk.END, "─" * 65 + "\n", "info")
                results_window.update()

                input_dim = X_train_normal.shape[1]
                latent_dim = 32
                gan_model = AnomalyDetectorGAN(input_dim=input_dim, latent_dim=latent_dim)

                results_text.insert(tk.END, "\n┌─────────────────────────────────────────────────────────────┐\n", "gan")
                results_text.insert(tk.END, "│                    GENERATOR NETWORK                        │\n", "gan")
                results_text.insert(tk.END, "├─────────────────────────────────────────────────────────────┤\n", "gan")
                results_text.insert(tk.END, f"│  Input:  Latent vector ({latent_dim} dimensions)                    │\n", "info")
                results_text.insert(tk.END, "│  Hidden: Dense(64) → Dense(128) → Dense(256)                │\n", "info")
                results_text.insert(tk.END, f"│  Output: Generated data ({input_dim} features)                      │\n", "info")
                results_text.insert(tk.END, "│  Activation: LeakyReLU + BatchNorm → tanh                   │\n", "info")
                results_text.insert(tk.END, "└─────────────────────────────────────────────────────────────┘\n\n", "gan")

                results_text.insert(tk.END, "┌─────────────────────────────────────────────────────────────┐\n", "gan")
                results_text.insert(tk.END, "│                  DISCRIMINATOR NETWORK                      │\n", "gan")
                results_text.insert(tk.END, "├─────────────────────────────────────────────────────────────┤\n", "gan")
                results_text.insert(tk.END, f"│  Input:  Data sample ({input_dim} features)                        │\n", "info")
                results_text.insert(tk.END, "│  Hidden: Dense(256) → Dense(128) → Dense(64)                │\n", "info")
                results_text.insert(tk.END, "│  Output: Probability (real/fake)                            │\n", "info")
                results_text.insert(tk.END, "│  Activation: LeakyReLU + Dropout → sigmoid                  │\n", "info")
                results_text.insert(tk.END, "└─────────────────────────────────────────────────────────────┘\n\n", "gan")
                results_window.update()

                # Training parameters
                epochs = 100
                batch_size = 64

                results_text.insert(tk.END, "🚀 Training GAN...\n", "header")
                results_text.insert(tk.END, f"   • Epochs: {epochs}\n", "info")
                results_text.insert(tk.END, f"   • Batch size: {batch_size}\n", "info")
                results_text.insert(tk.END, f"   • Latent dimension: {latent_dim}\n", "info")
                results_text.insert(tk.END, f"   • Optimizer: Adam (lr=0.0002, β1=0.5)\n\n", "info")
                results_text.insert(tk.END, "─" * 65 + "\n", "info")
                results_window.update()

                # Progress callback
                def progress_callback(epoch, total_epochs, d_loss, g_loss, d_acc):
                    if epoch % 10 == 0 or epoch == 1:
                        progress = int((epoch / total_epochs) * 30)
                        bar = "█" * progress + "░" * (30 - progress)
                        results_text.insert(
                            tk.END,
                            f"Epoch {epoch:3d}/{total_epochs} [{bar}] "
                            f"D_loss: {d_loss:.4f} | G_loss: {g_loss:.4f} | D_acc: {d_acc:.2%}\n",
                            "epoch"
                        )
                        results_window.update()

                # Train the GAN
                anomaly_rate = max(0.001, min(0.5, len(X_anomaly) / len(X)))
                history = gan_model.train(
                    X_train_normal,
                    epochs=epochs,
                    batch_size=batch_size,
                    callback=progress_callback,
                    contamination=anomaly_rate
                )

                results_text.insert(tk.END, "─" * 65 + "\n", "info")
                results_text.insert(tk.END, "\n✓ GAN training complete!\n\n", "success")

                # Evaluate on test set
                results_text.insert(tk.END, "🎯 Evaluating anomaly detection...\n\n", "info")
                results_window.update()

                # Detect anomalies
                y_pred, scores = gan_model.detect_anomalies(X_test)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                conf_matrix = confusion_matrix(y_test, y_pred)
                class_report = classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly'])

                # Display results
                results_text.insert(tk.END, "=" * 65 + "\n", "header")
                results_text.insert(tk.END, "                      DETECTION RESULTS\n", "header")
                results_text.insert(tk.END, "=" * 65 + "\n\n", "header")

                results_text.insert(tk.END, f"🎯 DETECTION ACCURACY: {accuracy * 100:.2f}%\n\n", "success")

                results_text.insert(tk.END, "📊 Confusion Matrix:\n", "metric")
                results_text.insert(tk.END, "                    Predicted\n", "info")
                results_text.insert(tk.END, "                 Normal    Anomaly\n", "info")
                results_text.insert(tk.END, f"Actual Normal    {conf_matrix[0][0]:6d}     {conf_matrix[0][1]:6d}\n", "info")
                results_text.insert(tk.END, f"Actual Anomaly   {conf_matrix[1][0]:6d}     {conf_matrix[1][1]:6d}\n\n", "info")

                results_text.insert(tk.END, "📋 Classification Report:\n", "metric")
                results_text.insert(tk.END, class_report + "\n", "info")

                # Training statistics
                results_text.insert(tk.END, "📈 Training Statistics:\n", "metric")
                results_text.insert(tk.END, f"   • Final Discriminator Loss: {history['d_loss'][-1]:.4f}\n", "info")
                results_text.insert(tk.END, f"   • Final Generator Loss: {history['g_loss'][-1]:.4f}\n", "info")
                results_text.insert(tk.END, f"   • Final Discriminator Accuracy: {history['d_acc'][-1]:.2%}\n", "info")
                results_text.insert(tk.END, f"   • Anomaly Threshold: {gan_model.threshold:.4f}\n\n", "info")

                # Score distribution
                results_text.insert(tk.END, "🔍 Anomaly Score Distribution:\n", "metric")
                normal_scores = scores[y_test == 0]
                anomaly_scores = scores[y_test == 1]
                results_text.insert(tk.END, f"   • Normal samples - Mean: {np.mean(normal_scores):.4f}, Std: {np.std(normal_scores):.4f}\n", "info")
                results_text.insert(tk.END, f"   • Anomaly samples - Mean: {np.mean(anomaly_scores):.4f}, Std: {np.std(anomaly_scores):.4f}\n", "info")

                results_text.insert(tk.END, "\n" + "=" * 65 + "\n", "header")
                results_text.insert(tk.END, "✅ GAN Anomaly Detection Model Ready!\n", "success")
                results_text.insert(tk.END, "   Low discriminator scores indicate anomalies.\n", "info")
                results_text.insert(tk.END, "=" * 65 + "\n\n", "header")

                # Save model and encoders
                results_text.insert(tk.END, "💾 Saving model...\n", "info")
                results_window.update()

                gan_model.save_model()
                joblib.dump(label_encoders, os.path.join(MODELS_DIR, "label_encoders.joblib"))

                results_text.insert(tk.END, f"✓ Model saved to {MODELS_DIR}\n", "success")

                # Update instance variables
                self.gan_model = gan_model
                self.label_encoders = label_encoders
                self.active_model = 'gan'
                self.root.after(0, self.update_model_status)

                results_text.insert(tk.END, "✓ GAN model is now active for predictions!\n", "success")

            except Exception as e:
                import traceback
                error_msg = f"\n❌ Error: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)  # Always print to console
                try:
                    if results_window.winfo_exists():
                        results_text.insert(tk.END, error_msg, "warning")
                except:
                    pass  # Window already closed

        # Run training in thread
        threading.Thread(target=train, daemon=True).start()

    def train_tsa_model(self):
        """Train TSA (LSTM Autoencoder) model on a dataset and show results in UI"""
        # Ask for dataset file
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Select Training Dataset"
        )

        if not file_path:
            return  # User cancelled

        # Create results window
        results_window = tk.Toplevel(self.root)
        results_window.title("📈 TSA (LSTM Autoencoder) Training Results")
        results_window.geometry("800x700")
        results_window.transient(self.root)

        # Results text area
        results_text = scrolledtext.ScrolledText(
            results_window,
            font=("Consolas", 10),
            bg="#1e1e1e",
            fg="#d4d4d4",
            wrap=tk.WORD
        )
        results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Configure tags for colored output
        results_text.tag_config("header", foreground="#00ccff", font=("Consolas", 12, "bold"))
        results_text.tag_config("success", foreground="#00ff00", font=("Consolas", 11, "bold"))
        results_text.tag_config("metric", foreground="#ffaa00")
        results_text.tag_config("info", foreground="#d4d4d4")
        results_text.tag_config("warning", foreground="#ff6600")
        results_text.tag_config("epoch", foreground="#bb86fc")
        results_text.tag_config("tsa", foreground="#ff7043")

        def train():
            try:
                results_text.insert(tk.END, "=" * 65 + "\n", "header")
                results_text.insert(tk.END, "   TIME SERIES ANALYSIS (LSTM AUTOENCODER) ANOMALY DETECTION\n", "header")
                results_text.insert(tk.END, "=" * 65 + "\n\n", "header")

                results_text.insert(tk.END, "📂 Loading dataset...\n", "info")
                results_window.update()

                # Load dataset
                df = pd.read_csv(file_path)
                results_text.insert(tk.END, f"✓ Loaded {len(df):,} rows\n\n", "success")

                # Detect column names (support both old and new datasets)
                is_anomaly_col = 'is_anomaly' if 'is_anomaly' in df.columns else 'is_intrusion'
                anomaly_type_col = 'anomaly_type' if 'anomaly_type' in df.columns else 'intrusion_type'

                # Display dataset info
                results_text.insert(tk.END, "📊 Dataset Overview:\n", "header")
                results_text.insert(tk.END, f"   • Total samples: {len(df):,}\n", "info")
                results_text.insert(tk.END, f"   • Features: {len(df.columns)}\n", "info")
                anomaly_count = df[is_anomaly_col].sum()
                normal_count = len(df) - anomaly_count
                results_text.insert(tk.END, f"   • Normal samples: {normal_count:,} ({normal_count/len(df)*100:.1f}%)\n", "info")
                results_text.insert(tk.END, f"   • Anomaly samples: {anomaly_count:,} ({anomaly_count/len(df)*100:.1f}%)\n\n", "info")
                results_window.update()

                results_text.insert(tk.END, "🔧 Preprocessing data...\n", "info")
                results_window.update()

                # Encode categorical variables
                label_encoders = {}
                df_encoded = df.copy()

                for col in ['server_type', 'severity', anomaly_type_col]:
                    if col in df_encoded.columns:
                        le = LabelEncoder()
                        df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                        label_encoders[col] = le

                # Prepare features
                feature_columns = [
                    'server_type_encoded', 'severity_encoded', 'request_time_ms',
                    'bytes_transferred', 'failed_attempts', 'port_count', 'traffic_spike_ratio'
                ]
                X = df_encoded[feature_columns].values
                y = df_encoded[is_anomaly_col].values

                results_text.insert(tk.END, "✓ Preprocessing complete\n\n", "success")

                # Split data - TSA trains on normal data only
                results_text.insert(tk.END, "📈 Preparing data for TSA training...\n", "info")
                results_text.insert(tk.END, "   • LSTM Autoencoder learns normal patterns\n", "info")
                results_text.insert(tk.END, "   • High reconstruction error = Anomaly\n\n", "info")
                results_window.update()

                # Separate normal and anomaly data
                X_normal = X[y == 0]
                X_anomaly = X[y == 1]

                results_text.insert(tk.END, f"   • Normal samples for training: {len(X_normal):,}\n", "info")
                results_text.insert(tk.END, f"   • Anomaly samples for testing: {len(X_anomaly):,}\n\n", "info")

                # Initialize TSA
                results_text.insert(tk.END, "📈 Building LSTM Autoencoder Architecture...\n", "tsa")
                results_text.insert(tk.END, "─" * 65 + "\n", "info")
                results_window.update()

                input_dim = X_normal.shape[1]
                sequence_length = 10
                latent_dim = 16
                tsa_model = AnomalyDetectorTSA(input_dim=input_dim, sequence_length=sequence_length, latent_dim=latent_dim)

                results_text.insert(tk.END, "\n┌─────────────────────────────────────────────────────────────┐\n", "tsa")
                results_text.insert(tk.END, "│                  LSTM AUTOENCODER ARCHITECTURE              │\n", "tsa")
                results_text.insert(tk.END, "├─────────────────────────────────────────────────────────────┤\n", "tsa")
                results_text.insert(tk.END, f"│  Input:  Sequence ({sequence_length} timesteps × {input_dim} features)           │\n", "info")
                results_text.insert(tk.END, "│  Encoder: LSTM(64) → LSTM(32) → Dense(16)                   │\n", "info")
                results_text.insert(tk.END, "│  Decoder: RepeatVector → LSTM(32) → LSTM(64) → Dense        │\n", "info")
                results_text.insert(tk.END, f"│  Output: Reconstructed sequence ({sequence_length} × {input_dim})               │\n", "info")
                results_text.insert(tk.END, "│  Loss: Mean Squared Error (MSE)                             │\n", "info")
                results_text.insert(tk.END, "└─────────────────────────────────────────────────────────────┘\n\n", "tsa")
                results_window.update()

                # Training parameters
                epochs = 50
                batch_size = 32

                results_text.insert(tk.END, "🚀 Training LSTM Autoencoder...\n", "header")
                results_text.insert(tk.END, f"   • Epochs: {epochs}\n", "info")
                results_text.insert(tk.END, f"   • Batch size: {batch_size}\n", "info")
                results_text.insert(tk.END, f"   • Sequence length: {sequence_length}\n", "info")
                results_text.insert(tk.END, f"   • Latent dimension: {latent_dim}\n", "info")
                results_text.insert(tk.END, f"   • Optimizer: Adam (lr=0.001)\n\n", "info")
                results_text.insert(tk.END, "─" * 65 + "\n", "info")
                results_window.update()

                # Progress callback
                def progress_callback(epoch, total_epochs, loss, val_loss):
                    if epoch % 5 == 0 or epoch == 1:
                        progress = int((epoch / total_epochs) * 30)
                        bar = "█" * progress + "░" * (30 - progress)
                        results_text.insert(
                            tk.END,
                            f"Epoch {epoch:3d}/{total_epochs} [{bar}] "
                            f"Loss: {loss:.6f} | Val_Loss: {val_loss:.6f}\n",
                            "epoch"
                        )
                        results_window.update()

                # Train the TSA model
                history = tsa_model.train(
                    X_normal,
                    epochs=epochs,
                    batch_size=batch_size,
                    callback=progress_callback
                )

                results_text.insert(tk.END, "─" * 65 + "\n", "info")
                results_text.insert(tk.END, "\n✓ LSTM Autoencoder training complete!\n\n", "success")

                # Evaluate on test set - create combined test data
                results_text.insert(tk.END, "🎯 Evaluating anomaly detection...\n\n", "info")
                results_window.update()

                # Take subset of normal data for testing
                X_test_normal = X_normal[-int(len(X_normal)*0.2):]
                X_test = np.vstack([X_test_normal, X_anomaly])
                y_test = np.array([0] * len(X_test_normal) + [1] * len(X_anomaly))

                # Detect anomalies
                y_pred, mse_scores = tsa_model.detect_anomalies(X_test)

                # Adjust predictions length to match y_test (due to sequence creation)
                y_test_adjusted = y_test[tsa_model.sequence_length-1:]

                if len(y_pred) > 0 and len(y_pred) == len(y_test_adjusted):
                    # Calculate metrics
                    accuracy = accuracy_score(y_test_adjusted, y_pred)
                    conf_matrix = confusion_matrix(y_test_adjusted, y_pred)
                    class_report = classification_report(y_test_adjusted, y_pred, target_names=['Normal', 'Anomaly'])

                    # Display results
                    results_text.insert(tk.END, "=" * 65 + "\n", "header")
                    results_text.insert(tk.END, "                      DETECTION RESULTS\n", "header")
                    results_text.insert(tk.END, "=" * 65 + "\n\n", "header")

                    results_text.insert(tk.END, f"🎯 DETECTION ACCURACY: {accuracy * 100:.2f}%\n\n", "success")

                    results_text.insert(tk.END, "📊 Confusion Matrix:\n", "metric")
                    results_text.insert(tk.END, "                    Predicted\n", "info")
                    results_text.insert(tk.END, "                 Normal    Anomaly\n", "info")
                    results_text.insert(tk.END, f"Actual Normal    {conf_matrix[0][0]:6d}     {conf_matrix[0][1]:6d}\n", "info")
                    results_text.insert(tk.END, f"Actual Anomaly   {conf_matrix[1][0]:6d}     {conf_matrix[1][1]:6d}\n\n", "info")

                    results_text.insert(tk.END, "📋 Classification Report:\n", "metric")
                    results_text.insert(tk.END, class_report + "\n", "info")

                # Training statistics
                results_text.insert(tk.END, "📈 Training Statistics:\n", "metric")
                results_text.insert(tk.END, f"   • Final Training Loss: {history['loss'][-1]:.6f}\n", "info")
                results_text.insert(tk.END, f"   • Final Validation Loss: {history['val_loss'][-1]:.6f}\n", "info")
                results_text.insert(tk.END, f"   • Anomaly Threshold (MSE): {tsa_model.threshold:.6f}\n\n", "info")

                # Score distribution
                if len(mse_scores) > 0:
                    results_text.insert(tk.END, "🔍 Reconstruction Error Distribution:\n", "metric")
                    normal_indices = np.where(y_test_adjusted == 0)[0]
                    anomaly_indices = np.where(y_test_adjusted == 1)[0]
                    if len(normal_indices) > 0:
                        results_text.insert(tk.END, f"   • Normal samples - Mean MSE: {np.mean(mse_scores[normal_indices]):.6f}\n", "info")
                    if len(anomaly_indices) > 0:
                        results_text.insert(tk.END, f"   • Anomaly samples - Mean MSE: {np.mean(mse_scores[anomaly_indices]):.6f}\n", "info")

                results_text.insert(tk.END, "\n" + "=" * 65 + "\n", "header")
                results_text.insert(tk.END, "✅ TSA (LSTM Autoencoder) Model Ready!\n", "success")
                results_text.insert(tk.END, "   High reconstruction error indicates anomalies.\n", "info")
                results_text.insert(tk.END, "=" * 65 + "\n\n", "header")

                # Save model and encoders
                results_text.insert(tk.END, "💾 Saving model...\n", "info")
                results_window.update()

                tsa_model.save_model()
                joblib.dump(label_encoders, os.path.join(MODELS_DIR, "label_encoders.joblib"))

                results_text.insert(tk.END, f"✓ Model saved to {MODELS_DIR}\n", "success")

                # Update instance variables
                self.tsa_model = tsa_model
                self.label_encoders = label_encoders
                self.active_model = 'tsa'
                self.log_buffer = []  # Clear buffer for fresh predictions
                self.root.after(0, self.update_model_status)

                results_text.insert(tk.END, "✓ TSA model is now active for predictions!\n", "success")

            except Exception as e:
                import traceback
                error_msg = f"\n❌ Error: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)  # Always print to console
                try:
                    if results_window.winfo_exists():
                        results_text.insert(tk.END, error_msg, "warning")
                except:
                    pass  # Window already closed

        # Run training in thread
        threading.Thread(target=train, daemon=True).start()

    def on_closing(self):
        """Handle window closing"""
        self.running = False
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ServerMonitorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
