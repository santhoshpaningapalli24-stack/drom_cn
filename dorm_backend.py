# dorm_backend.py
"""
DORM backend module:
- dataset generation
- ML training (quick defaults)
- Satellite network and optimizer
- detection API used by Streamlit UI
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
import time
import random

# ---------- Data generator ----------
class NTNDatasetGenerator:
    def __init__(self, n_samples=2000, seed=42):
        self.n_samples = n_samples
        np.random.seed(seed)

    def generate_dataset(self):
        n_normal = int(self.n_samples * 0.7)
        n_jamming = int(self.n_samples * 0.1)
        n_spoofing = int(self.n_samples * 0.1)
        n_dos = self.n_samples - n_normal - n_jamming - n_spoofing

        def make_block(n, params, label):
            d = {}
            for k, (mu, sigma, kind) in params.items():
                if kind == "normal":
                    d[k] = np.random.normal(mu, sigma, n)
                elif kind == "uniform":
                    d[k] = np.random.uniform(mu, sigma, n)
                elif kind == "expo":
                    d[k] = np.random.exponential(mu, n)
                elif kind == "poisson":
                    d[k] = np.random.poisson(mu, n)
            d['attack_type'] = [label] * n
            return pd.DataFrame(d)

        normal_params = {
            'latency': (30, 5, 'normal'),
            'packet_loss': (0, 2, 'uniform'),
            'signal_strength': (-70, 5, 'normal'),
            'frequency_deviation': (0, 0.5, 'normal'),
            'transmission_power': (50, 5, 'normal'),
            'data_rate': (100, 10, 'normal'),
            'connection_duration': (300, 1, 'expo'),
            'retry_count': (1, 0, 'poisson'),
            'handover_frequency': (2, 0, 'poisson'),
            'doppler_shift': (0, 1, 'normal')
        }
        jamming_params = dict(normal_params)
        jamming_params.update({'latency': (150, 30, 'normal'), 'packet_loss': (40, 80, 'uniform'),
                               'signal_strength': (-95, 10, 'normal'), 'frequency_deviation': (5, 2, 'normal'),
                               'transmission_power': (80, 10, 'normal'), 'data_rate': (20, 10, 'normal'),
                               'connection_duration': (50, 1, 'expo'), 'retry_count': (10, 0, 'poisson'),
                               'handover_frequency': (8, 0, 'poisson')})
        spoofing_params = dict(normal_params)
        spoofing_params.update({'latency': (45, 15, 'normal'), 'packet_loss': (5, 20, 'uniform'),
                                'signal_strength': (-65, 8, 'normal'), 'frequency_deviation': (3, 1.5, 'normal'),
                                'transmission_power': (55, 8, 'normal'), 'data_rate': (90, 15, 'normal'),
                                'connection_duration': (200, 1, 'expo'), 'retry_count': (3, 0, 'poisson'),
                                'handover_frequency': (15, 0, 'poisson'), 'doppler_shift': (8, 3, 'normal')})
        dos_params = dict(normal_params)
        dos_params.update({'latency': (250, 50, 'normal'), 'packet_loss': (60, 95, 'uniform'),
                           'signal_strength': (-75, 10, 'normal'), 'frequency_deviation': (1, 1, 'normal'),
                           'transmission_power': (45, 10, 'normal'), 'data_rate': (10, 5, 'normal'),
                           'connection_duration': (30, 1, 'expo'), 'retry_count': (20, 0, 'poisson'),
                           'handover_frequency': (5, 0, 'poisson')})

        df = pd.concat([
            make_block(n_normal, normal_params, 'normal'),
            make_block(n_jamming, jamming_params, 'jamming'),
            make_block(n_spoofing, spoofing_params, 'spoofing'),
            make_block(n_dos, dos_params, 'dos'),
        ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

        return df

# ---------- ML detector (simple, fast) ----------
class MLThreatDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.accuracy = None

    def prepare_and_train(self, df, test_size=0.2, random_state=42, n_estimators=50):
        X = df.drop('attack_type', axis=1)
        y = df['attack_type']
        self.feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        t0 = time.time()
        clf.fit(X_train_s, y_train)
        t1 = time.time()
        self.model = clf
        y_pred = clf.predict(X_test_s)
        self.accuracy = accuracy_score(y_test, y_pred)
        return {'accuracy': self.accuracy, 'train_time_s': t1 - t0}

    def predict(self, sample_array):
        """sample_array: 1D array in order of feature_names"""
        if self.model is None:
            raise RuntimeError("Model not trained")
        Xs = self.scaler.transform([np.array(sample_array)])
        return self.model.predict(Xs)[0]

# ---------- Satellite & Network ----------
class Satellite:
    def __init__(self, sat_id, altitude_km=550.0):
        self.id = sat_id
        self.altitude = altitude_km
        self.resources = {'power': 100.0, 'bandwidth': 100.0, 'compute': 100.0}
        self.status = 'operational'
        self.active_threats = []

    def apply_threat(self, threat_label):
        self.active_threats.append(threat_label)
        self.status = 'under_attack'

    def mitigate(self, threat_label):
        if threat_label in self.active_threats:
            self.active_threats.remove(threat_label)
        if not self.active_threats:
            self.status = 'operational'

class SatelliteNetwork:
    def __init__(self, n=6):
        self.satellites = [Satellite(i, 550 + i*10) for i in range(n)]
    def get_state(self):
        return [{'id': s.id, 'status': s.status, 'resources': s.resources, 'threats': list(s.active_threats)} for s in self.satellites]

# ---------- Optimizer (simple) ----------
class DORMOptimizer:
    def __init__(self, alpha=0.6, beta=0.4):
        self.alpha = alpha
        self.beta = beta

    def utility(self, resources, n_threats):
        power, bandwidth, compute = np.array(resources) / 100.0
        performance = 0.3*power + 0.5*bandwidth + 0.2*compute
        security_cost = n_threats * 0.1
        return self.alpha * performance - self.beta * security_cost

    def optimize_for_satellite(self, sat: Satellite):
        n_threats = len(sat.active_threats)

        def objective(x):
            return -self.utility(x, n_threats)

        bounds = [(20, 100), (20, 100), (20, 100)]
        x0 = [sat.resources['power'], sat.resources['bandwidth'], sat.resources['compute']]
        res = minimize(objective, x0, bounds=bounds, method='SLSQP')
        if res.success:
            return {'power': float(res.x[0]), 'bandwidth': float(res.x[1]), 'compute': float(res.x[2]), 'utility': -res.fun}
        else:
            return {'power': x0[0], 'bandwidth': x0[1], 'compute': x0[2], 'utility': self.utility(x0, n_threats)}

# ---------- Full DORM Framework ----------
class DORMFramework:
    def __init__(self, n_satellites=6):
        self.network = SatelliteNetwork(n_satellites)
        self.generator = NTNDatasetGenerator(n_samples=2000)
        self.ml = MLThreatDetector()
        self.optimizer = DORMOptimizer()
        self.dataset = None
        self.results = {}

    def prepare_and_train(self, dataset_size=2000, n_estimators=50):
        self.generator = NTNDatasetGenerator(n_samples=dataset_size)
        self.dataset = self.generator.generate_dataset()
        info = self.ml.prepare_and_train(self.dataset, n_estimators=n_estimators)
        self.results['ml'] = info
        return info

    def detect_and_mitigate_sample(self, sample_array, sat_id):
        pred = self.ml.predict(sample_array)
        sat = self.network.satellites[sat_id]
        if pred != 'normal':
            sat.apply_threat(pred)
            optimal = self.optimizer.optimize_for_satellite(sat)
            # simulate mitigation success
            success = random.random() < 0.8
            if success:
                sat.mitigate(pred)
            return {'detected': True, 'type': pred, 'mitigated': success, 'optimal': optimal}
        return {'detected': False, 'type': 'normal', 'mitigated': False}

    def inject_random_attack(self):
        typ = random.choice(['jamming', 'spoofing', 'dos'])
        sat_id = random.randint(0, len(self.network.satellites)-1)
        self.network.satellites[sat_id].apply_threat(typ)
        return {'sat': sat_id, 'type': typ}

    def get_network_state(self):
        return self.network.get_state()
