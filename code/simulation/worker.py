import agentpy as ap
import numpy as np
from datetime import datetime

class WorkerAgent(ap.Agent):

    def setup(self, worker_id):
        self.worker_id = worker_id
        self.load_initial_data()

    def load_initial_data(self):
        self.worker_data = self.model.dataset.loc[self.worker_id]
        self.stress = self.worker_data.iloc[0]['stress']
        self.sleep = self.worker_data.iloc[0]['sleep']
        self.workload = self.worker_data.iloc[0]['workload']
        self.voice = self.worker_data.iloc[0]['voice']
        self.stress_history = []
        self.sleep_history = []
        self.workload_history = []
        self.voice_history = []

    def load_step_data(self):
        self.stress_history.append(self.stress)
        self.sleep_history.append(self.sleep)
        self.workload_history.append(self.workload)
        self.voice_history.append(self.workload)
        self.sleep = self.worker_data.iloc[self.model.t]['sleep']
        self.workload = self.worker_data.iloc[self.model.t]['workload']
        self.voice = self.worker_data.iloc[self.model.t]['voice']
        self.weekday = datetime.strptime(self.worker_data.iloc[self.model.t].name, '%Y-%m-%d').weekday()

    def calculate_stress(self):
        self.load_step_data()
        wl_avg = (self.workload_history[-1] + self.workload) / 2
        sp_avg = (self.sleep_history[-1] + self.sleep) / 2
        voice_avg = (self.voice_history[-1] + self.voice) / 2
        wl_std = np.std([self.workload_history[-1], self.workload])
        sp_std = np.std([self.sleep_history[-1], self.sleep])
        voice_std = np.std([self.voice_history[-1], self.voice])
        wl_max = np.max([self.workload_history[-1], self.workload])
        sp_max = np.max([self.sleep_history[-1], self.sleep])
        voice_max = np.max([self.voice_history[-1], self.voice])
        wl_min = np.min([self.workload_history[-1], self.workload])
        sp_min = np.min([self.sleep_history[-1], self.sleep])
        voice_min = np.min([self.voice_history[-1], self.voice])
        wl_delta = self.workload - self.workload_history[-1]
        sp_delta = self.sleep - self.sleep_history[-1]
        voice_delta = self.voice - self.voice_history[-1]

        ['workload', 'sleep', 'voice', 'weekday', 'prev_stress', 'wl_avg_2',
       'sp_avg_2', 'wl_std_2', 'sp_std_2', 'wl_max_2', 'sp_max_2', 'wl_min_2',
       'sp_min_2', 'wl_delta_2', 'sp_delta_2', 'voice_avg_2', 'voice_std_2',
       'voice_max_2', 'voice_min_2', 'voice_delta_2']
        input_data = np.asarray([self.workload, self.sleep, self.voice, self.weekday, self.stress_history[-1], wl_avg, wl_std, wl_max, wl_min, wl_delta, sp_avg, sp_std, sp_max, sp_min, sp_delta, voice_avg, voice_std, voice_max, voice_min, voice_delta]).reshape(1, 20)
        #input_data = np.asarray([self.sleep, self.workload, self.stress_history[-1], self.weekday, self.duration]).reshape(1, 5)
        #probabilities = self.p.predictor.predict_proba(input_data)[0]
        self.stress = self.p.predictor.predict(input_data)[0]
        # self.stress = self.model.nprandom.choice([1, 2, 3], p=probabilities)        