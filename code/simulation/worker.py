import agentpy as ap
import numpy as np
from datetime import datetime

class WorkerAgent(ap.Agent):

    def setup(self, worker_id):
        self.worker_id = worker_id
        self.load_initial_data()

    def load_initial_data(self):
        self.worker_data = self.model.dataset.loc[self.worker_id]
        self.stress_tolerance = self.model.dataset.iloc[0]['stress_tolerance']
        self.stress = self.worker_data.iloc[0]['stress']
        self.sleep = self.worker_data.iloc[0]['sleep']
        self.workload = self.worker_data.iloc[0]['workload']
        self.voice = self.worker_data.iloc[0]['voice']
        self.stress_history = []
        self.sleep_history = []
        self.workload_history = []
        self.voice_history = []

    def load_step_data(self):
        self.stress_history.append(self.stress)#self.stress_history.append(self.worker_data.iloc[self.model.t]['stress'])
        self.sleep_history.append(self.sleep)
        self.workload_history.append(self.workload)
        self.voice_history.append(self.voice)
        self.sleep = self.worker_data.iloc[self.model.t]['sleep']
        self.workload = self.worker_data.iloc[self.model.t]['workload']
        self.voice = self.worker_data.iloc[self.model.t]['voice']
        self.weekday = datetime.strptime(self.worker_data.iloc[self.model.t].name, '%Y-%m-%d').weekday()

    def calculate_stress(self):
        self.load_step_data()
        
        if self.model.t == 1:
            self.stress = self.worker_data.iloc[1]['stress']
        else:
            wl_avg = (self.workload_history[-2] + self.workload_history[-1] + self.workload) / 3
            sp_avg = (self.sleep_history[-2] + self.sleep_history[-1] + self.sleep) / 3
            voice_avg = (self.voice_history[-2] +self.voice_history[-1] + self.voice) / 3
            stress_avg = (self.stress_history[-1] + self.stress_history[-2]) / 2
            wl_std = np.std([self.workload_history[-2],self.workload_history[-1], self.workload])
            sp_std = np.std([self.sleep_history[-2], self.sleep_history[-1], self.sleep])
            voice_std = np.std([self.voice_history[-2], self.voice_history[-1], self.voice])
            stress_std = np.std([self.stress_history[-2], self.stress_history[-1]])
            wl_max = np.max([self.workload_history[-2], self.workload_history[-1], self.workload])
            sp_max = np.max([self.sleep_history[-2], self.sleep_history[-1], self.sleep])
            voice_max = np.max([self.voice_history[-2], self.voice_history[-1], self.voice])
            stress_max = np.max([self.stress_history[-2], self.stress_history[-1]])
            wl_min = np.min([self.workload_history[-2], self.workload_history[-1], self.workload])
            sp_min = np.min([self.sleep_history[-2], self.sleep_history[-1], self.sleep])
            voice_min = np.min([self.voice_history[-2], self.voice_history[-1], self.voice])
            stress_min = np.min([self.stress_history[-2], self.stress_history[-1]])
            wl_delta = self.workload - self.workload_history[-1]- self.workload_history[-2]
            sp_delta = self.sleep - self.sleep_history[-1] - self.sleep_history[-2]
            voice_delta = self.voice - self.voice_history[-1] - self.voice_history[-2]
            stress_delta = self.stress_history[-1] - self.stress_history[-2]
            
            '''
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
            '''
            ['workload', 'sleep', 'voice', 'weekday', 'prev_stress', 'wl_avg_2',
        'sp_avg_2', 'wl_std_2', 'sp_std_2', 'wl_max_2', 'sp_max_2', 'wl_min_2',
        'sp_min_2', 'wl_delta_2', 'sp_delta_2', 'voice_avg_2', 'voice_std_2',
        'voice_max_2', 'voice_min_2', 'voice_delta_2']
            
            # all + stress stats:
            input_data = np.asarray([self.workload, self.sleep, self.voice, self.stress_tolerance, self.stress_history[-1], wl_avg, sp_avg,  wl_std, sp_std, wl_max, sp_max, wl_min, sp_min, wl_delta, sp_delta, voice_avg, voice_std, voice_max, voice_min, voice_delta, stress_avg, stress_std, stress_max, stress_min, stress_delta]).reshape(1, 25)
            # all:
            #input_data = np.asarray([self.workload, self.sleep, self.voice, self.weekday, self.stress_history[-1], wl_avg, sp_avg,  wl_std, sp_std, wl_max, sp_max, wl_min, sp_min, wl_delta, sp_delta, voice_avg, voice_std, voice_max, voice_min, voice_delta]).reshape(1, 20)
            
            # w/o statisticals:
            # input_data = np.asarray([self.workload, self.sleep, self.voice, self.stress_tolerance, self.weekday, self.stress_history[-1]]).reshape(1, 6)
            
            # w/o voice
            # input_data = np.asarray([self.workload, self.sleep, self.voice, self.weekday, self.stress_history[-1], wl_avg, sp_avg,  wl_std, sp_std, wl_max, sp_max, wl_min, sp_min, wl_delta, sp_delta]).reshape(1, 15)

            #  ['sleep','workload','prev_stress', 'weekday', 'duration', 'wl_avg_{}'.format(ws), 'wl_std_{}'.format(ws), 'wl_max_{}'.format(ws), 'wl_min_{}'.format(ws), 'wl_delta_{}'.format(ws), 'sp_avg_{}'.format(ws), 'sp_std_{}'.format(ws), 'sp_max_{}'.format(ws), 'sp_min_{}'.format(ws), 'sp_delta_{}'.format(ws)]
            # top classifier
            #input_data = np.asarray([self.sleep, self.workload, self.stress_history[-1],self.weekday, self.voice,  wl_avg,  wl_std, wl_max, wl_min, wl_delta, sp_avg,sp_std,  sp_max, sp_min, sp_delta]).reshape(1, 15)
            #input_data = np.asarray([self.sleep, self.workload, self.stress_history[-1], self.weekday, self.duration]).reshape(1, 5)
            #probabilities = self.p.predictor.predict_proba(input_data)[0]
            
            #input_data_change = self.p.bin_scaler.transform(input_data)
            input_data_pred = self.model.scaler.transform(input_data)

            #change_proba = self.p.change_predictor.predict_proba(input_data_change)[0]
            #prediction_probs = self.p.predictor.predict_proba(input_data_pred)[0]
            prediction = self.model.predictor.predict(input_data_pred)[0]
            #change = self.p.change_predictor.predict(input_data_change)[0]
            #print('{} -----> {}'.format(input_data, prediction))
            
            self.stress = prediction
            #self.stress = self.model.nprandom.choice([1, 2, 3])  
            if False:
                if self.stress_tolerance == 0:
                    prediction_probs = np.multiply(prediction_probs, [1, self.p.high_tolerance_reduction, self.p.high_tolerance_reduction/2])
                    prediction_probs[0] = 1 - prediction_probs[1] - prediction_probs[2]
                elif self.stress_tolerance == 1:
                    prediction_probs = np.multiply(prediction_probs, [self.p.medium_tolerance_reduction, 1, self.p.medium_tolerance_reduction/2])
                    prediction_probs[1] = 1 - prediction_probs[0] - prediction_probs[2]
                elif self.stress_tolerance == 2:
                    prediction_probs = np.multiply(prediction_probs, [self.p.low_tolerance_reduction/2, self.p.low_tolerance_reduction, 1])
                    prediction_probs[2] = 1 - prediction_probs[0] - prediction_probs[1]
                self.stress = self.model.nprandom.choice([1, 2, 3], p=prediction_probs) 
            '''
            if change == 0:
                pass
            else:
                if self.stress_tolerance == 0:
                    prediction_probs = np.multiply(prediction_probs, [1, self.p.soft_reduction, self.p.hard_reduction])
                    prediction_probs[0] = 1 - prediction_probs[1] - prediction_probs[2]
                elif self.stress_tolerance == 1:
                    prediction_probs = np.multiply(prediction_probs, [self.p.soft_reduction, 1, self.p.hard_reduction])
                    prediction_probs[1] = 1 - prediction_probs[0] - prediction_probs[2]
                elif self.stress_tolerance == 2:
                    prediction_probs = np.multiply(prediction_probs, [self.p.hard_reduction, self.p.soft_reduction, 1])
                    prediction_probs[2] = 1 - prediction_probs[0] - prediction_probs[1]
                self.stress = self.model.nprandom.choice([1, 2, 3], p=prediction_probs)  
            '''
              