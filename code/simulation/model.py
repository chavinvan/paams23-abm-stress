import agentpy as ap
from worker import WorkerAgent
import pandas as pd
import pickle

class StressModel(ap.Model):

    def setup(self):
        self.dataset = pd.read_csv(self.p.dataset_path)#
        test_users = pickle.load(open('{}_{}.pickle'.format(self.p.test_users, self.p.i), "rb"))
        self.dataset = self.dataset[self.dataset['user'].isin(test_users)].set_index(['user', 'date'])
        self.predictor = pickle.load(open('{}_{}.pickle'.format(self.p.predictor_path, self.p.i), "rb"))
        self.scaler = pickle.load(open('{}_{}.pickle'.format(self.p.scaler_path, self.p.i), "rb"))
        agent_ids = self.dataset.reset_index()['user'].unique().tolist()
        num_agents = len(agent_ids)
        self.agents = ap.AgentList(self, num_agents, WorkerAgent, worker_id=ap.AttrIter(agent_ids))

    def step(self):
        self.agents.calculate_stress()

    def update(self):
        self.agents.record('stress')