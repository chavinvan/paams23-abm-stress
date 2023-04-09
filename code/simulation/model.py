import agentpy as ap
from worker import WorkerAgent
import pandas as pd

class StressModel(ap.Model):

    def setup(self):
        self.dataset = pd.read_csv(self.p.dataset_path).set_index(['user', 'date'])
        agent_ids = self.dataset.reset_index()['user'].unique().tolist()
        num_agents = len(agent_ids)
        self.agents = ap.AgentList(self, num_agents, WorkerAgent, worker_id=ap.AttrIter(agent_ids))

    def step(self):
        self.agents.calculate_stress()

    def update(self):
        self.agents.record('stress')