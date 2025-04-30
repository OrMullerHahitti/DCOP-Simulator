from typing import Protocol


class Agent(Protocol):
    def send_messages(self):
        pass

class AgentGraph(Protocol):
    def __init__(self, agents: list[Agent]):
        self.agents = agents
        self.neighbors = {}
        self.iteration = 0
        self.global_cost = float('inf')

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    def remove_agent(self, agent: Agent):
        self.agents.remove(agent)

    def get_neighbors(self, agent: Agent) -> list[Agent]:
        return self.neighbors.get(agent, [])

class Method(Protocol):
    def __call__(self, *args, **kwargs):
        pass
