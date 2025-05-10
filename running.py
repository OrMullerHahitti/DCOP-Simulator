from typing import Callable,Dict

import numpy as np
DOMAIN_SIZE = 5
global_mapping_ct_function = {
    'uniform':  np.random.uniform
}
from base import Agent, AgentGraph


class Engine:
    def __init__(self,graph:AgentGraph):
       self.graph = graph

    def _run(self, iterations:int,method:Callable):
        whi
    def run_all(self, iterations:int,methods:Dict[str,Callable]):
        for method_name, method in methods.items():
            self._run(iterations,method)


