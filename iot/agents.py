"""
Agent implementations for distributed constraint optimization.
Contains different algorithm implementations that extend the base Agent class.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from iot.core import Agent, Message

class DSA(Agent):
    """Distributed Stochastic Algorithm with parameter p"""
    p: float = 0.7  # deafule, can be changed

    def decide(self) -> None:
        neighbors_assignments = {m.sender: int(m.data[0]) for m in self.inbox if m.msg_type == "assignment"}  # gives a dictionary with neighbors as key and assignments as value
        self.update_neighbors_assignments(neighbors_assignments)  # update the neighbors assignments in the agent

        cur_cost = self._get_local_cost(self.assignment)  # local cost of current assignment
        best_assignment, best_cost = self.assignment, cur_cost # initialize best assignment and cost with current assignment and cost

        for a in range(self.domain_size):
            if a == self.assignment:
                continue
            c = self._get_local_cost(a)
            if c < best_cost:
                best_cost, best_assignment = c, a

        if best_cost < cur_cost and random.random() < self.p:
            self.assignment = best_assignment

        for n in self.neighbors:
            self._send(n, np.array([self.assignment]), "assignment")


class MGM(Agent):
    """Maximum lr Message algorithm"""
    
    def __init__(self, name: str, domain_size: int):
        super().__init__(name, domain_size)
        self.mode = "assignment"
        self.best_lr = 0
        self.best_assignment = self.assignment

    def decide(self) -> None:
        if self.mode == "assignment":
            for n in self.neighbors:
                self._send(n, np.array([self.assignment]), "assignment")
            self.mode = "lr"
            return

        if self.mode == "lr":
            neighbors_assignments = {m.sender: int(m.data[0]) for m in self.inbox if
                                     m.msg_type == "assignment"}  # gives a dictionary with neighbors as key and assignments as value
            self.update_neighbors_assignments(neighbors_assignments)  # update the neighbors assignments in the agent

        cur_cost = self._get_local_cost(self.assignment)
        self.best_lr, self.best_assignment = 0, self.assignment
        for a in range(self.domain_size):
            if a == self.assignment:
                continue
            lr = cur_cost - self._get_local_cost(a)
            if lr > self.best_lr:
                self.best_lr, self.best_assignment = lr, a

            for n in self.neighbors:
                self._send(n, np.array([self.best_lr]), "lr")
            self.mode = "select"
            return

        if self.mode == "select":
            lrs = {m.sender: int(m.data[0]) for m in self.inbox if m.msg_type == "lr"}
            for n in lrs:
                if lrs[n] > self.best_lr:
                    break
                elif lrs[n] == self.best_lr:
                    if self.name > n:
                        break
                else:
                    self.assignment = self.best_assignment

            for n in self.neighbors:
                self._send(n, np.array([self.assignment]), "assignment")
            self.mode = "assignment"


class MGM2(Agent):
    """MGM‑2 with the 'self‑lr' bug fixed"""

    def __init__(self, name: str, domain_size: int):
        super().__init__(name, domain_size)
        self.is_offerer = False
        self.current_partner: Optional[str] = None
        self.best_offer: Optional[Tuple[int, int, int]] = None  # (partner_assignment, my_assignment, lr)
        self.mode = "assignment"
        self.best_lr: int = 0
        self.best_assignment: int = self.assignment

    # ----------------------------------------------------------------------
    def decide(self) -> None:
        #phase 1: broadcast assignment
        if self.mode == "assignment":
            self.is_offerer = random.random() < 0.5
            self.current_partner = None
            self.best_offer = None

            for n in self.neighbors:
                self._send(n, np.array([self.assignment]), "assignment")

            self.mode = "compute_lr"
            return

        #phase 2: compute best lrs
        if self.mode == "compute_lr":
            assignments = {m.sender.name: int(m.data[0]) for m in self.inbox if m.msg_type == "assignment"}
            for neighbor in self.neighbors:
                assignments.setdefault(neighbor.name, neighbor.assignment)

            cur_cost = self._get_local_cost(self.assignment)
            best_assignment, best_lr = self.assignment, 0

            for a in range(self.domain_size):
                if a == self.assignment:
                    continue
                lr = cur_cost - self._get_local_cost(a)
                if lr > best_lr:
                    best_lr, best_assignment = lr, v

            # keep for next phase
            self.best_lr = best_lr
            self.best_assignment = best_assignment

            # bilateral search if I'm an offerer
            if self.is_offerer:
                for neighbor in self.neighbors:
                    n_name = neighbor.name
                    n_assignment = assignments[n_name]

                    for my_assignment in range(self.domain_size):
                        for n_new_assignment in range(neighbor.domain_size):
                            if my_assignment == self.assignment and n_new_assignment == n_assignment:
                                continue

                            old_cost = self._cost(self.assignment, assignments) + \
                                       neighbor._cost(n_assignment, {self.name: self.assignment, **assignments})

                            mod_asg = assignments.copy()
                            mod_asg[n_name] = n_new_assignment
                            new_cost = self._cost(my_assignment, mod_asg) + \
                                       neighbor._cost(n_new_assignment, {self.name: my_assignment, **mod_asg})

                            lr = old_cost - new_cost
                            if lr > best_lrlr:
                                best_lr = lr
                                self.best_offer = (n_new_assignment, my_assignment, lr)
                                self.current_partner = n_name
                                self.best_lr = best_lr  # update

            # broadcast *my* best lr
            for n in self.neighbors:
                self._send(n, np.array([self.best_lr]), "lr")

            self.mode = "process_lrs"
            return

        #phase 3: compare lrs / make offers
        if self.mode == "process_lrs":
            lrs_received = {m.sender.name: int(m.data[0])
                              for m in self.inbox if m.msg_type == "lr"}

            # bilateral offer
            if self.is_offerer and self.best_offer and self.current_partner:
                partner_lr = lrs_received.get(self.current_partner, 0)
                if self.best_lr > partner_lr and self.best_lr > 0:
                    partner = next(n for n in self.neighbors if n.name == self.current_partner)
                    self._send(partner, np.array(self.best_offer), "offer")
                    self.mode = "finalize"
                    return

            #unilateral move
            best_external_lr = max(lrs_received.values(), default=0)
            if self.best_lr > 0 and self.best_lr >= best_external_lr:
                self.assignment = self.best_assignment

            for n in self.neighbors:
                self._send(n, np.array([self.assignment]), "assignment")

            self.mode = "finalize"
            return

        # phase 4: finalize
        if self.mode == "finalize":
            if not self.is_offerer:
                offers = [(m.sender.name, m.data) for m in self.inbox if m.msg_type == "offer"]
                if offers:
                    sender_name, offer_data = max(offers, key=lambda o: o[1][2])
                    my_new_assignment, lr = int(offer_data[0]), int(offer_data[2])
                    if lr > 0:
                        partner = next(n for n in self.neighbors if n == sender_name)
                        self._send(partner, np.array([1]), "accept")
                        self.assignment = my_new_assignment

            else:
                if self.best_offer and self.current_partner:
                    accepted = any(m.msg_type == "accept" and
                                   m.sender.name == self.current_partner
                                   for m in self.inbox)
                    if accepted:
                        self.assignment = self.best_offer[1]

            # reset
            self.mode = "assignment"
            self.is_offerer = False
            self.current_partner = None
            self.best_offer = None
            self.best_lr = 0

            for n in self.neighbors:
                self._send(n, np.array([self.assignment]), "assignment")