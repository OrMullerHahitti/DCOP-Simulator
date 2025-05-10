"""
Agent implementations for distributed constraint optimization.
Contains different algorithm implementations that extend the base Agent class.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from iot.core import Agent, Message

class DSAC(Agent):
    """Distributed Stochastic Algorithm with parameter p"""
    p: float = 0.7

    def decide(self) -> None:
        assignments = {m.sender.name: int(m.data[0]) for m in self.inbox if m.msg_type == "value"}

        cur_cost = self._cost(self.value, assignments)
        best_val, best_cost = self.value, cur_cost

        for v in range(self.domain_size):
            if v == self.value:
                continue
            c = self._cost(v, assignments)
            if c < best_cost:
                best_cost, best_val = c, v

        if best_cost < cur_cost and random.random() < self.p:
            self.value = best_val

        for n in self.neighbours:
            self._send(n, np.array([self.value]), "value")


class MGM(Agent):
    """Maximum Gain Message algorithm"""
    
    def __init__(self, name: str, domain_size: int):
        super().__init__(name, domain_size)
        self.mode = "value"
        self.best_gain = 0
        self.best_val = self.value

    def decide(self) -> None:
        if self.mode == "value":
            for n in self.neighbours:
                self._send(n, np.array([self.value]), "value")
            self.mode = "gain"
            return

        if self.mode == "gain":
            assignments = {m.sender.name: int(m.data[0]) for m in self.inbox if m.msg_type == "value"}
            cur_cost = self._cost(self.value, assignments)
            self.best_gain, self.best_val = 0, self.value
            for v in range(self.domain_size):
                if v == self.value:
                    continue
                gain = cur_cost - self._cost(v, assignments)
                if gain > self.best_gain:
                    self.best_gain, self.best_val = gain, v

            for n in self.neighbours:
                self._send(n, np.array([self.best_gain]), "gain")
            self.mode = "select"
            return

        if self.mode == "select":
            gains = [int(m.data[0]) for m in self.inbox if m.msg_type == "gain"]
            if self.best_gain > 0 and all(self.best_gain > g for g in gains):
                self.value = self.best_val
            for n in self.neighbours:
                self._send(n, np.array([self.value]), "value")
            self.mode = "value"


class MGM2(Agent):
    """MGM‑2 with the 'self‑gain' bug fixed"""

    def __init__(self, name: str, domain_size: int):
        super().__init__(name, domain_size)
        self.is_offerer = False
        self.current_partner: Optional[str] = None
        self.best_offer: Optional[Tuple[int, int, int]] = None  # (partner_val, my_val, gain)
        self.mode = "value"
        self.best_gain: int = 0
        self.best_val: int = self.value

    # ----------------------------------------------------------------------
    def decide(self) -> None:
        #phase 1: broadcast value
        if self.mode == "value":
            self.is_offerer = random.random() < 0.5
            self.current_partner = None
            self.best_offer = None

            for n in self.neighbours:
                self._send(n, np.array([self.value]), "value")

            self.mode = "compute_gain"
            return

        #phase 2: compute best gains
        if self.mode == "compute_gain":
            assignments = {m.sender.name: int(m.data[0]) for m in self.inbox if m.msg_type == "value"}
            for neighbor in self.neighbours:
                assignments.setdefault(neighbor.name, neighbor.value)

            cur_cost = self._cost(self.value, assignments)
            best_val, best_gain = self.value, 0

            for v in range(self.domain_size):
                if v == self.value:
                    continue
                gain = cur_cost - self._cost(v, assignments)
                if gain > best_gain:
                    best_gain, best_val = gain, v

            # keep for next phase
            self.best_gain = best_gain
            self.best_val = best_val

            # bilateral search if I'm an offerer
            if self.is_offerer:
                for neighbor in self.neighbours:
                    n_name = neighbor.name
                    n_val = assignments[n_name]

                    for my_val in range(self.domain_size):
                        for n_new_val in range(neighbor.domain_size):
                            if my_val == self.value and n_new_val == n_val:
                                continue

                            old_cost = self._cost(self.value, assignments) + \
                                       neighbor._cost(n_val, {self.name: self.value, **assignments})

                            mod_asg = assignments.copy()
                            mod_asg[n_name] = n_new_val
                            new_cost = self._cost(my_val, mod_asg) + \
                                       neighbor._cost(n_new_val, {self.name: my_val, **mod_asg})

                            gain = old_cost - new_cost
                            if gain > best_gain:
                                best_gain = gain
                                self.best_offer = (n_new_val, my_val, gain)
                                self.current_partner = n_name
                                self.best_gain = best_gain  # update

            # broadcast *my* best gain
            for n in self.neighbours:
                self._send(n, np.array([self.best_gain]), "gain")

            self.mode = "process_gains"
            return

        #phase 3: compare gains / make offers
        if self.mode == "process_gains":
            gains_received = {m.sender.name: int(m.data[0])
                              for m in self.inbox if m.msg_type == "gain"}

            # bilateral offer
            if self.is_offerer and self.best_offer and self.current_partner:
                partner_gain = gains_received.get(self.current_partner, 0)
                if self.best_gain > partner_gain and self.best_gain > 0:
                    partner = next(n for n in self.neighbours if n.name == self.current_partner)
                    self._send(partner, np.array(self.best_offer), "offer")
                    self.mode = "finalize"
                    return

            #unilateral move
            best_external_gain = max(gains_received.values(), default=0)
            if self.best_gain > 0 and self.best_gain >= best_external_gain:
                self.value = self.best_val

            for n in self.neighbours:
                self._send(n, np.array([self.value]), "value")

            self.mode = "finalize"
            return

        # phase 4: finalize
        if self.mode == "finalize":
            if not self.is_offerer:
                offers = [(m.sender.name, m.data) for m in self.inbox if m.msg_type == "offer"]
                if offers:
                    sender_name, offer_data = max(offers, key=lambda o: o[1][2])
                    my_new_val, gain = int(offer_data[0]), int(offer_data[2])
                    if gain > 0:
                        partner = next(n for n in self.neighbours if n.name == sender_name)
                        self._send(partner, np.array([1]), "accept")
                        self.value = my_new_val

            else:
                if self.best_offer and self.current_partner:
                    accepted = any(m.msg_type == "accept" and
                                   m.sender.name == self.current_partner
                                   for m in self.inbox)
                    if accepted:
                        self.value = self.best_offer[1]

            # reset
            self.mode = "value"
            self.is_offerer = False
            self.current_partner = None
            self.best_offer = None
            self.best_gain = 0

            for n in self.neighbours:
                self._send(n, np.array([self.value]), "value")