"""
Agent implementations for distributed constraint optimization.
Contains different algorithm implementations that extend the base Agent class.
"""

import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from iot.core import Agent, Message, ConstraintCost

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
        self.best_pair_lr = None
        self.received_lrs = None
        self.is_offerer = False
        self.current_partner: Optional[str] = None
        self.best_offer: Optional[Tuple[int, int, int]] = None  # (partner_assignment, my_assignment, lr)
        self.mode = "offer_phase"
        self.best_lr: int = 0
        self.best_assignment: int = self.assignment

    # ----------------------------------------------------------------------

    def compute_partner_cost(self,
            partner_assignment: int, computer_assignment: int,
            partner_neighbors: Dict[str, int],
            constraint_matrices: Dict[str, Union[ConstraintCost, list]]
    ) -> int:
        """
        Compute the local cost of `partner_assignment` for the partner agent,
        summing over all its neighbors.

        :param partner_assignment: the value to evaluate for the partner
        :param partner_neighbors: dict mapping neighbor_name -> neighbor_assignment
        :param constraint_matrices: dict mapping neighbor_name ->
                                    either a ConstraintCost instance or
                                    a raw cost-matrix (list of lists)
        :return: total local cost for the partner at that assignment
        """
        total_cost = 0
        for neighbor_name, neighbor_assignment in partner_neighbors.items():
            # get the cost descriptor for the edge (partner, neighbor)
            cm = constraint_matrices.get(neighbor_name)
            if cm is None:
                # no constraint defined (should not happen in a well-formed DCOP)
                continue

            if hasattr(cm, 'cost') and callable(cm.cost):
                # it's a ConstraintCost object
                if neighbor_name == self.name:
                    # this is the computer agent, so use its assignment
                    neighbor_assignment = computer_assignment
                total_cost += cm.cost(partner_assignment, neighbor_assignment)
            else:
                # assume it's a raw 2D list or numpy array
                if neighbor_name == self.name:
                    # this is the computer agent, so use its assignment
                    neighbor_assignment = computer_assignment
                total_cost += cm[partner_assignment][neighbor_assignment]

        return total_cost
    def decide(self) -> None:
        # -------------------- Mode: Offer Phase --------------------
        if self.mode == "offer_phase":
            # Update neighbor assignments from assignment messages
            assignments = {m.sender: int(m.data[0]) for m in self.inbox if m.msg_type == "assignment"}
            self.update_neighbors_assignments(assignments)

            # Reset per-iteration variables
            self.best_lr = 0
            self.best_pair_lr = 0
            self.received_lrs.clear()
            self.best_offer = None

            # Decide randomly whether to be an offerer
            self.is_offerer = random.random() < 0.5
            self.current_partner = None

            if self.is_offerer and self.neighbors:
                # Choose one neighbor as partner and send domain + neighbor info + constraint matrix
                self.current_partner = random.choice(list(self.neighbors.keys()))
                payload = np.array([
                    list(range(self.domain_size)),  # this agent's domain
                    self.neighbors,  # neighbors
                    self.constraints  # constraint costs
                ], dtype=object)
                self._send(self.current_partner, payload, "offer")

            self.mode = "respond_phase"
            return

        # -------------------- Mode: Respond Phase --------------------
        if self.mode == "respond_phase":
            # If not offerer and received an offer, compute best joint assignment and joint LR
            offers = [m for m in self.inbox if m.msg_type == "offer"]
            if offers and not self.is_offerer:
                offer = offers[0]
                sender = offer.sender
                partner_domain = offer.data[0]
                partner_neighbors = offer.data[1]  # neighbors of sender, including their assignments (it's a dictionary)
                constraint_matrices = offer.data[2]  # cost matrices of sender

                # Compute old local costs
                old_self_cost = self._get_local_cost(self.assignment)
                # Access partner agent to compute its local cost
                partner_agent = sender
                old_partner_cost = self.compute_partner_cost(self.neighbors[partner_agent], self.assignment, partner_neighbors, constraint_matrices) ######################## TO DO

                best_joint_lr = float('-inf')
                my_best = self.assignment
                partner_best = self.neighbors[partner_agent]

                # Search for best joint improvement
                for my_assignment in range(self.domain_size):
                    new_self_cost = self._get_local_cost(my_assignment)
                    for partner_assignment in partner_domain:
                        new_partner_cost = self.compute_partner_cost(partner_assignment, my_assignment, partner_neighbors, constraint_matrices)
                        joint_lr = (old_self_cost + old_partner_cost) - (new_self_cost + new_partner_cost)
                        if joint_lr > best_joint_lr:
                            best_joint_lr = joint_lr
                            my_best = my_assignment
                            partner_best = partner_assignment

                # Send proposal back to the offerer
                self._send(sender, np.array([my_best, partner_best, best_joint_lr]), "pair_proposal")

            self.mode = "lr_broadcast_phase"
            return

        # -------------------- Mode: LR Broadcast Phase --------------------
        if self.mode == "lr_broadcast_phase":
            # If I was the offerer, collect partner's response
            if self.is_offerer:
                proposals = [m for m in self.inbox if m.msg_type == "pair_proposal"]
                if proposals:
                    prop = proposals[0]
                    # (partner_assignment, my_assignment, joint_lr)
                    self.best_offer = (prop.data[0], prop.data[1], prop.data[2])
                    self.best_pair_lr = prop.data[2]

                else:  # If doesn't have a partner Compute own best local LR
                    self.best_lr = 0
                    self.best_assignment = self.assignment
                    cur_cost = self._get_local_cost(self.assignment)
                    for a in range(self.domain_size):
                        if a == self.assignment:
                            continue
                        lr = cur_cost - self._get_local_cost(a)
                        if lr > self.best_lr:
                            self.best_lr = lr
                            self.best_assignment = a

            # Broadcast either joint LR (if offerer with offer) or local LR
            lr_value = self.best_pair_lr if self.best_offer else self.best_lr
            for neighbor in self.neighbors:
                self._send(neighbor, np.array([lr_value]), "lr")

            self.mode = "confirm_phase"
            return
        # -------------------- Mode: Confirm Phase --------------------
        if self.mode == "confirm_phase":
            # Gather all neighbor LRs
            self.received_lrs = {m.sender: int(m.data[0]) for m in self.inbox if m.msg_type == "lr"}
            my_lr = self.best_pair_lr if self.best_offer else self.best_lr

            lrs = self.received_lrs

            greatest = True
            for n in lrs:
                if lrs[n] > my_lr:
                    if self.current_partner:
                        self._send(self.current_partner, np.array([0]), "confirm")
                        greatest = False
                    break
                elif lrs[n] == self.best_lr:
                    if self.name > n:
                        if self.current_partner:
                            self._send(self.current_partner, np.array([0]), "confirm")
                            greatest = False
                        break

            if greatest:
                if self.current_partner:
                    self._send(self.current_partner, np.array([1]), "confirm")

            self.mode = "apply_phase"
            return

        # -------------------- Mode: Apply Phase --------------------
        if self.mode == "apply_phase":
            # Check partner's confirmation for offerer or apply local if not in pair
            confirms = {m.sender: m.data[0] for m in self.inbox if m.msg_type == "confirm"}

            if self.best_offer and self.current_partner in confirms and confirms[self.current_partner] == 1:
                # Partner accepted joint proposal
                self.assignment = self.best_offer[1]
            else:
                # If no joint change, check individual LR
                lrs = self.received_lrs
                my_lr = self.best_lr
                greatest = True
                for n in lrs:
                    if lrs[n] > my_lr:
                        greatest = False
                        break
                    elif lrs[n] == self.best_lr:
                        if self.name > n:
                            greatest = False
                            break

                if greatest:
                    self.assignment = self.best_assignment

            # Broadcast updated assignment to all neighbors
            for neighbor in self.neighbors:
                self._send(neighbor, np.array([self.assignment]), "assignment")

            # Prepare for next round
            self.mode = "offer_phase"
            return
###################OR's########################################################
    # def decide(self) -> None:
    #     #phase 1: broadcast assignment
    #     if self.mode == "assignment":
    #         self.is_offerer = random.random() <= 0.5
    #         self.current_partner = None
    #         self.best_offer = None
    #
    #         for n in self.neighbors:
    #             self._send(n, np.array([self.assignment]), "assignment")
    #
    #         self.mode = "compute_lr"
    #         return
    #
    #     #phase 2: compute best lrs
    #     if self.mode == "compute_lr":
    #         assignments = {m.sender: int(m.data[0]) for m in self.inbox if m.msg_type == "assignment"}
    #         self.update_neighbors_assignments(assignments)  # update the neighbors assignments in the agent
    #
    #         cur_cost = self._get_local_cost(self.assignment)
    #         best_assignment, best_lr = self.assignment, 0
    #
    #         for a in range(self.domain_size):  # find my best assignment and lr
    #             if a == self.assignment:
    #                 continue
    #             lr = cur_cost - self._get_local_cost(a)
    #             if lr > best_lr:
    #                 best_lr, best_assignment = lr, a
    #
    #         # keep for next phase
    #         self.best_lr = best_lr
    #         self.best_assignment = best_assignment
    #
    #         # bilateral search if I'm an offerer
    #         if self.is_offerer:
    #             for neighbor in self.neighbors:
    #                 n_name = neighbor.name
    #                 n_assignment = assignments[n_name]
    #
    #                 for my_assignment in range(self.domain_size):
    #                     for n_new_assignment in range(neighbor.domain_size):
    #                         if my_assignment == self.assignment and n_new_assignment == n_assignment:
    #                             continue
    #
    #                         old_cost = self._cost(self.assignment, assignments) + \
    #                                    neighbor._cost(n_assignment, {self.name: self.assignment, **assignments})
    #
    #                         mod_asg = assignments.copy()
    #                         mod_asg[n_name] = n_new_assignment
    #                         new_cost = self._cost(my_assignment, mod_asg) + \
    #                                    neighbor._cost(n_new_assignment, {self.name: my_assignment, **mod_asg})
    #
    #                         lr = old_cost - new_cost
    #                         if lr > best_lr:
    #                             best_lr = lr
    #                             self.best_offer = (n_new_assignment, my_assignment, lr)
    #                             self.current_partner = n_name
    #                             self.best_lr = best_lr  # update
    #
    #         # broadcast *my* best lr
    #         for n in self.neighbors:
    #             self._send(n, np.array([self.best_lr]), "lr")
    #
    #         self.mode = "process_lrs"
    #         return
    #
    #     #phase 3: compare lrs / make offers
    #     if self.mode == "process_lrs":
    #         lrs_received = {m.sender.name: int(m.data[0])
    #                           for m in self.inbox if m.msg_type == "lr"}
    #
    #         # bilateral offer
    #         if self.is_offerer and self.best_offer and self.current_partner:
    #             partner_lr = lrs_received.get(self.current_partner, 0)
    #             if self.best_lr > partner_lr and self.best_lr > 0:
    #                 partner = next(n for n in self.neighbors if n.name == self.current_partner)
    #                 self._send(partner, np.array(self.best_offer), "offer")
    #                 self.mode = "finalize"
    #                 return
    #
    #         #unilateral move
    #         best_external_lr = max(lrs_received.values(), default=0)
    #         if self.best_lr > 0 and self.best_lr >= best_external_lr:
    #             self.assignment = self.best_assignment
    #
    #         for n in self.neighbors:
    #             self._send(n, np.array([self.assignment]), "assignment")
    #
    #         self.mode = "finalize"
    #         return
    #
    #     # phase 4: finalize
    #     if self.mode == "finalize":
    #         if not self.is_offerer:
    #             offers = [(m.sender.name, m.data) for m in self.inbox if m.msg_type == "offer"]
    #             if offers:
    #                 sender_name, offer_data = max(offers, key=lambda o: o[1][2])
    #                 my_new_assignment, lr = int(offer_data[0]), int(offer_data[2])
    #                 if lr > 0:
    #                     partner = next(n for n in self.neighbors if n == sender_name)
    #                     self._send(partner, np.array([1]), "accept")
    #                     self.assignment = my_new_assignment
    #
    #         else:
    #             if self.best_offer and self.current_partner:
    #                 accepted = any(m.msg_type == "accept" and
    #                                m.sender.name == self.current_partner
    #                                for m in self.inbox)
    #                 if accepted:
    #                     self.assignment = self.best_offer[1]
    #
    #         # reset
    #         self.mode = "assignment"
    #         self.is_offerer = False
    #         self.current_partner = None
    #         self.best_offer = None
    #         self.best_lr = 0
    #
    #         for n in self.neighbors:
    #             self._send(n, np.array([self.assignment]), "assignment")