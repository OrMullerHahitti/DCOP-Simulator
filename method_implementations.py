from typing import Tuple

from protocols import Method, assignment, lr


def update_neighbours_assignments(self, messages: list[Message]) -> None:
    for message in messages:  # Update neighbours assignments based on received messages
        neighbour = message.sender
        if neighbour in self.neighbours:
            self.neighbours_assignments[neighbour] = message.content

def find_assignment(self) -> Tuple [assignment, lr]:

############### TO DO : Update neighbours assignments before finding the best assignment - think where.############

    potential_cost = float('inf')  # Initialize potential cost to infinity
    best_assignment = self.assignment  # Initialize best assignment to current assignment

    for assignment in range(self.domain): # Iterate over all possible assignments
        temp_cost = 0
        for neighbour, ct in self.neighbours.items():
            neighbour_assignment = self.neighbours_assignments[neighbour]
            if ct.connections[self.name] == 0: # Means that in the cost table the rows represents the agents assignment and the columns the neighbours
                temp_cost += ct.table[assignment][neighbours_assignment]
            else:
                temp_cost += ct.table[neighbours_assignment][assignment]
        if assignment == self.assignment:  # update the local cost according to the neighbours assignments
            self.local_cost = temp_cost
        if temp_cost < potential_cost:  # Check if the current assignment is better than the previous best
            potential_cost = temp_cost
            best_assignment = assignment

    lr = self.local_cost - potential_cost
    return best_assignment, lr  # Return the best assignment and the change in cost

# GLOBAL INFORMATION: ALL PREVIOUSE ITERATIONS GLOBAL COST, ASSIGNMRNTS.
# finished = True if in the last iteration all the agents have the same assignment and the global cost is the same as the previous iteration

def perform_DSA(self, graph, seed, p) -> void:
    init_agents(seed) # needs to be the same for each algorithem in the same index of running this problem (30 runs for the same problem (graph) on each of the 5 algorithems.)

    for agent in graph.agents: # Every agent sends all neighbours its assignment
        agent.send_messages(agent.assignment)
    save_iteration_information()  # Save the initial global cost and assignments of all agents
    i = 1
    while (i <= 50):
        if i > 2 and self.global_cost[i-1] == self.global_cost[i-2] and self.global_assignments[i-1] == self.global_assignments[i-2]:
            break  # The algorithm has converged
        else:
            empty_mailboxs()  # Empty all mailboxes
            dispatch_messages()  # Dispatch messages to all agents

            for agent in self.agents: # זה מנקודת מבט על
                perform_DSA_itr()  # Perform the iteration according to the algorithm for each agent.
            save_iteration_information()  # Save the current global cost and assignments of all agents
            i+=1

################# agent method to perform the DSA algorithm ######################
def perform_DSA_itr(self):
    messages = self.receive_messages()
    self.update_neighbours_assignments(messages)
    best_assignment = self.find_assignment()[0]
    if np.random.rand() <= self.p and best_assignment != self.assignment:  # Probability to accept the new assignment
        self.assignment = best_assignment
        self.send_messages(self.assignment)  # Send the assignment to neighbours only if it was changed


def perform_MGM(self, graph, seed, p) -> void:
    init_agents(seed)  # needs to be the same for each algorithem in the same index of running this problem (30 runs for the same problem (graph) on each of the 5 algorithems.)

    for agent in graph.agents:  # Every agent sends all neighbours its assignment
        agent.send_messages(agent.assignment)
    save_iteration_information()  # Save the initial global cost and assignments of all agents
    i = 1
    while (i <= 50):
        if i > 2 and self.global_cost[i - 1] == self.global_cost[i - 2] and self.global_assignments[i - 1] == \
                self.global_assignments[i - 2]:
            break  # The algorithm has converged
        else:
            empty_mailboxs()  # Empty all mailboxes
            dispatch_messages()  # Dispatch messages to all agents

            for agent in self.agents:  # זה מנקודת מבט על
                agent.perform_MGM_itr_1()  #perform first half of the iteration
            for agent in self.agents:
                agent.perform_MGM_itr_2()  # perform second half of the iteration



                perform_DSA_itr()  # Perform the iteration according to the algorithm for each agent.
            save_iteration_information()  # Save the current global cost and assignments of all agents
            i += 1
def perform_MGM_itr_1(self) -> void: # MGM algorithm implementation
    messages = self.receive_messages()
    self.update_neighbours_assignments(messages)
    best_assignment, lr = self.find_assignment()
    if best_assignment != self.assignment:
        self.send_messages(lr)
        self.send_message(Message(self, self, best_assignment))  # Send myself the best assignment theres no need to compute it again in the second half or the MGM iteration
        self.send_messages(Message(self, self, lr))  # Send the lr to myself so I can use it in the second half of the iteration
def perform_MGM_itr_2(self) -> void: # MGM algorithm implementation
    messages = self.receive_messages()



class DsaMethod(Method):
    def __call__(self, *args, **kwargs)-> Tuple[assignment, lr]:
        # Implement the DSA algorithm here
        pass


class MGMMethod(Method):

    def __call__(self, *args, **kwargs)-> Tuple[assignment, lr]:
        # Implement the MGM algorithm here
        pass
    def _private_one(self):
        # Private method for MGM
        pass
    def _private_two(self):
        # Another private method for MGM
        pass