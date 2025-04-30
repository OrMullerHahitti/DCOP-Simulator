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
finished = False
i=0
def DSA(self, p):
    while (not finished and i<50):
        save_iteration_information() # Save the current global cost and assignments of all agents
        empty_mailboxs() # Empty all mailboxes
        dispatch_messages() # Dispatch messages to all agents

        for agent in self.agents: # זה מנקודת מבט על
            messages = agent.receive_messages()
            agent.update_neighbours_assignments(messages)
            best_assignment = agent.find_assignment()[0]



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