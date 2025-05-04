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


def perform(self, seed, graph: AgentGraph, perform_algorithm) -> None:
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
                agent.perform_algorithm()  # Perform the iteration according to the algorithm for each agent.
            save_iteration_information()  # Save the current global cost and assignments of all agents
            i += 1


# agent method to perform the DSA algorithm
def perform_DSA(self):
    messages = self.receive_messages()
    self.update_neighbours_assignments(messages)
    best_assignment = self.find_assignment()[0]
    if np.random.rand() <= self.p:  # Probability to accept the new assignment
        self.assignment = best_assignment
        self.send_messages(assignment, self.assignment)  # Send the assignment to neighbours only if it was changed


# agent method to perform the MGM algorithm
def perform_MGM(self) -> void:
    messages = self.receive_messages()
    if any(message.type == 'lr' for message in messages):  # meaning we are in phase 2 of the algorithm
        max_lr = float('-inf')
        sender_with_max_lr = None
        best_assignment = None

        for message in messages:  # Check who has the highest lr
            if message.type == 'lr' and message.content > max_lr:
                max_lr = message.content
                sender_with_max_lr = message.sender
            elif message.type == 'lr' and message.content == max_lr:
                # Resolve tie by index
                if str(message.sender) < str(sender_with_max_lr):
                    sender_with_max_lr = message.sender
            elif message.type == 'assignment' and message.sender == self.id:
                best_assignment = message.content

        if sender_with_max_lr == self.id:  # If I have the highest lr, update my assignment
            self.assignment = best_assignment
            self.send_messages(assignment, self.assignment)

    else:  # Phase 1 of the algorithm
        self.update_neighbours_assignments(messages)
        best_assignment, lr = self.find_assignment()
        self.send_messages(lr, lr)
        self.send_message(Message(self, self, assignment, best_assignment))  # Send myself the best assignment so theres no need to compute it again in the second phase of the MGM iteration
        self.send_messages(Message(self, self,lr, lr))  # Send the lr to myself so I can use it in the second half of the iteration


def pergorm_MGM2(self):
    if (condition_for_phase1):
        # 50-50 chance to send an invitation
        if np.random.rand() < 0.5:
            self.invited = True
            # Randomly choose one of the neighbors
            if self.neighbours:
                chosen_neighbor = np.random.choice(list(self.neighbours.keys()))
                self.send_message(Message(self, chosen_neighbor, invitation))
        else:
            self.invited = False
    elif (condition_for_phase2):
        if not self.invited:
            recieive_messages()
            # Check if I received an invitation
            for message in self.mailbox:
                if message.type == "invitation":
                    partner = message.sender
                    partner_neighbours = message.content
                    my_assignment, partner_assignment, lr = self.find_self_and_partner_assignment(partner, partner_neighbours)
                    self.send_message(Message(self, partner, "mgm2", (assignment, my_assignment, lr)))  # Need to think how the partner knows his is the first one in the tuple
    elif (condition_for_phase3):
        recieve_messages()
        if any(message.type == "mgm2" for message in self.mailbox):
            for message in self.mailbox:
                if message.type == "mgm2":
                    self.partner = message.sender
                    my_assignment, partner_assignment, lr = message.content[0], message.content[1], message.content[2]
                    self.send_messages("lr", lr)
                    # Send the assignment and lr to myself so I can use it in the next phase
                    self.send_message(Message(self, self, "lr", lr))
                    self.send_message(Message(self, self, "assignment", my_assignment))
        else:  # I dont have a partner
            best_assignment, lr = self.find_assignment()
            self.send_messages("lr", lr)
            self.send_message(Message(self, self, "assignment", best_assignment))  # Send myself the best assignment so theres no need to compute it again in the next phase
            self.send_messages(Message(self, self, "lr", lr))  # Send the lr to myself so I can use it in the next phase

    elif (condition_for_phase4):
        # get all the information from messages
        if (is_best_lr() and self.partner != None):
            self.send_message(Message(self, self.partner, "best_lr", True))
            self.send_message(Message(self, self, "assignment", best_assignment)
        elif (is_best_lr() and self.partner == None):
            #  change assignment? ot in next phase?
        else:
            self.send_message(Message(self, self.partner, "best_lr", False))

    elif (condition_for_phase5):
        # get all the information from messages
            if #אם השותף שלי שלח לי שיש לו את הבסט אל אר וגם לי יש, להחליף השמה ולשלוח לכל השכנים את ההשמה החדשה


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