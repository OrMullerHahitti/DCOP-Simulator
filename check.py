
 def main(different_graphs:List[AgentGraph], algorithms:List[Algorithm]):
     for graphs in different_graphs:

            for algorithm in algorithms:
                for agents in graphs.agents:
                    agents.empty_mailbox()

                def perform(self, seed, graph : AgentGraph, algorithem : Algorithem) -> void:
                    # init_agents(seed) - needs to be
                    graph.algorithm = algorithem

                    for agent in graph.agents: # Every agent sends all neighbours its assignment
                        agent.send_messages(agent.assignment)
                    save_iteration_information()  # Save the initial global cost and assignments of all agents
                    i = 1
                    while (i <= 50):
                        if i > 2 and self.gloval_cost[i-1] == self.global_cost[i-2] and self.global_assignments[i-1] == self.global_assignments[i-2]:
                            break  # The algorithm has converged
                        else:
                            empty_mailboxs()  # Empty all mailboxes
                            dispatch_messages()  # Dispatch messages to all agents

                            for agent in self.agents: # זה מנקודת מבט על
                                agent.preform_itr()  # Perform the DSA iteration algorithm for each agent
                            save_iteration_information()  # Save the current global cost and assignments of all agents
                            i+=1
