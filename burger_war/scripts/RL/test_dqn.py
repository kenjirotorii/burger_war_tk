import numpy as np

import DQN

class DQNTEST:

    def __init__(self):

        self.agent = DQN.Agent(num_states=24, num_actions=9, memory_cap=1000)
        self.num_states = 24
        self.num_actions = 9
        
    def run(self, num_epochs=1000, episode=0):

        state = np.random.rand(self.num_states).reshape(1, self.num_states)

        for i in range(num_epochs):

            action = self.agent.get_action(state=state, episode=episode)

            state_next = np.random.rand(self.num_states).reshape(1, self.num_states)
            reward = np.random.rand(1).reshape(1, 1)

            print("step:{}, state:{}, action:{}, state_next:{}, reward:{}".format(i, state, action, state_next, reward))

            self.agent.memorize(state, action, state_next, reward)

            self.agent.update_q_function(batch_size=32)
            state = state_next

        print("Save parameters...")
        self.agent.brain.save_model('./test_weight.hdf5')
        
    def reset(self):

        self.agent = DQN.Agent(num_states=self.num_states, num_actions=self.num_actions, memory_cap=1000)
        print("Load parameters...")
        self.agent.brain.load_model('./test_weight.hdf5')


if __name__ == "__main__":

    dqn = DQNTEST()
    for i in range(10):
        dqn.run(num_epochs=1000, episode=i)
        dqn.reset()

