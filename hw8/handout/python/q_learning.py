import sys
import numpy as np
from environment import MountainCar


class LinearModel:
    def __init__(self, state_size: int, action_size: int,
                 lr: float):
        self.weights = np.zeros((state_size, action_size))
        self.lr = lr
        self.bias = 0

    def predict(self, state):
        return np.argmax(self.weights.T @ state + self.bias)

    def update(self, state, action, target):
        coeff = self.lr * (np.dot(self.weights[:, action], state) + self.bias - target)
        self.weights[:, action] -= coeff * state
        self.bias -= coeff
        

class QLearningAgent:
    def __init__(self, env: MountainCar, mode: str = None, gamma: float = 0.9,
                 lr: float = 0.01, epsilon:float = 0.05):
            self.env = env
            self.epsilon = epsilon
            self.lr = lr
            self.gamma = gamma
            self.mode = mode
            if mode == "tile":
                self.linear_model = LinearModel(2048, 3, lr)
            if mode == "raw":
                self.linear_model = LinearModel(2, 3, lr)
    
    def get_action(self, state) -> int:
        """epsilon-greedy strategy.
        Given state, returns action.
        """
        rand = np.random.choice([True, False], p=[self.epsilon, 1-self.epsilon])
        if rand:
            return np.random.randint(0, 3)
        else:
            return self.linear_model.predict(state)

    def trans_state(self, state):
        if self.mode =="raw":
            return np.array(list(state.values()))
        else:
            trans_state = np.zeros(2048)
            indices = list(state.keys())
            trans_state[indices] = 1
            return trans_state

    def train(self, episodes: int, max_iterations: int) -> list[float]:
        """training function.
        Train for ’episodes’ iterations, where at most ’max_iterations‘ iterations
        should be run for each episode. Returns a list of returns.
        """
        returns = []
        for i in range(episodes):
            # print(env.reset())
            rewards = []
            state = self.trans_state(self.env.reset())
            for j in range(max_iterations):
                # print(state)
                action = self.get_action(state)
                next_state, reward, done = self.env.step(action)
                next_state = self.trans_state(next_state)
                rewards.append(reward)
                target = reward + self.gamma*np.max(self.linear_model.weights.T @ next_state + self.linear_model.bias)
                self.linear_model.update(state, action, target)

                state = next_state.copy()
                if done:
                    break
            returns.append(np.sum(rewards))
        return returns
            
if __name__ == "__main__":
    mode = sys.argv[1]
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    episodes = int(sys.argv[4])
    max_iterations = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7])
    learning_rate = float(sys.argv[8])
    
    # mode = "tile"
    # weight_out = "./OUTPUT_weights"
    # returns_out = "./OUTPUT_returns"
    # episodes = 4
    # max_iterations = 200
    # epsilon = 0.05
    # gamma = 0.99
    # learning_rate = 0.01

    env = MountainCar(mode=mode)
    agent = QLearningAgent(env, mode=mode, gamma=gamma, epsilon=epsilon, lr=learning_rate)
    returns = agent.train(episodes, max_iterations)
    # print(agent.linear_model.weights)
    # print(agent.linear_model.bias)

    with open(weight_out, "w") as f:
        f.write("{:.10f}\n".format(agent.linear_model.bias))
        for w in agent.linear_model.weights.flatten():
            f.write("{:.10f}\n".format(w))
    
    with open(returns_out, "w") as f:
        for r in returns:
            f.write("{:.2f}\n".format(r))
