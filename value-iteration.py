import gym
import pandas as pd

def main():
    env = gym.make("FrozenLake-v1")
    pl = Planner(env)
    print(env.desc)
    print(pl.plan())

class Planner:
    def __init__(self, env):
        self.env = env
        
    def s_to_loc(self, s):
        row = s//self.env.ncol
        col = s%self.env.ncol
        return row, col
    
    def reward(self, s):
        row, col = self.s_to_loc(s)
        if self.env.desc[row][col] == b'H':
            return -1
        elif self.env.desc[row][col] == b'G':
            return 1
        else:
            return 0
        
    # 動的計画法（価値反復法）
    def plan(self, num_states=16, num_actions=4, gamma=0.9, threshold=0.001):
        self.env.reset()
        V = {}

        # 価値関数をV(s)=0で初期化
        for s in range(num_states): 
            V[s] = 0 

        while True:
            delta = 0
            for s in range(num_states):
                row, col = self.s_to_loc(s)
                if self.env.desc[row][col] in [b'H', b'G']:
                    continue
                expected_rewards = []
                for a in range(num_actions):
                    r = 0
                    for prob, next_s, _, _ in self.env.P[s][a]:
                        r += prob*(self.reward(next_s)+gamma*V[next_s])
                    expected_rewards.append(r)
                V_new = max(expected_rewards)
            
                # |V_i+1 - V_i|
                delta = max(delta, abs(V_new-V[s]))
                V[s] = V_new
                
            if delta < threshold:
                break
        return self.dict_to_grid(V)
    
    def dict_to_grid(self, V):
        grid = []
        for i in range(self.env.nrow):
            row = [0]*self.env.ncol
            grid.append(row)
        for s in V:
            row, col = self.s_to_loc(s)
            grid[row][col] = V[s]
        return pd.DataFrame(grid)

if __name__ == '__main__':
    main()