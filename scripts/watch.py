from race_sim.env import RaceEnv
from agents.random_agent import RandomAgent
import time

def main():
    print("Starting simulation...")
    env = RaceEnv('tracks/Track0.png')
    agent = RandomAgent()
    
    for _ in range(1000):
        if env.viewer and not env.viewer.is_open:
            break 
        obs = env.reset()
        agent.reset()
        
        total_rewards = 0
        while True:
            if env.viewer and not env.viewer.is_open:
                break 
            action = agent.act(obs)
            obs, reward, done = env.step(action)
            env.render()
            total_rewards += reward
            time.sleep(0.05)
            
            if done:
                print(f"Total rewards in episode: {total_rewards}")
                time.sleep(2)
                break
    
if __name__ == "__main__":
    main()