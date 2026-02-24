from race_sim.env import RaceEnv
from agents.random_agent import RandomAgent
from race_sim.track import Track
from race_sim.viewer import Viewer

def main():
    print("Starting simulation...")
    env = RaceEnv()
    env.reset()
    agent = RandomAgent()
    agent.reset()
    agent.act(None)
    track = Track('tracks/Track0.png')
    viewer = Viewer(track)
    
if __name__ == "__main__":
    main()