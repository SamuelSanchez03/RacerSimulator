from race_sim.env import RaceEnv
from race_sim.track import Track
from agents.dqn_agent import TestDQNAgent 
from race_sim.types import CarState, Action
import time
import os

def main():
    training_tracks = ['tracks/Track0.png', 'tracks/Track1.png', 'tracks/Track2.png']
    testing_tracks = ['tracks/Track3.png', 'tracks/Track4.png'] 
    
    max_episodes_per_track = 1000 # Episodios por cada pista fácil
    save_interval = 100           # Guardar el cerebro cada 50 episodios
    
    print("Starting simulation...")

    env = RaceEnv('tracks/Track0.png')
    
    obs_prueba = env.reset()
    state_size = len(obs_prueba)
    
    agent = TestDQNAgent(state_size=state_size)
    
    archivo_pesos = "runs/pesos.pth"
    if os.path.exists(archivo_pesos):
        print(f"Cargando pesos previos desde {archivo_pesos}...")
        agent.load(archivo_pesos)
        # agent.eps = 0.5
    
    eps_decay = 0.995 
    min_eps = 0.01
    
    print("\n" + "="*40)
    print(" INICIANDO FASE DE ENTRENAMIENTO")
    print("="*40)

    for track_path in training_tracks:
        print(f"\n>>> Entrenando en el circuito: {track_path} <<<")
        env = RaceEnv(track_path)
        for episode in range(max_episodes_per_track):
            if env.viewer and not env.viewer.is_open:
                break 
                
            obs = env.reset()
            
            total_rewards = 0
            step_count = 0
            
            while True:
                if env.viewer and not env.viewer.is_open:
                    break 
                    
                # 1. El agente decide (act devuelve un objeto Action)
                action = agent.act(obs)
                
                # 2. El entorno avanza un paso
                next_obs, reward, done = env.step(action)
                total_rewards += reward
                
                # 3. ENTRENAMIENTO
                agent.train(obs, agent.last_action_idx, reward, next_obs, done)
                
                obs = next_obs.copy()
                step_count += 1
                
                # Renderizar solo cuando ya está avanzado el entrenamiento
                #'''
                if episode > (max_episodes_per_track - 2):
                    env.render()
                    time.sleep(0.02) 
                #'''
                
                if done or step_count > 1000:
                    break
            
            print(f"Episodio {episode+1}/{max_episodes_per_track} | Rew: {total_rewards:.2f} | Pasos: {step_count} | Eps: {agent.eps:.3f}")
            
            if agent.eps > min_eps:
                agent.eps = max(min_eps, agent.eps * eps_decay)
                
            if env.viewer and env.viewer.is_open:
                env.viewer.root.destroy()
                env.viewer = None
                
            # GUARDADO PERIÓDICO
            if (episode + 1) % save_interval == 0:
                agent.save(archivo_pesos)
                print(f"[*] Checkpoint guardado en {archivo_pesos}")
                
    print("\n" + "="*40)
    print(" BUSCANDO CIRCUITOS DE EVALUACIÓN...")
    print("="*40)
    
    # Filtramos para ver cuáles pistas realmente existen en la carpeta
    pistas_encontradas = [t for t in testing_tracks if os.path.exists(t)]
    
    if not pistas_encontradas:
        print("Pistas de prueba no encontradas (Fase oculta).")
        print(f"¡Entrenamiento completado! Tu modelo fue guardado en: '{archivo_pesos}'.")
        return
        
    print(f"¡Se encontraron {len(pistas_encontradas)} pistas de prueba! Iniciando evaluación...")
    
    # Modo evaluación: No exploramos, usamos solo lo aprendido
    agent.eps = 0.0 
    
    for track_path in pistas_encontradas:
        print(f"\n>>> Evaluando en: {track_path} <<<")
        env = RaceEnv(track_path)
        
        obs = env.reset()
        total_rewards = 0
        step_count = 0
        done = False
        
        while not done and step_count < 2000:
            if env.viewer and not env.viewer.is_open:
                return
                
            action = agent.act(obs)
            next_obs, reward, done = env.step(action)
            total_rewards += reward
            obs = next_obs.copy()
            step_count += 1
            
            env.render()
            time.sleep(0.03) 
            
        print(f"Resultado final en {track_path} -> Recompensa: {total_rewards:.2f} en {step_count} pasos.")
    
if __name__ == "__main__":
    main()