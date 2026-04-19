import os
from arca import ARCAConfig, NetworkEnv, ARCAAgent

def main():
    print("🚀 Testing ARCA-Agent Library!")
    print("=" * 50)
    
    # 1. Initialize Configuration
    cfg = ARCAConfig.default()
    # Let's do a fast training run for the test (2,000 steps)
    cfg.rl.total_timesteps = 2000  
    
    # 2. Create the network environment
    print("\n[1] Setting up 'small_office' simulation environment...")
    env = NetworkEnv.from_preset("small_office", cfg=cfg)
    
    # 3. Initialize Agent
    print("\n[2] Initializing ARCA Agent...")
    agent = ARCAAgent(env=env, cfg=cfg)
    
    # 4. Train Agent
    print(f"\n[3] Training RL Agent for {cfg.rl.total_timesteps} timesteps...")
    agent.train(timesteps=cfg.rl.total_timesteps, progress_bar=True)
    
    # 5. Run an evaluation episode
    print("\n[4] Running evaluation episode (Agent attacking the network)...")
    print("-" * 50)
    result = agent.run_episode(render=True)
    print("-" * 50)
    
    # 6. Summary
    print("\n[5] Final Summary:")
    print(result.summary())
    print("=" * 50)
    print("✅ ARCA is fully installed and working perfectly!")

if __name__ == "__main__":
    main()
