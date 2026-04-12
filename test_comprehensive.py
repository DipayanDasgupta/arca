from arca import ARCAAgent, NetworkEnv

def main():
    print("🚀 Starting ARCA Debugging Run...\n")
    
    # Setup Environment
    print("[1] Initializing 'small_office' Network Environment...")
    env = NetworkEnv.from_preset("small_office")
    
    # Initialize Agent
    print("[2] Initializing Agent...")
    agent = ARCAAgent(env=env)
    
    # No need to train, just run one episode to get the result object
    print("[3] Running one episode to inspect the output...")
    result = agent.run_episode()
    
    # --- DEBUGGING STEP ---
    # Let's find out what attributes the 'result' object has
    print("\n\n" + "="*50)
    print("DEBUGGING OUTPUT")
    print(f"The 'result' object is of type: {type(result)}")
    print("\nIt has the following attributes and methods:")
    print(dir(result))
    print("\nIts string representation is:")
    print(result)
    print("="*50 + "\n\n")

if __name__ == "__main__":
    main()
