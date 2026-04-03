import yaml
from chaperone.engine import GemmaEngine
from chaperone.memory import RAGMemory

def main():
    engine = GemmaEngine()
    memory = RAGMemory()

    # Load prompt template
    with open('configs/agent_prompts.yaml', 'r') as f:
        prompts = yaml.safe_load(f)

    # Pre-load the context into the agent's memory
    system_setup = prompts['system_setup'].format(documentation=memory.get_docs())
    engine.chat(system_setup)
    
    print("Chaperone Agent Ready.")
    print("-" * 40)

    # The Interactive Loop
    while True:
        user_question = input("\nAsk Chaperone (or type 'exit' to quit):\n> ")
        
        if user_question.lower() in ['exit', 'quit']:
            print("Shutting down agent...")
            break

        print("\nThinking...")
        response = engine.chat(user_question)
        
        print("\n--- CHAPERONE ---")
        print(response)

if __name__ == "__main__":
    main()
