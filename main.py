# main.py

from multi_agent import Orchestrator

def main():
    orchestrator = Orchestrator()

    print("OpenAI + Gemini + Grok + Claude Multi-Model Panel 👋")
    print("Modeller tartışacak, DecisionAgent ortak cevap verecek.")
    print("Çıkmak için 'q' veya 'quit' yaz.\n")

    while True:
        user_message = input("Sen: ")

        if user_message.strip().lower() in {"q", "quit", "exit"}:
            print("Görüşürüz! 👋")
            break

        result = orchestrator.ask_panel(user_message)

        print("\n--- OpenAI Cevabı ---")
        print(result["openai"])

        print("\n--- Gemini Cevabı ---")
        print(result["gemini"])

        print("\n--- Grok Cevabı ---")
        print(result["grok"])

        print("\n--- Claude Cevabı ---")
        print(result["claude"])

        print("\n=== ORTAK SONUÇ (DecisionAgent) ===")
        print(result["final"])
        print("====================================\n")


if __name__ == "__main__":
    main()
