# orchestrator.py

from typing import List, Dict
from agents import (
    ChatGPTAgent,
    GrokAgent,
    GeminiAgent,
    LlamaAgent,
    CriticAgent,
    DecisionAgent,
)
from config import DEBUG


class Orchestrator:
    """
    Tüm agent'ları yöneten sınıf.
    Kullanıcıdan soru alır, ajanlara dağıtır, cevapları toplar, tartışma loop'u kurar.
    """

    def __init__(self):
        # Agent'ları burada oluşturuyoruz
        self.chatgpt_agent = ChatGPTAgent(
            name="ChatGPT",
            role_description="Yaratıcı ve açıklayıcı bir analist gibi davran. Detaylı açıkla."
        )

        self.grok_agent = GrokAgent(
            name="Grok",
            role_description="Gerçek zamanlı düşünüyormuş gibi davran, hızlı, direkt ve hafif alaycı ol."
        )

        self.gemini_agent = GeminiAgent(
            name="Gemini",
            role_description="Veri odaklı ve analitik davran. Artıları/eksileri listelemeye odaklan."
        )

        self.llama_agent = LlamaAgent(
            name="Llama",
            role_description="Daha deneysel, yaratıcı, alternatif bakış açıları üret."
        )

        self.critic_agent = CriticAgent(
            name="Critic",
            role_description="Diğer agentların cevaplarını eleştir, zayıf noktaları bul."
        )

        self.decision_agent = DecisionAgent(
            name="DecisionMaker",
            role_description=(
                "Tüm agentların görüşlerini okuyup tek bir net sonuç çıkar. "
                "Özet, tavsiye ve gerekçeleri ver."
            )
        )

        # Konuşma geçmişi (basit hafıza)
        self.conversation_history: List[Dict[str, str]] = []

    def broadcast_to_main_agents(self, user_message: str) -> Dict[str, str]:
        """
        Kullanıcı mesajını ana agent'lara gönder ve cevaplarını döndür.
        """
        responses = {}

        for agent in [self.chatgpt_agent, self.grok_agent, self.gemini_agent, self.llama_agent]:
            response = agent.think(self.conversation_history, user_message)
            responses[agent.name] = response

            # Geçmişe ekleyelim
            self.conversation_history.append({"role": "assistant", "content": f"[{agent.name}] {response}"})

            if DEBUG:
                print(f"[{agent.name}] cevabı:\n{response}\n")

        return responses

    def run_critique_round(self, main_responses: Dict[str, str]) -> str:
        """
        Critic agent, diğer agent'ların cevaplarını eleştirir.
        """
        combined = ""
        for name, resp in main_responses.items():
            combined += f"{name} diyor ki: {resp}\n\n"

        critic_input = (
            "Aşağıda diğer agentların verdigi cevaplar var. "
            "Lütfen bunları detaylıca eleştir, zayıf ve güçlü yanlarını yaz:\n\n"
            + combined
        )

        critic_response = self.critic_agent.think(self.conversation_history, critic_input)
        self.conversation_history.append({"role": "assistant", "content": f"[Critic] {critic_response}"})

        if DEBUG:
            print("[Critic] cevabı:\n", critic_response, "\n")

        return critic_response

    def run_decision_round(self, main_responses: Dict[str, str], critic_response: str, user_message: str) -> str:
        """
        Decision agent, tüm cevapları ve eleştiriyi okuyup final kararı verir.
        """
        combined = f"Kullanıcının sorusu: {user_message}\n\n"
        combined += "Diğer agentların cevapları:\n\n"
        for name, resp in main_responses.items():
            combined += f"{name}: {resp}\n\n"

        combined += f"Eleştirmen (Critic) görüşü: {critic_response}\n\n"
        combined += "Lütfen tüm bunlara dayanarak, tek bir net sonuç ve tavsiye çıkar."

        final_response = self.decision_agent.think(self.conversation_history, combined)
        self.conversation_history.append({"role": "assistant", "content": f"[Decision] {final_response}"})

        if DEBUG:
            print("[DecisionMaker] final karar:\n", final_response, "\n")

        return final_response

    def ask_panel(self, user_message: str) -> str:
        """
        Kullanıcıdan gelen tek bir mesaja karşılık:
        1) Ana agent'lara sor
        2) Critic ile eleştiri turu yap
        3) Decision agent ile final cevap üret
        """
        # Kullanıcı mesajını geçmişe ekle
        self.conversation_history.append({"role": "user", "content": user_message})

        if DEBUG:
            print("\n=== Yeni Panel Sorusu ===")
            print("Kullanıcı:", user_message)

        main_responses = self.broadcast_to_main_agents(user_message)
        critic_response = self.run_critique_round(main_responses)
        final_answer = self.run_decision_round(main_responses, critic_response, user_message)

        return final_answer
