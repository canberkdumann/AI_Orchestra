# multi_agent.py

import os
import json
import datetime
import urllib.request
import urllib.error
from typing import List, Dict
from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_BASE_URL,
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    GROK_API_KEY,
    GROK_MODEL,
    GROK_BASE_URL,
    CLAUDE_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_BASE_URL,
    CLAUDE_VERSION,
    DEBUG,
    USE_GROK,
)

# Kalıcı soru-cevap hafızası dosyası
QA_MEMORY_PATH = "qa_memory.jsonl"


# ============================================================
#  Q/A HAFIZA YARDIMCI FONKSİYONLARI
# ============================================================

def append_qa_memory(question: str, answer: str) -> None:
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "q": question,
        "a": answer,
    }
    try:
        with open(QA_MEMORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        if DEBUG:
            print("[QA_MEMORY] Yazma hatası:", e)


def find_similar_memories(query: str, max_items: int = 3) -> List[Dict[str, str]]:
    if not os.path.exists(QA_MEMORY_PATH):
        return []

    q_words = set(query.lower().split())
    if not q_words:
        return []

    candidates = []

    try:
        with open(QA_MEMORY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                past_q = obj.get("q", "")
                past_a = obj.get("a", "")

                past_words = set(past_q.lower().split())
                overlap = q_words.intersection(past_words)
                score = len(overlap)

                if score > 0:
                    candidates.append({
                        "score": score,
                        "q": past_q,
                        "a": past_a,
                    })
    except Exception as e:
        if DEBUG:
            print("[QA_MEMORY] Okuma hatası:", e)
        return []

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:max_items]


# ============================================================
#  CEVAP İÇİN DUPLICATE PARAGRAF TEMİZLEYİCİ
# ============================================================

def deduplicate_paragraphs(text: str) -> str:
    if not isinstance(text, str):
        return text

    blocks = text.split("\n\n")
    seen = set()
    result = []

    for block in blocks:
        norm = block.strip()
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        result.append(block)

    return "\n\n".join(result)


# ============================================================
#  OpenAI'ye HTTP ile istek atan fonksiyon
# ============================================================

def call_openai_chat(messages: List[Dict[str, str]]) -> str:
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
    }

    data = json.dumps(payload).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    req = urllib.request.Request(
        OPENAI_BASE_URL,
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
    except Exception as e:
        return f"[HATA] OpenAI isteği başarısız oldu: {e}"

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return f"[HATA] OpenAI cevabı JSON formatında değil: {body}"

    try:
        content = parsed["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return f"[HATA] OpenAI cevabı beklenen formatta değil: {parsed}"

    return deduplicate_paragraphs(content)


# ============================================================
#  Gemini'ye HTTP ile istek atan fonksiyon
# ============================================================

def call_gemini_chat(prompt: str) -> str:
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    data = json.dumps(payload).encode("utf-8")

    url = GEMINI_BASE_URL

    if DEBUG:
        print("[Gemini] İstek URL:", url)

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY,
    }

    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
    except Exception as e:
        return f"[HATA] Gemini isteği başarısız oldu: {e}"

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return f"[HATA] Gemini cevabı JSON formatında değil: {body}"

    try:
        candidates = parsed["candidates"]
        first = candidates[0]
        parts = first["content"]["parts"]
        texts = [p.get("text", "") for p in parts]
        content = "\n".join(texts).strip()
        if not content:
            content = "[HATA] Gemini boş bir cevap döndürdü."
    except (KeyError, IndexError, TypeError):
        return f"[HATA] Gemini cevabı beklenen formatta değil: {parsed}"

    return deduplicate_paragraphs(content)


# ============================================================
#  Grok / xAI'ye HTTP ile istek atan fonksiyon (opsiyonel)
# ============================================================

def call_grok_chat(messages: List[Dict[str, str]]) -> str:
    if not USE_GROK:
        return "[Grok devre dışı] USE_GROK=False olduğu için bu ortamda çağrılmıyor."

    if not GROK_API_KEY:
        return "[Grok devre dışı] GROK_API_KEY ayarlı değil."

    payload = {
        "model": GROK_MODEL,
        "messages": messages,
        "stream": False,
    }

    data = json.dumps(payload).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROK_API_KEY}",
    }

    req = urllib.request.Request(
        GROK_BASE_URL,
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = "<body okunamadı>"

        if e.code == 403:
            return (
                "[Grok kullanılamıyor] Sunucu 403 Forbidden döndürdü. "
                "Büyük ihtimalle bu model için yetkin yok veya hesabında bir kısıtlama var.\n"
                f"HTTP 403 detay: {err_body}"
            )
        return f"[HATA] Grok HTTP hata döndürdü: {e.code} - {err_body}"
    except Exception as e:
        return f"[HATA] Grok isteği başarısız oldu: {e}"

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return f"[HATA] Grok cevabı JSON formatında değil: {body}"

    try:
        content = parsed["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return f"[HATA] Grok cevabı beklenen formatta değil: {parsed}"

    return deduplicate_paragraphs(content)


# ============================================================
#  Claude / Anthropic'e HTTP ile istek atan fonksiyon
# ============================================================

def call_claude_chat(messages: List[Dict[str, str]]) -> str:
    """
    Anthropic /v1/messages endpoint'i:
      - URL: CLAUDE_BASE_URL
      - Header:
          x-api-key: CLAUDE_API_KEY
          anthropic-version: CLAUDE_VERSION
      - Body:
          {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": "...",
            "messages": [
              {"role": "user", "content": [{"type": "text", "text": "..."}]},
              {"role": "assistant", "content": [{"type": "text", "text": "..."}]},
              ...
            ]
          }
    """
    if not CLAUDE_API_KEY:
        return "[Claude devre dışı] CLAUDE_API_KEY tanımlı değil."

    system_parts = []
    claude_messages = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant"):
            claude_messages.append(
                {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                        }
                    ],
                }
            )

    system_text = "\n\n".join(system_parts) if system_parts else None

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1024,
        "messages": claude_messages,
    }
    if system_text:
        payload["system"] = system_text

    data = json.dumps(payload).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": CLAUDE_VERSION,
    }

    req = urllib.request.Request(
        CLAUDE_BASE_URL,
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
    except Exception as e:
        return f"[HATA] Claude isteği başarısız oldu: {e}"

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return f"[HATA] Claude cevabı JSON formatında değil: {body}"

    try:
        parts = parsed["content"]
        texts = [
            p.get("text", "")
            for p in parts
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        content = "\n".join(texts).strip()
        if not content:
            content = "[HATA] Claude boş bir cevap döndürdü."
    except (KeyError, TypeError):
        return f"[HATA] Claude cevabı beklenen formatta değil: {parsed}"

    return deduplicate_paragraphs(content)


# ============================================================
#  Ortak Agent sınıfı
# ============================================================

class BaseAgent:
    def __init__(self, name: str, role_description: str):
        self.name = name
        self.role_description = role_description

    def think(self, conversation_history: List[Dict[str, str]], user_message: str) -> str:
        messages: List[Dict[str, str]] = []

        messages.append({"role": "system", "content": self.role_description})
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        if DEBUG:
            print(f"\n[{self.name}] → modele istek hazırlanıyor...")

        response = self._call_model(messages)

        if DEBUG:
            print(f"[{self.name}] ← modelden cevap alındı.")

        return response

    def _call_model(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError("Her agent kendi _call_model metodunu tanımlamalı.")


# ============================================================
#  Farklı Agent tipleri
# ============================================================

class OpenAIAgent(BaseAgent):
    def _call_model(self, messages: List[Dict[str, str]]) -> str:
        return call_openai_chat(messages)


class GeminiAgent(BaseAgent):
    def _call_model(self, messages: List[Dict[str, str]]) -> str:
        text_blocks = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                text_blocks.append(f"[SİSTEM ROLÜ]: {content}")
            elif role == "user":
                text_blocks.append(f"[KULLANICI]: {content}")
            else:
                text_blocks.append(f"[ASİSTAN]: {content}")

        full_prompt = "\n\n".join(text_blocks)
        return call_gemini_chat(full_prompt)


class GrokAgent(BaseAgent):
    def _call_model(self, messages: List[Dict[str, str]]) -> str:
        return call_grok_chat(messages)


class ClaudeAgent(BaseAgent):
    def _call_model(self, messages: List[Dict[str, str]]) -> str:
        return call_claude_chat(messages)


class DecisionAgent(OpenAIAgent):
    pass


# ============================================================
#  ORCHESTRATOR
# ============================================================

class Orchestrator:
    """
    OpenAI + Gemini + (isteğe bağlı Grok) + Claude + DecisionAgent
    """

    def __init__(self):
        self.openai_agent = OpenAIAgent(
            name="OpenAIExpert",
            role_description=(
                "Sen OpenAI tabanlı, analitik ve açıklayıcı bir uzmansın. "
                "Kullanıcının sorusunu mantıklı, detaylı ve öğretici biçimde yanıtla. "
                "Cevabında gereksiz tekrar yapmamaya çalış."
            ),
        )

        self.gemini_agent = GeminiAgent(
            name="GeminiExpert",
            role_description=(
                "Sen Gemini tabanlı farklı bir modelsin. "
                "Kullanıcının sorusuna alternatif bir bakış açısıyla cevap ver, "
                "özellikle eksik kalan noktaları tamamlamaya çalış. "
                "OpenAIExpert'in muhtemel söyleyeceklerini kopyalamak yerine, "
                "onu tamamlayan ve farklılaştıran noktalar üretmeye odaklan."
            ),
        )

        self.grok_agent = GrokAgent(
            name="GrokExpert",
            role_description=(
                "Sen Grok (xAI) tabanlı bir modelsin. "
                "Gerçek zamanlılık, pratiklik ve net açıklamalarla öne çık. "
                "Eğer ağ nedeniyle kullanılamıyorsan, bunu kibarca belirt."
            ),
        )

        self.claude_agent = ClaudeAgent(
            name="ClaudeExpert",
            role_description=(
                "Sen Claude (Anthropic) tabanlı stratejik bir uzmansın. "
                "Yapısal, tutarlı ve dengeli yanıtlar üretmeye odaklan. "
                "Diğer modellerin söylediklerini tamamlayacak, eksiklerini dolduracak "
                "bir bakış açısı sun."
            ),
        )

        self.decision_agent = DecisionAgent(
            name="DecisionAgent",
            role_description=(
                "Görevin OpenAIExpert, GeminiExpert, GrokExpert ve ClaudeExpert'in "
                "cevaplarını okuyup tek, net ve dengeli bir final sonuç üretmek. "
                "Çelişkileri düzelt, ortak noktaları bul, gerektiğinde artı/eksi analizi yap "
                "ve sonunda net bir tavsiye ver.\n\n"
                "ÇOK ÖNEMLİ KURALLAR:\n"
                "- Aynı fikri veya paragrafı farklı cümlelerle tekrar etme.\n"
                "- Başlıkları (örneğin 'Özet', 'Derinlemesine Analiz') sadece bir kez kullan.\n"
                "- Eğer modeller aynı şeyi söylüyorsa, tek bir kez özetle yaz.\n"
                "- Cevabın derli toplu, gereksiz uzunlukta olmayan ve tekrar içermeyen bir yapıda olsun.\n"
                "- LÜTFEN diğer uzmanların cevaplarını aynen kopyalama veya uzun uzun alıntılama. "
                "Sadece kendi cümlelerinle kısa bir ortak özet yaz."
            ),
        )

        self.conversation_history: List[Dict[str, str]] = []

    def ask_panel(self, user_message: str) -> Dict[str, str]:
        if DEBUG:
            print("\n==========================")
            print("Yeni soru:", user_message)
            print("==========================")

        self.conversation_history.append({"role": "user", "content": user_message})

        similar_memories = find_similar_memories(user_message, max_items=3)

        memory_context = ""
        if similar_memories:
            memory_lines = []
            for i, mem in enumerate(similar_memories, start=1):
                memory_lines.append(
                    f"{i}) Geçmiş soru: {mem['q']}\n   Verilen cevap: {mem['a']}"
                )
            memory_context = (
                "Bu kullanıcıyla geçmişte şu soru-cevaplar yaşandı, bunları da dikkate al:\n\n"
                + "\n\n".join(memory_lines)
                + "\n\n"
            )

        base_input = (
            memory_context + "Şimdiki soru: " + user_message
            if memory_context
            else user_message
        )

        # 1) OpenAI
        openai_resp = self.openai_agent.think(
            conversation_history=self.conversation_history,
            user_message=base_input,
        )
        openai_resp = deduplicate_paragraphs(openai_resp)
        self.conversation_history.append(
            {"role": "assistant", "content": f"[OpenAI] {openai_resp}"}
        )

        # 2) Gemini
        gemini_resp = self.gemini_agent.think(
            conversation_history=self.conversation_history,
            user_message=base_input,
        )
        gemini_resp = deduplicate_paragraphs(gemini_resp)
        self.conversation_history.append(
            {"role": "assistant", "content": f"[Gemini] {gemini_resp}"}
        )

        # 3) Grok (opsiyonel ama agent her zaman var, fonksiyon içi USE_GROK'e bakıyor)
        grok_resp = self.grok_agent.think(
            conversation_history=self.conversation_history,
            user_message=base_input,
        )
        grok_resp = deduplicate_paragraphs(grok_resp)
        self.conversation_history.append(
            {"role": "assistant", "content": f"[Grok] {grok_resp}"}
        )

        # 4) Claude
        claude_resp = self.claude_agent.think(
            conversation_history=self.conversation_history,
            user_message=base_input,
        )
        claude_resp = deduplicate_paragraphs(claude_resp)
        self.conversation_history.append(
            {"role": "assistant", "content": f"[Claude] {claude_resp}"}
        )

        if DEBUG:
            print("\n--- OpenAIExpert Cevabı ---\n", openai_resp)
            print("\n--- GeminiExpert Cevabı ---\n", gemini_resp)
            print("\n--- GrokExpert Cevabı ---\n", grok_resp)
            print("\n--- ClaudeExpert Cevabı ---\n", claude_resp)

        # DecisionAgent için prompt
        decision_prompt = (
            "Aşağıda dört farklı uzmanın (OpenAI, Gemini, Grok, Claude) cevapları var.\n\n"
            "1) OpenAIExpert cevabı (sadece referans için):\n"
            f"{openai_resp}\n\n"
            "2) GeminiExpert cevabı (sadece referans için):\n"
            f"{gemini_resp}\n\n"
            "3) GrokExpert cevabı (sadece referans için):\n"
            f"{grok_resp}\n\n"
            "4) ClaudeExpert cevabı (sadece referans için):\n"
            f"{claude_resp}\n\n"
        )

        if similar_memories:
            decision_prompt += (
                "Ayrıca bu kullanıcıyla geçmişte şu soru-cevaplar yaşandı (bunları da referans olarak kullan):\n"
            )
            for mem in similar_memories:
                decision_prompt += f"- Soru: {mem['q']}\n  Cevap: {mem['a']}\n"
            decision_prompt += "\n"

        decision_prompt += (
            "Görevin bu dört cevabı ve varsa geçmiş soru-cevapları dikkate alarak, "
            "çelişkileri düzeltmek, en mantıklı noktaları birleştirmek ve kullanıcı için tek, net bir sonuç çıkarmaktır.\n\n"
            "ÖZEL TALİMATLAR:\n"
            "- Diğer uzmanların cevaplarını aynen tekrar etme.\n"
            "- Eğer modeller aynı şeyi farklı kelimelerle söylüyorsa, bunları tek bir kısa cümlede birleştir.\n"
            "- Başlıkları en fazla bir kez kullan; iki defa aynı başlık açma.\n"
            "- Gereksiz uzun tekrarlardan kaçın; bu cevabı okuyan kişinin zamanı kısıtlı.\n"
            "- Sadece kendi cümlelerinle, tek bir temiz ve tekrar içermeyen cevap yaz.\n\n"
            "Şimdi, tek bir temiz, tekrar içermeyen, iyi yapılandırılmış cevap üret."
        )

        final_resp = self.decision_agent.think(
            conversation_history=self.conversation_history,
            user_message=decision_prompt,
        )

        final_resp = deduplicate_paragraphs(final_resp)

        self.conversation_history.append(
            {"role": "assistant", "content": f"[Decision] {final_resp}"}
        )

        if DEBUG:
            print("\n=== DecisionAgent Final Cevap ===\n", final_resp)

        append_qa_memory(user_message, final_resp)

        return {
            "openai": openai_resp,
            "gemini": gemini_resp,
            "grok": grok_resp,
            "claude": claude_resp,
            "final": final_resp,
        }
