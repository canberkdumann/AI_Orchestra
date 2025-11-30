# analyze_document.py

from multi_agent import Orchestrator
from document_utils import load_document_for_model


def main():
    print("📄 Doküman Analiz Modu (OpenAI + Gemini + Grok + Claude + DecisionAgent)")
    print("Desteklenen dosya türleri: .txt, .csv, .xls, .xlsx, .xlsm, .xlsb")
    print("Çıkmak için dosya yolu sormadan sonra sohbet ekranında 'q' yazabilirsin.\n")

    file_path = input("Analiz etmek istediğin dosyanın TAM yolunu yaz: ").strip()
    if not file_path:
        print("Dosya yolu verilmedi, çıkılıyor.")
        return

    try:
        doc_main, doc_extra = load_document_for_model(file_path)
    except Exception as e:
        print(f"❌ Dosya okunurken / analiz edilirken hata oldu:\n{e}")
        return

    print("\n✅ Dosya yüklendi. Modele göndereceğim özet içerik aşağıda:\n")
    print("-" * 80)
    print(doc_main[:1500])
    print("\n--- EK ANALİZ / İSTATİSTİKLER ÖZETİ (ilk 1000 karakter) ---\n")
    print(doc_extra[:1000])
    print("-" * 80)

    doc_context = (
        "Aşağıda kullanıcıdan gelen bir dokümanın (Excel/CSV/TXT) içeriği ve senin için "
        "hazırlanmış özetler var.\n\n"
        "---------------- DOKÜMAN ÖN İZLEME BAŞI ----------------\n"
        f"{doc_main}\n"
        "---------------- DOKÜMAN ÖN İZLEME SONU ----------------\n\n"
        "---------------- EK ANALİZ / İSTATİSTİK BAŞI ----------------\n"
        f"{doc_extra}\n"
        "---------------- EK ANALİZ / İSTATİSTİK SONU ----------------\n\n"
        "Bu dokümanla ilgili kullanıcı sana sorular soracak. Önce veriyi/raporu anladığını "
        "gösteren kısa bir özet yap, ardından kullanıcının isteğine göre derinlemesine analiz / "
        "yorum / fikir üret. Varsayım yapman gerekiyorsa mantıklı ve açık bir şekilde belirt.\n\n"
    )

    orchestrator = Orchestrator()

    print(
        "\nArtık bu doküman hakkında seninle sohbet edeceğiz. 🌟\n"
        "- Sorunu yaz ve Enter'a bas.\n"
        "- Çıkmak için sadece 'q' yazıp Enter'a bas.\n"
    )

    first_turn = True

    while True:
        question = input("Sen: ").strip()

        if question.lower() in {"q", "quit", "çı", "çık", "exit"}:
            print("\n👋 Görüşürüz, oturum sonlandırıldı.")
            break

        if not question:
            print("(Boş mesaj algılandı, lütfen bir soru yaz veya 'q' ile çık.)")
            continue

        if first_turn:
            full_prompt = (
                doc_context
                + "Kullanıcının bu dokümanla ilgili ilk isteği:\n"
                f"{question}\n\n"
                "Lütfen önce dokümanı anladığını gösteren kısa bir özet yap. "
                "Ardından kullanıcının isteğine göre detaylı cevap ver. "
                "Önemli metrikleri vurgula, trendleri ve riskleri/fırsatları açıkla."
            )
            first_turn = False
        else:
            full_prompt = (
                "Aynı doküman üzerinde konuşmaya devam ediyoruz. "
                "Dokümanı yeniden uzun uzun özetlemek zorunda değilsin; önceki konuşmaları da "
                "dikkate al.\n\n"
                "Kullanıcının yeni sorusu / isteği:\n"
                f"{question}\n\n"
                "Lütfen önceki cevaplarınla çelişmeden, bu yeni soruya odaklanan, net ve "
                "tekrara girmeyen bir analiz yap."
            )

        result = orchestrator.ask_panel(full_prompt)

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
