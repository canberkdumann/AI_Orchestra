# document_utils.py

import os
from typing import Tuple

import pandas as pd


def load_text_file(path: str, max_chars: int = 8000) -> str:
    """
    Basit .txt dosyasını okur, çok uzunsa kırpar.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    if len(content) > max_chars:
        content = content[:max_chars] + "\n\n...[metin kısaltıldı]..."
    return content


def summarize_dataframe(
    df: pd.DataFrame,
    max_rows_preview: int = 20,
    max_cols_preview: int = 10,
) -> Tuple[str, str]:
    """
    Verilen DataFrame için:
      - Metinsel tablo ön izlemesi (ilk satırlar)
      - Kolon listesi, tipler ve sayısal kolonlar için özet istatistikler
    döner.
    """

    n_rows, n_cols = df.shape
    info_lines = [
        f"Tablo boyutu: {n_rows} satır x {n_cols} kolon",
    ]

    col_names = list(df.columns)
    if len(col_names) > max_cols_preview:
        shown_cols = col_names[:max_cols_preview]
        info_lines.append(
            f"İlk {max_cols_preview} kolon: {shown_cols} ... (toplam {len(col_names)} kolon)"
        )
    else:
        info_lines.append(f"Tüm kolonlar: {col_names}")


    dtypes_str = ", ".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])
    info_lines.append(f"Kolon veri tipleri: {dtypes_str}")


    try:
        preview_df = df.iloc[:max_rows_preview, :max_cols_preview]
        preview_text = preview_df.to_string(index=False)
    except Exception:
        preview_text = "[Ön izleme oluşturulurken hata oluştu]"

    preview_block = (
        "\n".join(info_lines)
        + "\n\nTablonun ilk satırlarından ön izleme:\n\n"
        + preview_text
    )


    extra_lines = []
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] == 0:
        extra_lines.append("Sayısal kolon bulunamadı.")
    else:
        desc = numeric_df.describe().T  
        extra_lines.append(
            "Sayısal kolonlar için temel istatistikler (count, mean, std, min, 25%, 50%, 75%, max):\n"
        )
        extra_lines.append(desc.to_string())

    extra_block = "\n".join(extra_lines)

    return preview_block, extra_block


def load_document_for_model(path: str) -> Tuple[str, str]:
    """
    Model için kullanılacak metni döndürür.
    Dönüş:
      (doc_main_text, extra_analysis_text)

    Desteklenen formatlar:
      - .txt
      - .csv
      - .xls, .xlsx, .xlsm, .xlsb (openpyxl ile)
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")

    ext = os.path.splitext(path)[1].lower()

   
    if ext == ".txt":
        main_text = load_text_file(path)
        extra = "Bu doküman düz metin (.txt) olarak yüklendi. Ek tablo istatistiği üretilmedi."
        return main_text, extra

   
    if ext == ".csv":
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise RuntimeError(f"CSV dosyası pandas ile okunamadı: {e}")

        preview, stats = summarize_dataframe(df)
        main_text = (
            "Bu dosya CSV formatında bir tablo olarak yüklendi.\n\n"
            "=== TABLO ÖN İZLEME ===\n"
            f"{preview}\n"
        )
        extra = (
            "=== TABLO İSTATİSTİK ÖZETİ ===\n"
            f"{stats}\n"
        )
        return main_text, extra


    if ext in {".xls", ".xlsx", ".xlsm", ".xlsb"}:
        try:
        
            df = pd.read_excel(path, engine="openpyxl")
        except Exception as e:
            raise RuntimeError(f"Excel dosyası pandas + openpyxl ile okunamadı: {e}")

        preview, stats = summarize_dataframe(df)
        main_text = (
            "Bu dosya Excel formatında bir tablo olarak yüklendi "
            "(pandas + openpyxl ile DataFrame'e dönüştürüldü).\n\n"
            "=== TABLO ÖN İZLEME ===\n"
            f"{preview}\n"
        )
        extra = (
            "=== TABLO İSTATİSTİK ÖZETİ ===\n"
            f"{stats}\n"
        )
        return main_text, extra

    raise ValueError(
        f"Şu an sadece .txt, .csv ve Excel (.xls, .xlsx, .xlsm, .xlsb) dosyalarını destekliyorum. "
        f"Verilen uzantı: {ext}"
    )
