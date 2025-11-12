import whisper
import os
import argparse
import torch
from tqdm import tqdm
import soundfile as sf
import noisereduce as nr
# Impor library baru untuk summarization
from transformers import pipeline, logging

# Menyembunyikan pesan logging yang tidak perlu dari transformers
logging.set_verbosity_error()

def summarize_text(text: str, summarizer, max_chunk_length: int = 512) -> str:
    """
    Membuat ringkasan dari teks yang panjang dengan membaginya menjadi beberapa bagian.

    Args:
        text (str): Teks lengkap yang akan diringkas.
        summarizer: Model pipeline summarization dari transformers.
        max_chunk_length (int): Panjang maksimum setiap potongan teks. Model T5 memiliki batas sekitar 512 token.

    Returns:
        str: Teks ringkasan yang sudah digabungkan.
    """
    try:
        print("\n🔄 Membuat ringkasan...")
        # Pisahkan teks menjadi potongan-potongan (chunks) yang lebih kecil
        # Ini adalah cara sederhana, bisa disempurnakan dengan tokenizer jika perlu
        words = text.split()
        chunks = [' '.join(words[i:i + max_chunk_length]) for i in range(0, len(words), max_chunk_length)]

        # Buat ringkasan untuk setiap chunk
        summaries = summarizer(chunks, max_length=150, min_length=30, do_sample=False)
        
        # Gabungkan semua ringkasan
        full_summary = ' '.join([summ['summary_text'] for summ in summaries])
        print("✅ Ringkasan selesai.")
        return full_summary
    except Exception as e:
        print(f"❌ Gagal membuat ringkasan: {e}")
        return "[Gagal membuat ringkasan]"


def transcribe_audio(model: whisper.Whisper, file_path: str, output_folder: str, language: str) -> tuple[bool, str, str]:
    """
    Mentranskripsi satu file audio/video dan menyimpannya sebagai file .txt.

    Returns:
        Tuple berisi (status_sukses, path_file_output, teks_transkripsi).
    """
    full_text = ""
    output_file_path = ""
    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        if base_name.endswith('_cleaned'):
            base_name = base_name[:-8]
            
        output_file_path = os.path.join(output_folder, f"{base_name}.txt")
        
        if os.path.exists(output_file_path):
            with open(output_file_path, 'r', encoding='utf-8') as f:
                # Asumsikan file yang ada sudah lengkap (ringkasan + transkrip)
                # atau hanya transkrip. Kita baca isinya untuk diringkas jika perlu.
                full_text = f.read().split("--- TRANSKRIPSI LENGKAP ---")[-1].strip()
            return True, output_file_path, full_text

        result = model.transcribe(file_path, language=language, verbose=False)
        
        # Simpan teks lengkap untuk diringkas nanti
        full_text = result.get("text", "")

        formatted_transcription = "\n".join(
            f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text'].strip()}"
            for seg in result.get("segments", [])
        )

        with open(output_file_path, "w", encoding="utf-8") as f:
            # Langsung tulis transkripsi lengkap saja
            f.write(formatted_transcription)

        return True, output_file_path, full_text
    except Exception as e:
        print(f"\n❌ Gagal mentranskripsi {os.path.basename(file_path)}: {e}")
        return False, output_file_path, full_text


def main():
    parser = argparse.ArgumentParser(description="Transkripsi & Summarization Audio/Video dengan Whisper.")
    parser.add_argument("audio_folder", type=str, help="Path ke folder yang berisi file audio/video.")
    parser.add_argument("--output_folder", type=str, default="transkrip", help="Folder untuk menyimpan hasil.")
    parser.add_argument("--model", type=str, default="medium", choices=['tiny', 'base', 'small', 'medium', 'large'], help="Ukuran model Whisper.")
    parser.add_argument("--language", type=str, default="id", help="Kode bahasa audio (default: 'id').")
    parser.add_argument("--clean_noise", action='store_true', help="Aktifkan pembersihan noise sebelum transkripsi.")
    parser.add_argument("--summarize", action='store_true', help="Aktifkan pembuatan ringkasan setelah transkripsi.")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Menggunakan perangkat: {device.upper()}")

    print(f"📥 Memuat model Whisper ({args.model})...")
    model = whisper.load_model(args.model, device=device)
    print("✅ Model Whisper berhasil dimuat.")

    # Muat model summarization HANYA jika diperlukan
    summarizer = None
    if args.summarize:
        print("📥 Memuat model Summarization... Ini mungkin butuh waktu saat pertama kali.")
        # MODEL DIUBAH: Menggunakan model T5 yang tidak memerlukan sentencepiece secara eksplisit.
        summarizer = pipeline("summarization", model="Falconsai/text_summarization", device=0 if device == "cuda" else -1)
        print("✅ Model Summarization berhasil dimuat.")

    if not os.path.isdir(args.audio_folder):
        print(f"❌ Error: Folder audio '{args.audio_folder}' tidak ditemukan.")
        return

    os.makedirs(args.output_folder, exist_ok=True)
    
    cleaned_audio_folder = os.path.join(args.audio_folder, "cleaned")
    if args.clean_noise:
        os.makedirs(cleaned_audio_folder, exist_ok=True)

    SUPPORTED_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.mp4', '.mov', '.avi', '.mkv')
    audio_files = [f for f in os.listdir(args.audio_folder) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    if not audio_files:
        print(f"⚠️ Tidak ada file media yang didukung ditemukan di '{args.audio_folder}'.")
        return
        
    print(f"\n🎧 Ditemukan {len(audio_files)} file untuk diproses.\n")

    success_count = 0
    failure_count = 0

    for filename in tqdm(audio_files, desc="Proses Total", unit="file"):
        file_path_original = os.path.join(args.audio_folder, filename)
        file_to_transcribe = file_path_original

        if args.clean_noise:
            # ... (logika pembersihan noise tetap sama) ...
            try:
                base_name_no_ext = os.path.splitext(filename)[0]
                cleaned_audio_path = os.path.join(cleaned_audio_folder, f"{base_name_no_ext}_cleaned.wav")
                if not os.path.exists(cleaned_audio_path):
                    data, rate = sf.read(file_path_original)
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    reduced_noise_data = nr.reduce_noise(y=data, sr=rate)
                    sf.write(cleaned_audio_path, reduced_noise_data, rate)
                file_to_transcribe = cleaned_audio_path
            except Exception as e:
                print(f"\n⚠️ Gagal membersihkan noise untuk {filename}: {e}. Melanjutkan dengan file asli.")

        # Lakukan transkripsi
        success, output_path, transcribed_text = transcribe_audio(model, file_to_transcribe, args.output_folder, args.language)

        if success:
            success_count += 1
            # Jika opsi summarize aktif dan ada teks untuk diringkas
            if args.summarize and transcribed_text:
                # Baca kembali konten asli (hanya transkrip)
                with open(output_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()

                # Buat ringkasan
                summary = summarize_text(transcribed_text, summarizer)
                
                # Tulis ulang file dengan format baru: ringkasan + transkrip
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("--- RINGKASAN ---\n")
                    f.write(summary + "\n\n")
                    f.write("--- TRANSKRIPSI LENGKAP ---\n")
                    f.write(original_content)
        else:
            failure_count += 1
            
    print("\n--- ✨ Proses Selesai ✨ ---")
    print(f"Berhasil: {success_count} file")
    print(f"Gagal: {failure_count} file")
    print(f"Hasil disimpan di folder: '{args.output_folder}'")


if __name__ == "__main__":
    main()
