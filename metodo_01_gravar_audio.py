import pyaudio
import wave
import time
import threading
import numpy as np
from pydub import AudioSegment
import os
from datetime import datetime
import whisper

def normalize_audio(input_path, output_path, target_dBFS=-20):
    """Normaliza o áudio para um nível específico"""
    sound = AudioSegment.from_wav(input_path)
    change_in_dBFS = target_dBFS - sound.dBFS
    sound = sound.apply_gain(change_in_dBFS)
    sound.export(output_path, format="wav")

def transcrever_audio(caminho_audio, caminho_saida_txt):
    print("Transcrevendo áudio com Whisper...")
    model = whisper.load_model("base")  # ou "small", "medium", etc.
    resultado = model.transcribe(caminho_audio)
    texto = resultado["text"].strip()
    
    with open(caminho_saida_txt, "w", encoding="utf-8") as f:
        f.write(texto)
    
    print("\n--- Transcrição ---\n")
    print(texto)
    print("\n-------------------\n")

def play_and_record(input_wav, output_wav, gain=1.0):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    SILENCE_THRESHOLD = 500

    os.makedirs(os.path.dirname(output_wav), exist_ok=True)

    try:
        p = pyaudio.PyAudio()
        wf = wave.open(input_wav, 'rb')

        if wf.getnchannels() != CHANNELS:
            raise ValueError("Número de canais incompatível")
        if wf.getsampwidth() != p.get_sample_size(FORMAT):
            raise ValueError("Tamanho de amostra incompatível")
        if wf.getframerate() != RATE:
            raise ValueError("Taxa de amostragem incompatível")

        stream_out = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
        input_device_info = p.get_default_input_device_info()
        stream_in = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                           input=True, input_device_index=input_device_info['index'],
                           frames_per_buffer=CHUNK)

        wf_out = wave.open(output_wav, 'wb')
        wf_out.setnchannels(CHANNELS)
        wf_out.setsampwidth(p.get_sample_size(FORMAT))
        wf_out.setframerate(RATE)

        print(f"Iniciando reprodução e gravação simultânea com ganho {gain}x...")
        start_time = time.time()
        duration = wf.getnframes() / RATE

        def play_audio():
            data = wf.readframes(CHUNK)
            while data:
                stream_out.write(data)
                data = wf.readframes(CHUNK)

        def record_audio():
            while time.time() - start_time < duration + 0.1:
                try:
                    data = stream_in.read(CHUNK, exception_on_overflow=False)
                    audio_array = np.frombuffer(data, dtype=np.int16)

                    if np.max(np.abs(audio_array)) > SILENCE_THRESHOLD:
                        audio_array = (audio_array * gain).astype(np.int16)

                    wf_out.writeframes(audio_array.tobytes())
                except Exception as e:
                    print(f"Erro na gravação: {str(e)}")
                    break

        play_thread = threading.Thread(target=play_audio)
        record_thread = threading.Thread(target=record_audio)

        play_thread.start()
        record_thread.start()

        play_thread.join()
        record_thread.join()

        print("Processo concluído! Fechando streams...")

    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")
    finally:
        if 'stream_out' in locals():
            stream_out.stop_stream()
            stream_out.close()
        if 'stream_in' in locals():
            stream_in.stop_stream()
            stream_in.close()
        if 'wf' in locals():
            wf.close()
        if 'wf_out' in locals():
            wf_out.close()
        if 'p' in locals():
            p.terminate()

        try:
            normalize_audio(output_wav, output_wav)
        except Exception as e:
            print(f"Erro ao normalizar áudio: {str(e)}")

if __name__ == "__main__":
    pergunta = input("Digite a pergunta para a entidade: ").strip().lower().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_base = f"{pergunta}_{timestamp}"

    input_file = "audios_base/original.wav"
    output_file = f"gravacoes/metodo_01_{nome_base}.wav"
    transcricao_file = f"transcricoes/metodo_01_{nome_base}.txt"

    os.makedirs("transcricoes", exist_ok=True)

    # Grava e reproduz com ganho ajustado
    play_and_record(input_file, output_file, gain=2.0)

    # Transcreve e salva o texto
    transcrever_audio(output_file, transcricao_file)
