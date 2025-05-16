import pyaudio
import wave
import time
import threading
import numpy as np
from pydub import AudioSegment
import os

def play_and_record(input_wav, output_wav, gain=1.0):
    # Configurações do áudio
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    SILENCE_THRESHOLD = 500  # Limite para considerar como silêncio
    
    # Criar diretório de saída se não existir
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)

    try:
        # Inicializar PyAudio
        p = pyaudio.PyAudio()
        
        # Abrir arquivo de entrada para reprodução
        wf = wave.open(input_wav, 'rb')
        
        # Verificar configurações do arquivo
        if wf.getnchannels() != CHANNELS:
            raise ValueError("Número de canais incompatível")
        if wf.getsampwidth() != p.get_sample_size(FORMAT):
            raise ValueError("Tamanho de amostra incompatível")
        if wf.getframerate() != RATE:
            raise ValueError("Taxa de amostragem incompatível")

        # Configurar stream de saída (reprodução)
        stream_out = p.open(format=FORMAT,
                           channels=CHANNELS,
                           rate=RATE,
                           output=True)
        
        # Configurar stream de entrada (gravação) com dispositivo padrão
        input_device_info = p.get_default_input_device_info()
        stream_in = p.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          input_device_index=input_device_info['index'],
                          frames_per_buffer=CHUNK)
        
        # Configurar arquivo de saída para gravação
        wf_out = wave.open(output_wav, 'wb')
        wf_out.setnchannels(CHANNELS)
        wf_out.setsampwidth(p.get_sample_size(FORMAT))
        wf_out.setframerate(RATE)
        
        print(f"Iniciando reprodução e gravação simultânea com ganho {gain}x...")
        
        # Variável para controle de sincronização
        start_time = time.time()
        duration = wf.getnframes() / RATE
        
        # Função para reprodução
        def play_audio():
            data = wf.readframes(CHUNK)
            while data:
                stream_out.write(data)
                data = wf.readframes(CHUNK)
        
        # Função para gravação com ajuste de ganho
        def record_audio():
            while time.time() - start_time < duration + 0.1:  # Pequena margem
                try:
                    data = stream_in.read(CHUNK, exception_on_overflow=False)
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    
                    # Aplicar ganho apenas se não for silêncio
                    if np.max(np.abs(audio_array)) > SILENCE_THRESHOLD:
                        audio_array = (audio_array * gain).astype(np.int16)
                    
                    wf_out.writeframes(audio_array.tobytes())
                except Exception as e:
                    print(f"Erro na gravação: {str(e)}")
                    break
        
        # Criar e iniciar threads
        play_thread = threading.Thread(target=play_audio)
        record_thread = threading.Thread(target=record_audio)
        
        play_thread.start()
        record_thread.start()
        
        # Aguardar término das threads
        play_thread.join()
        record_thread.join()
        
        print("Processo concluído! Fechando streams...")
        
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")
    finally:
        # Garantir que todos os recursos sejam liberados
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
        
        # Normalizar o áudio gravado
        try:
            normalize_audio(output_wav, output_wav)
        except Exception as e:
            print(f"Erro ao normalizar áudio: {str(e)}")

def normalize_audio(input_path, output_path, target_dBFS=-20):
    """Normaliza o áudio para um nível específico"""
    sound = AudioSegment.from_wav(input_path)
    change_in_dBFS = target_dBFS - sound.dBFS
    sound = sound.apply_gain(change_in_dBFS)
    sound.export(output_path, format="wav")

if __name__ == "__main__":
    input_file = "audios_base/original.wav"  # Arquivo de entrada
    output_file = "gravacoes/gravado.wav"   # Arquivo de saída
    
    # Ajuste o ganho (1.0 = normal, 2.0 = dobro de volume, etc.)
    play_and_record(input_file, output_file, gain=2.0)