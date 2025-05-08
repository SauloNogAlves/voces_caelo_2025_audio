import pyaudio
import wave
import time
import threading

def play_and_record(input_wav, output_wav):
    # Configurações do áudio
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    
    # Inicializar PyAudio
    p = pyaudio.PyAudio()
    
    # Abrir arquivo de entrada para reprodução
    wf = wave.open(input_wav, 'rb')
    
    # Verificar se as configurações do arquivo de entrada são compatíveis
    if wf.getnchannels() != CHANNELS or wf.getsampwidth() != p.get_sample_size(FORMAT) or wf.getframerate() != RATE:
        print("As configurações do arquivo de entrada não são compatíveis.")
        return
    
    # Configurar stream de saída (reprodução)
    stream_out = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       output=True)
    
    # Configurar stream de entrada (gravação)
    stream_in = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
    
    # Configurar arquivo de saída para gravação
    wf_out = wave.open(output_wav, 'wb')
    wf_out.setnchannels(CHANNELS)
    wf_out.setsampwidth(p.get_sample_size(FORMAT))
    wf_out.setframerate(RATE)
    
    print("Iniciando reprodução e gravação simultânea...")
    
    # Variável para controle de sincronização
    start_time = time.time()
    
    # Função para reprodução
    def play_audio():
        data = wf.readframes(CHUNK)
        while data:
            stream_out.write(data)
            data = wf.readframes(CHUNK)
    
    # Função para gravação
    def record_audio():
        while time.time() - start_time < (wf.getnframes() / RATE):
            data = stream_in.read(CHUNK)
            wf_out.writeframes(data)
    
    # Criar e iniciar threads
    play_thread = threading.Thread(target=play_audio)
    record_thread = threading.Thread(target=record_audio)
    
    play_thread.start()
    record_thread.start()
    
    # Aguardar término das threads
    play_thread.join()
    record_thread.join()
    
    print("Processo concluído!")
    
    # Finalizar streams e arquivos
    stream_out.stop_stream()
    stream_out.close()
    stream_in.stop_stream()
    stream_in.close()   
    wf.close()
    wf_out.close()
    p.terminate()

if __name__ == "__main__":
    input_file = "audios_base/original.wav"  # Substitua pelo seu arquivo de entrada
    output_file = "gravacoes/gravado.wav"  # Nome do arquivo de saída
    
    play_and_record(input_file, output_file)