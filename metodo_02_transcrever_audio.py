import os
import random
import soundfile as sf
import librosa
import numpy as np
import datetime
import whisper

# Função principal
def embaralhar_e_transcrever(caminho_audio_base, pergunta):
    print("Carregando áudio base...")
    audio, sr = librosa.load(caminho_audio_base, sr=None)

    duracao_total = librosa.get_duration(y=audio, sr=sr)
    print(f"Duração do áudio base: {duracao_total:.2f} segundos")

    tamanho_segmento = int(sr * 0.35) #  segundo = sr amostras
    segmentos = [audio[i:i + tamanho_segmento] for i in range(0, len(audio), tamanho_segmento)]
    
    print(f"Total de segmentos: {len(segmentos)}")
    print("Embaralhando segmentos...")
    random.shuffle(segmentos)

    print("Concatenando segmentos embaralhados...")
    audio_embaralhado = np.concatenate(segmentos)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_base = pergunta.replace(" ", "_").replace("?", "").replace("!", "")
    nome_arquivo_audio = f"audios_embaralhados/metodo_02_{nome_base}_{timestamp}.wav"
    nome_arquivo_texto = f"transcricoes/metodo_02_{nome_base}_{timestamp}.txt"

    os.makedirs("audios_embaralhados", exist_ok=True)
    os.makedirs("transcricoes", exist_ok=True)

    print(f"Salvando áudio embaralhado em: {nome_arquivo_audio}")
    sf.write(nome_arquivo_audio, audio_embaralhado, sr)

    print("Carregando modelo Whisper...")
    model = whisper.load_model("base")  # ou "small", "medium", "large" se quiser mais precisão

    print("Transcrevendo áudio...")
    resultado = model.transcribe(nome_arquivo_audio)

    texto = resultado["text"].strip()
    print("Transcrição obtida:", texto)

    with open(nome_arquivo_texto, "w", encoding="utf-8") as f:
        f.write(texto)

    print(f"Transcrição salva em: {nome_arquivo_texto}")

# Execução
if __name__ == "__main__":
    pergunta = input("Digite a pergunta para a entidade: ")
    caminho_audio_base = "audios_base/base_audio.wav"  # ajuste conforme necessário
    embaralhar_e_transcrever(caminho_audio_base, pergunta)
