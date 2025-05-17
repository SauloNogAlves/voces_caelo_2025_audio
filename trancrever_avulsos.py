import whisper
import os

# Caminho da pasta onde está o áudio
pasta_audio = "analisar_avulsos"

# Nome do arquivo de áudio a ser transcrito (você pode substituir pelo nome desejado)
nome_arquivo = "audio_exemplo.wav"  # <- Altere aqui para o nome correto do arquivo

# Caminho completo do arquivo
caminho_audio = os.path.join(pasta_audio, nome_arquivo)

# Carregando o modelo Whisper
print("Carregando modelo Whisper...")
modelo = whisper.load_model("base")  # ou "small", "medium", "large"

# Transcrevendo o áudio
print(f"Transcrevendo: {nome_arquivo}")
resultado = modelo.transcribe(caminho_audio)

# Exibindo a transcrição
print("\n--- Transcrição ---\n")
print(resultado["text"])
print("\n-------------------\n")

# Salvando a transcrição em um arquivo .txt
nome_txt = os.path.splitext(nome_arquivo)[0] + ".txt"
caminho_saida = os.path.join(pasta_audio, nome_txt)

with open(caminho_saida, "w", encoding="utf-8") as f:
    f.write(resultado["text"])

print(f"Transcrição salva em: {caminho_saida}")
