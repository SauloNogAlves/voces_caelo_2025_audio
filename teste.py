import subprocess

try:
    subprocess.run(["ffmpeg", "-version"], check=True)
    print("ffmpeg encontrado com sucesso!")
except FileNotFoundError:
    print("ffmpeg NÃO encontrado no PATH do Python!")