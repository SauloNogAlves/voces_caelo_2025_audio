import numpy as np
import librosa
import matplotlib.pyplot as plt
from dtw import dtw
from scipy.spatial.distance import euclidean
import csv
import os
from tqdm import tqdm

def extract_mfcc(audio_path, n_mfcc=13):
    """Extrai coeficientes MFCC do áudio"""
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # Transpor para ter (frames, coeficientes)

def compare_segments(original_mfcc, recorded_mfcc, segment_length_sec=1, sr=16000):
    """Compara segmentos de áudio usando DTW"""
    frame_length = segment_length_sec * sr / 512  # 512 é o hop_length padrão do librosa
    results = []
    
    n_segments = min(len(original_mfcc), len(recorded_mfcc)) // int(frame_length)
    
    for i in range(n_segments):
        start = i * int(frame_length)
        end = (i + 1) * int(frame_length)
        
        orig_segment = original_mfcc[start:end]
        rec_segment = recorded_mfcc[start:end]
        
        # Calcular distância DTW
        d, _, _, _ = dtw(orig_segment, rec_segment, dist=euclidean)
        
        # Normalizar pela duração do segmento
        normalized_distance = d / len(orig_segment)
        
        # Considerar como anomalia se a distância for maior que o limiar
        threshold = 15.0  # Ajuste este valor conforme necessário
        is_anomaly = normalized_distance > threshold
        
        results.append({
            'segment': i,
            'start_time': i * segment_length_sec,
            'end_time': (i + 1) * segment_length_sec,
            'dtw_distance': normalized_distance,
            'is_anomaly': is_anomaly
        })
    
    return results

def generate_report(results, output_csv, plot_file=None):
    """Gera relatório CSV e opcionalmente um gráfico"""
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['segment', 'start_time', 'end_time', 'dtw_distance', 'is_anomaly']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    if plot_file:
        plt.figure(figsize=(12, 6))
        distances = [r['dtw_distance'] for r in results]
        anomalies = [r['start_time'] for r in results if r['is_anomaly']]
        
        plt.plot([r['start_time'] for r in results], distances, label='Distância DTW')
        plt.axhline(y=15.0, color='r', linestyle='--', label='Limiar de anomalia')
        
        for anomaly in anomalies:
            plt.axvline(x=anomaly, color='g', alpha=0.3)
        
        plt.xlabel('Tempo (s)')
        plt.ylabel('Distância DTW Normalizada')
        plt.title('Análise de Similaridade Fonética')
        plt.legend()
        plt.savefig(plot_file)
        plt.close()

def analyze_audio(original_path, recorded_path, output_csv, plot_file=None):
    """Executa análise completa"""
    print("Extraindo características MFCC...")
    original_mfcc = extract_mfcc(original_path)
    recorded_mfcc = extract_mfcc(recorded_path)
    
    print("Comparando segmentos...")
    results = compare_segments(original_mfcc, recorded_mfcc)
    
    print("Gerando relatório...")
    generate_report(results, output_csv, plot_file)
    
    anomalies = sum(1 for r in results if r['is_anomaly'])
    print(f"\nAnálise concluída. {anomalies} anomalias detectadas.")
    print(f"Relatório salvo em: {output_csv}")
    if plot_file:
        print(f"Gráfico salvo em: {plot_file}")

if __name__ == "__main__":
    # Configurações
    ORIGINAL_AUDIO = "audios_base/original.wav"
    RECORDED_AUDIO = "gravacoes/gravado.wav"
    OUTPUT_CSV = "relatorio_fonetico.csv"
    PLOT_FILE = "analise_fonetica.png"  # None para não gerar gráfico
    
    # Verificar se arquivos existem
    if not os.path.exists(ORIGINAL_AUDIO):
        print(f"Arquivo original não encontrado: {ORIGINAL_AUDIO}")
        exit(1)
    if not os.path.exists(RECORDED_AUDIO):
        print(f"Arquivo gravado não encontrado: {RECORDED_AUDIO}")
        exit(1)
    
    # Executar análise
    analyze_audio(ORIGINAL_AUDIO, RECORDED_AUDIO, OUTPUT_CSV, PLOT_FILE)