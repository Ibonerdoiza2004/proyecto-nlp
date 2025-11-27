import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

df = pd.read_csv('../../dataset/dataset_preprocesado.csv')

# 1. INFORMACIÓN GENERAL DEL DATASET
print("1. INFORMACIÓN GENERAL")

print(f"\nDimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"\nColumnas disponibles:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print(f"\nTipos de datos:")
print(df.dtypes)

print(f"\nValores nulos por columna:")
print(df.isnull().sum())

print(f"\nMemoria utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 2. ESTADÍSTICAS DE AUDIO/SEGMENTOS
print("2. ESTADÍSTICAS DE AUDIO Y SEGMENTOS")

print(f"\nNúmero total de audios únicos: {df['audio_id'].nunique()}")
print(f"Número total de segmentos: {len(df)}")

print(f"\nEstadísticas de duración de segmentos (segundos):")
print(df['duration_sec'].describe())

print(f"\nDistribución de duraciones:")
print(f"  - Segmentos < 5 seg: {(df['duration_sec'] < 5).sum()} ({(df['duration_sec'] < 5).sum()/len(df)*100:.2f}%)")
print(f"  - Segmentos 5-10 seg: {((df['duration_sec'] >= 5) & (df['duration_sec'] < 10)).sum()} ({((df['duration_sec'] >= 5) & (df['duration_sec'] < 10)).sum()/len(df)*100:.2f}%)")
print(f"  - Segmentos 10-20 seg: {((df['duration_sec'] >= 10) & (df['duration_sec'] < 20)).sum()} ({((df['duration_sec'] >= 10) & (df['duration_sec'] < 20)).sum()/len(df)*100:.2f}%)")
print(f"  - Segmentos ≥ 20 seg: {(df['duration_sec'] >= 20).sum()} ({(df['duration_sec'] >= 20).sum()/len(df)*100:.2f}%)")

segmentos_por_audio = df.groupby('audio_id').size()
print(f"\nSegmentos por audio:")
print(f"  - Media: {segmentos_por_audio.mean():.2f}")
print(f"  - Mediana: {segmentos_por_audio.median():.2f}")
print(f"  - Min: {segmentos_por_audio.min()}")
print(f"  - Max: {segmentos_por_audio.max()}")


# 3. ESTADÍSTICAS DE SPEAKERS
print("3. ESTADÍSTICAS DE SPEAKERS")

print(f"\nNúmero total de speakers únicos: {df['speaker'].nunique()}")
print(f"\nTop 5 speakers más frecuentes:")
speaker_counts = df['speaker'].value_counts()
for i, (speaker, count) in enumerate(speaker_counts.head(5).items(), 1):
    print(f"  {i}. {speaker}: {count} intervenciones ({count/len(df)*100:.2f}%)")

duracion_por_speaker = df.groupby('speaker')['duration_sec'].agg(['sum', 'mean', 'count'])
duracion_por_speaker = duracion_por_speaker.sort_values('sum', ascending=False)
print(f"\nTop 5 speakers por tiempo total de intervención:")
for i, (speaker, row) in enumerate(duracion_por_speaker.head(5).iterrows(), 1):
    print(f"  {i}. {speaker}: {row['sum']:.2f} seg total, {row['mean']:.2f} seg promedio, {int(row['count'])} intervenciones")


# 4. ESTADÍSTICAS DE TEXTO
print("4. ESTADÍSTICAS DE TEXTO")

print(f"\nEstadísticas de caracteres:")
print(df['n_chars'].describe())

print(f"\nEstadísticas de palabras:")
print(df['n_words'].describe())

df['chars_per_word'] = df['n_chars'] / df['n_words']
print(f"\nPromedio de caracteres por palabra:")
print(df['chars_per_word'].describe())

print(f"\nDistribución por número de palabras:")
print(f"  - Textos < 10 palabras: {(df['n_words'] < 10).sum()} ({(df['n_words'] < 10).sum()/len(df)*100:.2f}%)")
print(f"  - Textos 10-30 palabras: {((df['n_words'] >= 10) & (df['n_words'] < 30)).sum()} ({((df['n_words'] >= 10) & (df['n_words'] < 30)).sum()/len(df)*100:.2f}%)")
print(f"  - Textos 30-50 palabras: {((df['n_words'] >= 30) & (df['n_words'] < 50)).sum()} ({((df['n_words'] >= 30) & (df['n_words'] < 50)).sum()/len(df)*100:.2f}%)")
print(f"  - Textos ≥ 50 palabras: {(df['n_words'] >= 50).sum()} ({(df['n_words'] >= 50).sum()/len(df)*100:.2f}%)")

df['words_per_second'] = df['n_words'] / df['duration_sec']
print(f"\nVelocidad de habla (palabras por segundo):")
print(df['words_per_second'].describe())

# 5. ANÁLISIS DE TOKENS Y LEMAS
print("5. ANÁLISIS DE TOKENS Y LEMAS")

def parse_list_column(col):
    """Convierte strings de listas a listas reales"""
    try:
        return ast.literal_eval(col) if isinstance(col, str) else []
    except:
        return []

all_tokens = []
for tokens_str in df['tokens_no_stop'].head(1000):
    tokens = parse_list_column(tokens_str)
    all_tokens.extend(tokens)

if all_tokens:
    token_freq = Counter(all_tokens)
    print(f"\nTop 20 tokens más frecuentes (sin stopwords):")
    for i, (token, freq) in enumerate(token_freq.most_common(20), 1):
        print(f"  {i}. {token}: {freq}")
    
    print(f"\nVocabulario único (muestra): {len(token_freq)} tokens únicos")

print("\nAnalizando lemas sin stopwords...")
all_lemmas = []
for lemmas_str in df['lemmas_no_stop'].head(1000):  
    lemmas = parse_list_column(lemmas_str)
    all_lemmas.extend(lemmas)

if all_lemmas:
    lemma_freq = Counter(all_lemmas)
    print(f"\nTop 20 lemas más frecuentes (sin stopwords):")
    for i, (lemma, freq) in enumerate(lemma_freq.most_common(20), 1):
        print(f"  {i}. {lemma}: {freq}")


# VISUALIZACIONES
import os
os.makedirs('../../graficos', exist_ok=True)

# Gráfico 1: Distribución de duraciones
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['duration_sec'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Duración (segundos)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Duraciones de Segmentos')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(df['duration_sec'])
plt.ylabel('Duración (segundos)')
plt.title('Boxplot de Duraciones')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../../graficos/duraciones.png', dpi=300, bbox_inches='tight')

# Gráfico 2: Distribución de palabras
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['n_words'], bins=50, edgecolor='black', alpha=0.7, color='green')
plt.xlabel('Número de palabras')
plt.ylabel('Frecuencia')
plt.title('Distribución de Número de Palabras')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(df['duration_sec'], df['n_words'], alpha=0.3)
plt.xlabel('Duración (segundos)')
plt.ylabel('Número de palabras')
plt.title('Relación Duración vs Palabras')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../../graficos/palabras.png', dpi=300, bbox_inches='tight')

# Gráfico 3: Top speakers
plt.figure(figsize=(14, 6))
top_speakers = speaker_counts.head(10)
plt.barh(range(len(top_speakers)), top_speakers.values)
plt.yticks(range(len(top_speakers)), top_speakers.index)
plt.xlabel('Número de intervenciones')
plt.title('Top 10 Speakers por Número de Intervenciones')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('../../graficos/top_speakers.png', dpi=300, bbox_inches='tight')

# Gráfico 4: Velocidad de habla
plt.figure(figsize=(10, 6))
plt.hist(df['words_per_second'], bins=50, edgecolor='black', alpha=0.7, color='orange')
plt.xlabel('Palabras por segundo')
plt.ylabel('Frecuencia')
plt.title('Distribución de Velocidad de Habla')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../../graficos/velocidad_habla.png', dpi=300, bbox_inches='tight')


