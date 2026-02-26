# main4-1.py - Benchmark de Performance
import time
import psutil
import os
from openai import OpenAI

# Configuração
client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
MODELO = "llama3.2"

def medir_recursos():
    """Retorna o uso de RAM em MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def executar_teste(prompt, num_runs=3):
    print(f"--- Iniciando Benchmark do modelo {MODELO} ---")
    print(f"Pergunta: {prompt[:50]}...")

    tempos = []
    tokens_totais = 0

    # Medir memória antes (Baseline)
    mem_antes = medir_recursos()

    for i in range(num_runs):
        print(f"Execução {i+1}/{num_runs}...", end=" ")

        start_time = time.time()

        # Chamada ao modelo
        response = client.chat.completions.create(
            model=MODELO,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        end_time = time.time()
        duration = end_time - start_time

        # Contagem aproximada de tokens (palavras * 1.3)
        resposta = response.choices[0].message.content
        num_tokens = len(resposta.split()) * 1.3

        tempos.append(duration)
        tokens_totais += num_tokens
        print(f"Concluída em {duration:.2f}s ({int(num_tokens)} tokens)")

    # Estatísticas
    media_tempo = sum(tempos) / len(tempos)
    media_tokens_seg = (tokens_totais / sum(tempos))
    mem_depois = medir_recursos()

    print("\n--- RESULTADOS ---")
    print(f"Tempo Médio de Resposta: {media_tempo:.2f} segundos")
    print(f"Velocidade: {media_tokens_seg:.2f} tokens/segundo")
    print(f"Impacto na RAM do Python: {mem_depois - mem_antes:.2f} MB (O grosso está no processo do Ollama)")
    print("------------------")

# Input de Teste: Um texto longo para forçar o processamento
prompt_teste = """
Analise o seguinte cenário e gere um relatório técnico de 3 parágrafos:
Foi encontrado um dispositivo eletrônico modificado em um caixa eletrônico (Skimmer).
Descreva os procedimentos de cadeia de custódia e preservação digital.
"""

if __name__ == "__main__":
    executar_teste(prompt_teste)