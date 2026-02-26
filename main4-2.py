# main4-2.py - Simulação de Especialização (LoRA Concept)
from openai import OpenAI

client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

# Jargões policiais para teste
texto_policial = """
A VTR 345 em patrulhamento visualizou um indivíduo em atitude suspeita.
Após abordagem, foi encontrado um simulacro. O QTH foi preservado até a chegada da perícia.
O meliante foi conduzido para a DP para lavratura do APF.
"""

def consultar_modelo(system_prompt, user_input, modelo_nome):
    print(f"\n[{modelo_nome}] Processando...")
    response = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.0
    )
    print(response.choices[0].message.content)

# 1. Simulação do Modelo Genérico (Sem LoRA/Adaptação)
prompt_generico = "Você é um assistente útil. Explique o texto para um leigo."
consultar_modelo(prompt_generico, texto_policial, "Modelo Genérico (Base)")

print("-" * 50)

# 2. Simulação do Modelo Especializado (Efeito do LoRA)
# O LoRA injetaria esse conhecimento matematicamente, mas aqui injetamos via contexto.
prompt_especializado = """
Você é um Especialista em Terminologia Policial da PCDF (Modelo Fine-Tuned).
Sua função é traduzir jargões técnicos para linguagem civil formal.
Glossário Interno:
- VTR: Viatura
- Simulacro: Arma falsa
- QTH: Local da ocorrência
- APF: Auto de Prisão em Flagrante
- DP: Delegacia de Polícia
Traduza o relato mantendo o tom jurídico.
"""
consultar_modelo(prompt_especializado, texto_policial, "Modelo PCDF (Simulando LoRA)")