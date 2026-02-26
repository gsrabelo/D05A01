# main3-3.py - RLM (Resumidor Recursivo)
from openai import OpenAI

client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

# Função que resume um pedaço de texto
def resumir_bloco(texto):
    response = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "Resuma o seguinte texto focando em fatos policiais, nomes e crimes. Seja conciso."},
            {"role": "user", "content": texto}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

# Simulação de Degravação de Grampo (Muitas conversas inúteis e um crime no meio)
# Multiplicamos para gerar volume de tokens e forçar a IA a ler tudo.

conversa_fiada = """
[ÁUDIO 01 - 10:00] ALVO: E aí, vai pro jogo do Flamengo?
[ÁUDIO 01 - 10:01] INTERLOCUTOR: Não sei, tá caro o ingresso.
[ÁUDIO 01 - 10:02] ALVO: Pois é, mas vai ser jogão. Comprou a carne pro churrasco?
[ÁUDIO 01 - 10:03] INTERLOCUTOR: Ainda não, vou passar no mercado mais tarde.
""" * 40  # Repete 40 vezes para encher linguiça

momento_do_crime = """
[ÁUDIO 45 - 23:15] ALVO: Presta atenção. A carga "branca" chega amanhã no galpão 4 do SIA.
[ÁUDIO 45 - 23:16] INTERLOCUTOR: Tem certeza? A polícia tá rondando.
[ÁUDIO 45 - 23:17] ALVO: Tá limpo. O código do cadeado mudou para 9988. Pega as armas no fundo falso da Hilux.
"""

mais_conversa_fiada = """
[ÁUDIO 99 - 08:00] ALVO: Bom dia, mãe. Dormiu bem?
[ÁUDIO 99 - 08:01] MÃE: Dormi sim, meu filho. Vai vir almoçar?
""" * 40

# Montagem do texto final para o RLM processar
texto_longo = conversa_fiada + momento_do_crime + mais_conversa_fiada

# Configuração do RLM
TAMANHO_BLOCO = 500 # Caracteres por bloco (ajuste conforme a janela do modelo)

def executar_rlm(texto_completo):
    print(f"Texto original tem {len(texto_completo)} caracteres.")

    # 1. Fatiar (Map)
    blocos = [texto_completo[i:i+TAMANHO_BLOCO] for i in range(0, len(texto_completo), TAMANHO_BLOCO)]
    print(f"Dividido em {len(blocos)} blocos para processamento.")

    resumos_parciais = []

    # 2. Resumir cada bloco (Processamento Sequencial)
    for i, bloco in enumerate(blocos):
        print(f"Resumindo bloco {i+1}/{len(blocos)}...")
        resumo = resumir_bloco(bloco)
        resumos_parciais.append(resumo)

    # 3. Consolidar (Reduce)
    print("Consolidando resumos...")
    texto_consolidado = "\n".join(resumos_parciais)

    # Resumo Final do Resumo
    resumo_final = resumir_bloco(texto_consolidado)
    return resumo_final

# Execução
if __name__ == "__main__":
    resultado = executar_rlm(texto_longo)
    print("\n=== RESUMO DE INTELIGÊNCIA (RLM) ===")
    print(resultado)