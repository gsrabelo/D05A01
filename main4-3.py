# main4-3.py - Calculadora de Viabilidade
def calcular_custos():
    print("=== CALCULADORA DE CUSTOS: INTELLIDOC vs CLOUD ===")

    # Premissas
    num_delegacias = 30
    bos_por_dia = 50  # Média por delegacia
    media_palavras_bo = 500
    dias_uteis = 22

    # Conversão: 1 palavra ~= 1.3 tokens
    tokens_por_bo = media_palavras_bo * 1.3
    total_tokens_mes = num_delegacias * bos_por_dia * tokens_por_bo * dias_uteis

    print(f"Volume Mensal de Processamento: {total_tokens_mes:,.0f} tokens")

    # Custo Cloud (Ex: GPT-4o Mini - US$ 0.15 / 1M tokens entrada + US$ 0.60 saída)
    # Vamos fazer uma média de US$ 0.40 por milhão de tokens (input+output)
    custo_milhao_token_usd = 0.40
    cotacao_dolar = 5.80

    custo_cloud_usd = (total_tokens_mes / 1_000_000) * custo_milhao_token_usd
    custo_cloud_brl = custo_cloud_usd * cotacao_dolar

    # Custo Local (Energia)
    # Supondo PC ligado 24h consumindo 200W a mais por causa da IA (estimativa alta)
    kwh_custo = 0.80 # R$
    consumo_kwh_mes = (0.2 * 24 * 30) * num_delegacias # 0.2kW * 24h * 30d * 30PCs
    custo_local_brl = consumo_kwh_mes * kwh_custo

    print("\n--- COMPARATIVO MENSAL ---")
    print(f"Custo Cloud (OpenAI/Azure): R$ {custo_cloud_brl:,.2f}")
    print(f"Custo Local (Energia):      R$ {custo_local_brl:,.2f}")

    economia = custo_cloud_brl - custo_local_brl
    print(f"\nECONOMIA MENSAL ESTIMADA:   R$ {economia:,.2f}")
    print(f"ECONOMIA ANUAL ESTIMADA:    R$ {economia * 12:,.2f}")

    print("\nAlém da economia, o custo da PRIVACIDADE é incalculável.")

if __name__ == "__main__":
    calcular_custos()