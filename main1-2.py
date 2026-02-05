from fastapi import FastAPI
from pydantic import BaseModel
# [NOVO] Importamos a biblioteca para falar com a IA
from openai import OpenAI 

# [NOVO] Configuração do Cliente de IA
# Aponta para o Ollama rodando no seu PC (localhost), garantindo privacidade.
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama' # Chave falsa, necessária apenas para a biblioteca não reclamar
)

app = FastAPI(title="IntelliDoc PCDF - Módulo 1")

class BoletimOcorrencia(BaseModel):
    relato: str
    delegacia: str = "PCDF Geral"

@app.get("/")
def verificar_status():
    return {"status": "online"}

@app.post("/analisar")
def receber_relato(bo: BoletimOcorrencia):
    return {"recebido": bo.relato}

# [NOVO] Rota Inteligente v1 (Zero-Shot)
@app.post("/analisar_inteligente")
def analisar_com_ia(bo: BoletimOcorrencia):
    print(f"Enviando para o Llama: {bo.relato}...")
    
    # PROMPT SIMPLES (ZERO-SHOT)
    # Damos a ordem direta, sem exemplos.
    prompt_sistema = """
    Você é um especialista criminal da PCDF.
    Classifique o relato ABAIXO como: FURTO, ROUBO ou ESTELIONATO.
    Responda apenas a classificação.
    """
    
    # Chamada ao Modelo (O "Estagiário")
    response = client.chat.completions.create(
        model="llama3.2", # O modelo leve (3B) que baixamos
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": bo.relato}
        ],
        temperature=0.2 # Baixa criatividade para evitar invenções
    )
    
    return {
        "relato": bo.relato,
        "classificacao_ia": response.choices[0].message.content
    }