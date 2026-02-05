from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI 

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama' 
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

@app.post("/analisar_inteligente")
def analisar_com_ia(bo: BoletimOcorrencia):
    print(f"Processando com Few-Shot...")
    
    # [MUDANÇA AQUI] PROMPT AVANÇADO (FEW-SHOT + CONSTRAINTS)
    # Ensinamos o padrão através de exemplos.
    prompt_sistema = """
    Você é um classificador automático da PCDF.
    
    REGRAS OBRIGATÓRIAS:
    1. Analise o relato.
    2. Classifique ESTRITAMENTE em uma destas categorias: [FURTO, ROUBO, ESTELIONATO].
    3. Responda APENAS a palavra da categoria. Sem ponto final.
    
    EXEMPLOS DE TREINAMENTO (Siga este padrão):
    
    Relato: "Levaram meu celular da mesa sem eu ver."
    Classificação: FURTO
    
    Relato: "Dois homens armados levaram meu carro."
    Classificação: ROUBO
    
    Relato: "Recebi um link falso e perdi dinheiro."
    Classificação: ESTELIONATO
    
    Agora classifique o novo relato:
    """
    
    response = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": bo.relato}
        ],
        temperature=0.0 # Temperatura ZERO para máxima precisão
    )
    
    return {
        "tecnica": "Few-Shot Prompting",
        "classificacao_ia": response.choices[0].message.content
    }