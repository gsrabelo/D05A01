from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI 

# Libraries for GEMINI API usage
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the API key using os.getenv()
api_key = os.getenv('GEMINI_API_KEY')

if api_key is None:
    print("Gemini API key not found. Please set the environment variable.")
else:
    print(f"Successfully retrieved Gemini API key starting with: {api_key[:8]}")

os.environ['GEMINI_API_KEY'] = api_key

local_client = OpenAI(
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

    my_temperature = 0.0 # Temperatura ZERO para máxima precisão
    
    local_response = local_client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": bo.relato}
        ],
        temperature=my_temperature 
    )

    # Chamada ao Gemini para testes
    gemini_client = genai.Client()
    gemini_response = gemini_client.models.generate_content(
        model = "gemini-3-flash-preview",
        config = types.GenerateContentConfig(
            system_instruction = prompt_sistema,
            temperature = my_temperature),
            contents = bo.relato,
    )

    return {
        "tecnica": "Few-Shot Prompting",
        "classificacao_local": local_response.choices[0].message.content,
        "classificacao_gemini": gemini_response.text
    }