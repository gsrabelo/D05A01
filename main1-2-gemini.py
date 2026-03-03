from fastapi import FastAPI
from pydantic import BaseModel
# [NOVO] Importamos a biblioteca para falar com a IA
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

# [NOVO] Configuração do Cliente de IA
# Aponta para o Ollama rodando no seu PC (localhost), garantindo privacidade.
local_client = OpenAI(
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

    my_temperature = 0.2
    
    # Chamada ao Modelo (O "Estagiário")
    local_response = local_client.chat.completions.create(
        model="llama3.2", # O modelo leve (3B) que baixamos
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": bo.relato}
        ],
        temperature=my_temperature # Baixa criatividade para evitar invenções
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
        "relato": bo.relato,
        "classificacao_local": local_response.choices[0].message.content,
        "classificacao_gemini": gemini_response.text
    }