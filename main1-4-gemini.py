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

# Rota anterior (Few-Shot) continua aqui...
@app.post("/analisar_inteligente")
def analisar_com_ia(bo: BoletimOcorrencia):
    prompt_sistema = """
    Você é um classificador. Classifique em: [FURTO, ROUBO, ESTELIONATO].
    Exemplos:
    "Arma apontada" -> ROUBO
    "Sumiu da mesa" -> FURTO
    """

    my_temperature = 0.0

    local_response = local_client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "system", "content": prompt_sistema},
                  {"role": "user", "content": bo.relato}],
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
        "classificacao_local": local_response.choices[0].message.content,
        "classificacao_gemini": gemini_response.text}

# [NOVO] Rota Avançada com Chain of Thought (CoT)
@app.post("/analisar_cot")
def analisar_raciocinio(bo: BoletimOcorrencia):
    print(f"Raciocínando sobre: {bo.relato}...")
    
    # PROMPT CoT: Passo a Passo
    prompt_cot = """
    Aja como um Delegado. Analise o caso seguindo este roteiro mental:
    
    PASSO 1: Fatos - Liste o que realmente aconteceu.
    PASSO 2: Violência - Houve grave ameaça ou violência física? (Sim/Não)
    PASSO 3: Subtração - O bem foi retirado ou entregue voluntariamente?
    
    Com base nisso, defina a tipificação penal.
    
    Formato de Resposta:
    RACIOCINIO: [Sua análise detalhada]
    VEREDITO: [FURTO, ROUBO ou ESTELIONATO]
    """

    my_temperature = 0.1 # Leve criatividade para escrever a explicação
    
    local_response = local_client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": prompt_cot},
            {"role": "user", "content": bo.relato}
        ],
        temperature=my_temperature
    )

    # Chamada ao Gemini para testes
    gemini_client = genai.Client()
    gemini_response = gemini_client.models.generate_content(
        model = "gemini-3-flash-preview",
        config = types.GenerateContentConfig(
            system_instruction = prompt_cot,
            temperature = my_temperature),
            contents = bo.relato,
    )
    
    return {
        "tecnica": "Chain of Thought (CoT)",
        "analise_completa_local": local_response.choices[0].message.content,
        "analise_completa_gemini": gemini_response.text
    }