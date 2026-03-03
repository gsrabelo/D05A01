import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Access the API key using os.getenv()
api_key = os.getenv('GEMINI_API_KEY')

if api_key is None:
    print("Gemini API key not found. Please set the environment variable.")
else:
    print(f"Successfully retrieved Gemini API key starting with: {api_key[:8]}")

os.environ['GEMINI_API_KEY'] = api_key

class BoletimOcorrencia(BaseModel):
    relato: str = "Relato padrão"
    delegacia: str = "PCDF Geral"

# [NOVO] Rota Inteligente v1 (Zero-Shot)
def analisar_com_gemini(bo: BoletimOcorrencia):
    #print(f"Enviando para o Gemini: {bo.relato}...")
    
    # PROMPT SIMPLES (ZERO-SHOT)
    # Damos a ordem direta, sem exemplos.
    prompt_sistema = """
    Você é um especialista criminal da PCDF.
    Classifique o relato ABAIXO como: FURTO, ROUBO ou ESTELIONATO.
    Responda apenas a classificação.
    """

    client = genai.Client()

    response = client.models.generate_content(
        model = "gemini-3-flash-preview",
        config = types.GenerateContentConfig(
            system_instruction = prompt_sistema),
            contents = "Deixei meu celular em cima da mesa e levaram ele"
    )

    print(response.text)
    
    return {
        "relato": "teste",
        "classificacao_ia": "teste"
    }

analisar_com_gemini(["teste","teste"])