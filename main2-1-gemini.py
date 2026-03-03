# ==============================================================================
# ARQUIVO: main2-1.py
# OBJETIVO: Implementar Chain of Thought (CoT) para tipificação jurídica complexa
# ==============================================================================
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

local_client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
app = FastAPI(title="IntelliDoc - Módulo CoT")

class Caso(BaseModel):
    relato: str

@app.post("/classificar_simples") # O jeito "burro" (Zero-Shot)
def classificar_rapido(caso: Caso):
    prompt = "Classifique este crime juridicamente: " + caso.relato

    my_temperature = 0.0

    local_response = local_client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        temperature=my_temperature
    )

    # Chamada ao Gemini para testes
    gemini_client = genai.Client()
    gemini_response = gemini_client.models.generate_content(
        model = "gemini-3-flash-preview",
        config = types.GenerateContentConfig(
            temperature = my_temperature),
            contents = caso.relato,
    )

    return {
        "classificacao_local": local_response.choices[0].message.content,
        "classificacao_gemini": gemini_response.text}

@app.post("/classificar_cot") # O jeito "inteligente" (CoT)
def classificar_pensando(caso: Caso):
    # Engenharia de Prompt: Forçando a estrutura de pensamento [2][1]
    prompt_sistema = """
    Analise o relato como um Delegado. Siga ESTRICTAMENTE estes passos:
    
    PASSO 1 - Fatos: O que ocorreu objetivamente?
    PASSO 2 - Violência: Houve grave ameaça ou violência física?
    PASSO 3 - Vontade: A entrega do bem foi espontânea (mesmo que por engano) ou forçada?
    PASSO 4 - Tipificação: Cruzes os passos 2 e 3 para definir o crime (Furto, Roubo, Estelionato, Apropriação Indébita).
    
    Saída:
    RACIOCINIO: [Seus passos]
    VEREDITO: [Apenas o nome do crime]
    """

    my_temperature = 0.2 # Um pouco de criatividade para a explicação
    
    local_response = local_client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": caso.relato}
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
            contents = caso.relato,
    )

    return {
        "analise_local": local_response.choices[0].message.content,
        "analise_gemini": gemini_response.text}

# COMANDO NO TERMINAL: uvicorn main2-1:app --reload