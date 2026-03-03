# ==============================================================================
# ARQUIVO: trabalhofinal.py
# CONTEÚDO: Trabalho final da Disciplina Prompt Engineering Avançado
# ALUNO: Guilherme Silveira Rabelo - 238.758-1
# CONTATO: guilherme.rabelo@pcdf.df.gov.br
# ==============================================================================

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI 
from typing import Literal

# Read prompt from file
file_path_few_shot = "./prompt_few_shot.txt"
file_path_cot = "./prompt_cot.txt"

# Set models temperature
few_shot_temperature = 0.0
cot_temperature = 0.2

try:
    # Open the file in read mode ('r')
    with open(file_path_few_shot, 'r', encoding = "utf-8") as file:
        # Read the entire content of the file into a string
        prompt_few_shot = file.read()
    
    # Print or use the file content as a string
    print("Few shot prompt: " + prompt_few_shot[:80])

except FileNotFoundError:
    print(f"Error: The file {file_path_few_shot} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

try:
    # Open the file in read mode ('r')
    with open(file_path_cot, 'r', encoding = "utf-8") as file:
        # Read the entire content of the file into a string
        prompt_cot = file.read()
    
    # Print or use the file content as a string
    print("Chain of thought prompt: " + prompt_cot[:80])

except FileNotFoundError:
    print(f"Error: The file {file_path_few_shot} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama' 
)

app = FastAPI(title="Triagem de Perícias de Local de Crime")

class RelatoOcorrencia(BaseModel):
    relato: str = "Houve crime, há vestígios"
    relator: Literal["agente público", "testemunha", "vítima", "suspeito", "autor", "outros"] = "outros"
    delegacia: str = "Delegacia Geral"

@app.get("/")
def verificar_status():
    return {"status": "online"}

# Rota anterior (Few-Shot) continua aqui...
@app.post("/triagem_pericia")
def classificar_pericia_com_ia(relato: RelatoOcorrencia):

    response = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "system", "content": prompt_few_shot},
                  {"role": "user", "content": relato.relato}],
        temperature= few_shot_temperature
    )
    return {"Classificacao": response.choices[0].message.content}

# [NOVO] Rota Avançada com Chain of Thought (CoT)
@app.post("/triagem_pericia_cot")
def classificar_pericia_com_raciocinio(relato: RelatoOcorrencia):
    print(f"Raciocínando sobre: {relato.relato}...")
    
    # PROMPT CoT: Passo a Passo 
    response = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": prompt_cot},
            {"role": "user", "content": relato.relato}
        ],
        temperature=cot_temperature
    )
    
    return {
        "tecnica": "Chain of Thought (CoT)",
        "analise_completa": response.choices[0].message.content
    }