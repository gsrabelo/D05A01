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
import chromadb
from chromadb.utils import embedding_functions

app = FastAPI(title="Triagem de Perícias de Local de Crime")

# ==============================================================================
# PROMPT READING
# ==============================================================================

# Read prompt from file
file_path_few_shot = "./prompt_few_shot.txt"
file_path_cot = "./prompt_cot.txt"
file_path_com_os = "./prompt_com_os.txt"

try:
    with open(file_path_few_shot, 'r', encoding = "utf-8") as file:
        prompt_few_shot = file.read()
    print(f"Few shot prompt: {prompt_few_shot[:80]}")

except FileNotFoundError:
    print(f"Error: The file {file_path_few_shot} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

try:
    with open(file_path_cot, 'r', encoding = "utf-8") as file:
        prompt_cot = file.read()
    print(f"Chain of thought prompt: {prompt_cot[:80]}")

except FileNotFoundError:
    print(f"Error: The file {file_path_cot} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

try:
    with open(file_path_com_os, 'r', encoding = "utf-8") as file:
        prompt_com_os = file.read()
    print(f"Prompt com ordem de serviço: {prompt_com_os[:80]}")

except FileNotFoundError:
    print(f"Error: The file {file_path_com_os} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# ==============================================================================
# TEMPERATURE SETTINGS
# ==============================================================================

few_shot_temperature = 0.0
cot_temperature = 0.2
rag_temperature = 0.0

# ==============================================================================
# COLLECTION SETTINGS
# ==============================================================================

# Uso de pasta local './banco_vetorial' para salvar os dados
chroma_client = chromadb.PersistentClient(path="./banco_vetorial")

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text"
)

collection = chroma_client.get_or_create_collection(
    name="normas_pcdf",
    embedding_function=ollama_ef
)

with open("./data/ordem_de_servico.txt", "r", encoding="utf-8") as f:
    texto_completo = f.read()

print(f"Leitura de ordem de serviço: {texto_completo[:80]}...")

# Chunking
documentos = [p for p in texto_completo.split("\n") if p.strip()] # Divide por linha vazia
ids = [f"doc_{i}" for i in range(len(documentos))] # IDs: doc_0, doc_1...

print(f"Processando {len(documentos)} pedaços de informação...")

# Upsert
collection.upsert(
    documents=documentos,
    ids=ids
)

print("Memória criada com sucesso! Dados vetorizados no ChromaDB.")

# ==============================================================================
# OLLAMA CLIENT
# ==============================================================================

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama' 
)

# ==============================================================================
# PYDANTIC MODELS
# ============================================================================== 
class RelatoOcorrencia(BaseModel):
    texto: str = "Houve crime, há vestígios"
    relator: Literal["agente público", "testemunha", "vítima", "suspeito", "autor", "outros"] = "outros"
    delegacia: str = "Delegacia Geral"

# ==============================================================================
# ENDPOINT CONFIGURATION
# ============================================================================== 

# Root endpoint
@app.get("/")
def verificar_status():
    return {"status": "online"}

# Few shot endpoint
@app.post("/triagem_pericia")
def classificar_pericia_com_ia(relato: RelatoOcorrencia):

    response = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "system", "content": prompt_few_shot},
                  {"role": "user", "content": relato.texto}],
        temperature= few_shot_temperature
    )
    return {"Classificacao": response.choices[0].message.content}

# Chain of thought endpoint
@app.post("/triagem_pericia_cot")
def classificar_pericia_com_raciocinio(relato: RelatoOcorrencia):
    print(f"Raciocínando sobre: {relato.texto}...")
    
    # PROMPT CoT: Passo a Passo 
    response = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": prompt_cot},
            {"role": "user", "content": relato.texto}
        ],
        temperature=cot_temperature
    )
    
    return {
        "tecnica": "Chain of Thought (CoT)",
        "analise_completa": response.choices[0].message.content
    }

# RAG endpoint
@app.post("/triagem_com_memoria")
def classificar_pericia_com_memoria(relato: RelatoOcorrencia):
    print(f"Buscando informações para: {relato.texto}")

    # PASSO 1: Retrieval (Recuperação)
    # Buscamos no banco os 3 trechos mais parecidos com a pergunta
    resultados = collection.query(
        query_texts=[relato.texto],
        n_results=6 # Traz os top 3 pedaços mais relevantes
    )

    # Juntamos os pedaços recuperados em um único texto
    contexto_recuperado = "\n".join(resultados['documents'][0])
    print(f"Contexto encontrado: {contexto_recuperado}")

    # PASSO 2: Augmented Generation (Geração Aumentada)
    # Colamos o contexto no prompt do sistema
    prompt_sistema = prompt_com_os + contexto_recuperado

    response = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": relato.texto}
        ],
        temperature = rag_temperature
    )

    return {
        "pergunta": relato.texto,
        "resposta": response.choices[0].message.content,
        "fontes_utilizadas": resultados['documents']
    }