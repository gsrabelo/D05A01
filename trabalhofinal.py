# ==============================================================================
# ARQUIVO: trabalhofinal.py
# CONTEÚDO: Trabalho final da Disciplina Prompt Engineering Avançado
# ALUNO: Guilherme Silveira Rabelo - 238.758-1
# CONTATO: guilherme.rabelo@pcdf.df.gov.br
# ==============================================================================

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from openai import OpenAI 
from typing import Literal
import chromadb
from chromadb.utils import embedding_functions
import base64
import ollama

app = FastAPI(title="Encaminhamento de Perícias de Local de Crime")

# ==============================================================================
# PROMPT READING
# ==============================================================================

# Read prompt from file
file_path_few_shot = "./prompt_few_shot.txt"
file_path_cot = "./prompt_cot.txt"
file_path_com_os = "./prompt_com_os.txt"
file_path_com_imagem = "./prompt_com_imagem.txt"

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

try:
    with open(file_path_com_imagem, 'r', encoding = "utf-8") as file:
        prompt_com_imagem = file.read()
    print(f"Prompt com imagem: {prompt_com_imagem[:80]}")

except FileNotFoundError:
    print(f"Error: The file {file_path_com_imagem} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# ==============================================================================
# TEMPERATURE SETTINGS
# ==============================================================================

few_shot_temperature = 0.0
cot_temperature = 0.2
rag_temperature = 0.0
image_extraction_temperature = 0.1
image_temperature = 0.1

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

class Ocorrencia(BaseModel):
    relato: str = "Houve crime, há vestígios"
    comunicante: Literal["agente público", "testemunha", "vítima", "suspeito", "autor", "outros"] = "outros"
    delegacia: str = "Delegacia Geral"

# ==============================================================================
# IMAGE CODING FUNCTION
# ============================================================================== 

def encode_image(file_content):
    return base64.b64encode(file_content).decode('utf-8')

# ==============================================================================
# TRANSLATE FUNCTION
# ============================================================================== 

def translate_text(text, source_lang, target_lang, model='llama3.2'):

    system_prompt = f"You are a professional translator. Translate the following text from {source_lang} to {target_lang}. Provide only the translation, no extra explanations."
    
    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': text},
        ],
    )
    return response['message']['content']

# ==============================================================================
# ENDPOINT CONFIGURATION
# ============================================================================== 

# Root endpoint
@app.get("/")
def verificar_status():
    return {"status": "online"}

# Few shot endpoint
@app.post("/encaminhamento_pericia_fewshot")
def classificar_pericia_com_ia(ocorrencia: Ocorrencia):

    response = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "system", "content": prompt_few_shot},
                  {"role": "user", "content": ocorrencia.relato}],
        temperature= few_shot_temperature
    )
    #json_response = jsonable_encoder(response.choices[0].message.content)
    #print(f"Resposta few-shot: \n{json_response}")
    return {
        "tecnica": "Few-Shot",
        "relato": ocorrencia.relato,
        "encaminhamento_JSON": response.choices[0].message.content
    }

# Chain of thought endpoint
@app.post("/encaminhamento_pericia_cot")
def classificar_pericia_com_raciocinio(ocorrencia: Ocorrencia):
    print(f"Raciocínando sobre: {ocorrencia.relato}...")
    
    # PROMPT CoT: Passo a Passo 
    response = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": prompt_cot},
            {"role": "user", "content": ocorrencia.relato}
        ],
        temperature=cot_temperature
    )
    
    return {
        "tecnica": "Chain of Thought (CoT)",
        "encaminhamento_JSON": response.choices[0].message.content
    }

# RAG endpoint
@app.post("/encaminhamento_com_memoria")
def classificar_pericia_com_memoria(ocorrencia: Ocorrencia):
    print(f"Buscando informações para: {ocorrencia.relato}")

    # PASSO 1: Retrieval (Recuperação)
    # Buscamos no banco os 3 trechos mais parecidos com a pergunta
    resultados = collection.query(
        query_texts=[ocorrencia.relato],
        n_results=7 # Traz os top 7 pedaços mais relevantes
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
            {"role": "user", "content": ocorrencia.relato}
        ],
        temperature = rag_temperature
    )

    return {
        "tecnica": "Retrieval-Augmented Generation (RAG)",
        "encaminhamento_JSON": response.choices[0].message.content,
        "resultados_encontrados": contexto_recuperado
    }

# Image endpoint
@app.post("/encaminhamento_baseado_em_imagem")
async def classificar_pericia_com_imagem(
    foto_local: UploadFile = File(...) # A Foto
):
    # PASSO 1: A IA "Vê" (Usando Moondream)
    img_bytes = await foto_local.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')

    resp_visao = client.chat.completions.create(
        model="moondream",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Descreva objetivamente o que há nesta imagem."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        temperature = image_extraction_temperature
    )
    descricao_visual_eng = resp_visao.choices[0].message.content
    descricao_visutal_ptbr = translate_text(descricao_visual_eng, "English", "Portuguese")
    
    # PASSO 2: A IA "Julga" (Usando Llama 3.2 com CoT)
    # Aqui usamos Engenharia de Contexto para cruzar dados
    prompt_analise = prompt_com_imagem + descricao_visutal_ptbr
    
    resp_final = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt_analise}],
        temperature=0.0
    )
    
    return {
        "descricao_da_foto": descricao_visutal_ptbr,
        "encaminhamento_JSON": resp_final.choices[0].message.content
    }