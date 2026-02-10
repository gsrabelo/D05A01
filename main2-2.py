# ==============================================================================
# ARQUIVO: main2-2.py
# OBJETIVO: Receber uma imagem (arquivo) e gerar um laudo descritivo
# IMPORTANTE: Necessário ter rodado ollama pull moondream
# ==============================================================================
from fastapi import FastAPI, UploadFile, File
from openai import OpenAI
import base64

app = FastAPI(title="IntelliDoc - Visão")
client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

def encode_image(file_content):
    return base64.b64encode(file_content).decode('utf-8')

@app.post("/descrever_evidencia")
async def ver_imagem(
    arquivo: UploadFile = File(...),
    foco: str = "Descreva a imagem para um laudo policial."
):
    print(f"Processando imagem: {arquivo.filename}")
    conteudo = await arquivo.read()
    img_b64 = encode_image(conteudo)
    
    # response = client.chat.completions.create(
    #     model="moondream", # Modelo multimodal leve
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": foco, # O prompt guia o olhar da IA
    #             "images": [img_b64]
    #         }
    #     ]
    # )
    # return {"laudo_visual": response.choices[0].message.content}

    response = client.chat.completions.create(
        model="moondream",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": foco},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        temperature=0.2
    )
    return {"laudo_visual": response.choices[0].message.content}


# COMANDO: uvicorn main2-2:app --reload