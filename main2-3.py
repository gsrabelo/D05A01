# ==============================================================================
# ARQUIVO: main2-3.py (INTEGRAÇÃO TOTAL)
# OBJETIVO: Receber Texto + Imagem e validar consistência (Detector de Fraude)
# ==============================================================================
from fastapi import FastAPI, UploadFile, File, Form
from openai import OpenAI
import base64

app = FastAPI(title="IntelliDoc - Verificador de Verdade")
client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

@app.post("/validar_coerencia")
async def validar(
    relato: str = Form(...), # O Texto da vítima
    evidencia: UploadFile = File(...) # A Foto
):
    # PASSO 1: A IA "Vê" (Usando Moondream)
    img_bytes = await evidencia.read()
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
        temperature=0.1
    )
    descricao_visual = resp_visao.choices[0].message.content
    
    # PASSO 2: A IA "Julga" (Usando Llama 3.2 com CoT)
    # Aqui usamos Engenharia de Contexto para cruzar dados
    prompt_analise = f"""
    Você é um perito criminal. Verifique a coerência das provas.
    
    RELATO DA VÍTIMA: "{relato}"
    
    O QUE A PERÍCIA (IA) VIU NA FOTO: "{descricao_visual}"
    
    TAREFA:
    Compare o relato com a foto. Existe contradição?
    Exemplo: Vítima diz "carro vermelho", foto mostra "moto azul" -> CONTRADIÇÃO.
    
    VEREDITO: [CONSISTENTE / INCONSISTENTE]
    JUSTIFICATIVA: [Explique por quê]
    """
    
    resp_final = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt_analise}],
        temperature=0.0
    )
    
    return {
        "relato_vitima": relato,
        "analise_ia_foto": descricao_visual,
        "conclusao_pericia": resp_final.choices[0].message.content
    }


# COMANDO: uvicorn main2-3:app --reload