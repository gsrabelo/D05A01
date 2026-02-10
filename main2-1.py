# ==============================================================================
# ARQUIVO: main2-1.py
# OBJETIVO: Implementar Chain of Thought (CoT) para tipificação jurídica complexa
# ==============================================================================
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
app = FastAPI(title="IntelliDoc - Módulo CoT")

class Caso(BaseModel):
    relato: str

@app.post("/classificar_simples") # O jeito "burro" (Zero-Shot)
def classificar_rapido(caso: Caso):
    prompt = "Classifique este crime juridicamente: " + caso.relato
    resp = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return {"classificacao": resp.choices[0].message.content}

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
    
    resp = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": caso.relato}
        ],
        temperature=0.2 # Um pouco de criatividade para a explicação
    )
    return {"analise": resp.choices[0].message.content}

# COMANDO NO TERMINAL: uvicorn main2-1:app --reload