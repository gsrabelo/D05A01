# main5-1.py - Versão de Produção (Auditável)
from fastapi import FastAPI, Request
from pydantic import BaseModel
from openai import OpenAI
import time

# Importações de Bibliotecas de Produção (Instalar antes: pip install structlog prometheus-fastapi-instrumentator)
import structlog # Biblioteca de logs profissionais
from prometheus_fastapi_instrumentator import Instrumentator # Métricas

# 1. Configuração de Logs Estruturados (JSON) ao invés de texto solto
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"), # Adiciona data/hora ISO
        structlog.processors.JSONRenderer() # Converte tudo para JSON final
    ]
)
logger = structlog.get_logger() # Cria o objeto que usaremos para logar

app = FastAPI(title="IntelliDoc PCDF - PROD")

# 2. Instrumentação Automática (Métricas para Prometheus)
# Isso cria uma rota /metrics que expõe dados de performance
Instrumentator().instrument(app).expose(app)

client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

# Modelo de Dados de Entrada
class Boletim(BaseModel):
    relato: str
    policial_id: str # Novo campo para auditoria - quem está usando

@app.post("/analisar_auditado")
async def analisar_bo(bo: Boletim, request: Request):
    # Recebe um relato, classifica com IA, e gera logs detalhados de auditoria.
    
    # PASSO 1: Preparar o Contexto do Log
    # "Bind" significa: Anexe esses dados em TODOS os logs gerados nesta função.
    # Se der erro lá na frente, saberemos quem foi (policial_id) e de onde veio (IP).
    log = logger.bind(policial=bo.policial_id, ip=request.client.host)
    log.info("analise_iniciada", tamanho_texto=len(bo.relato))

    start_time = time.time()

    try:
	    # PASSO 2: Chamada para a Inteligência Artificial
        response = client.chat.completions.create(
            model="llama3.2",
            messages=[
                {"role": "system", "content": "Classifique o crime: Furto, Roubo ou Estelionato."},
                {"role": "user", "content": bo.relato}
            ],
            temperature=0.0 # Determinístico
        )
        resultado = response.choices.message[0].content

        # PASSO 3: Cálculo de Performance
        tempo_gasto = time.time() - start_time
        
        # PASSO 4: Log de Sucesso (Auditoria Final)
        log.info("analise_concluida",
                 classificacao=resultado,
                 tempo_s=round(tempo_gasto, 2))

        return {"classificacao": resultado, "tempo_processamento": tempo_gasto}

    except Exception as e:
        # Log de Erro (Crucial para suporte)
        log.error("erro_processamento", erro=str(e))
        
        return {"status": "erro", "mensagem": "Falha interna. TI notificada."}

# COMANDO: uvicorn main5-1:app --reload