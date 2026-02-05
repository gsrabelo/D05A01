# Importamos o FastAPI para criar o servidor web
from fastapi import FastAPI

# Importamos o Pydantic para validar os dados que chegam (segurança de tipos)
# Isso garante que o 'relato' seja sempre um texto, evitando erros grosseiros.
from pydantic import BaseModel

# 1. Instanciação da Aplicação
# A variável 'app' é o coração do sistema. É ela que o Uvicorn vai rodar.
app = FastAPI(
    title="IntelliDoc PCDF - Módulo 1",
    description="API de Triagem de Ocorrências Policiais (Versão Inicial)"
)

# 2. Modelo de Dados (O Formulário)
# Define o "contrato" de dados. O Policial DEVE enviar um JSON com 'relato'.
# O campo 'delegacia' é opcional e tem um valor padrão.
class BoletimOcorrencia(BaseModel):
    relato: str
    delegacia: str = "PCDF Geral"

# 3. Rota de Monitoramento (Health Check)
# Endpoint GET para verificar se o servidor está online.
# Acesso: http://localhost:8000/
@app.get("/")
def verificar_status():
    return {
        "status": "online",
        "sistema": "IntelliDoc API",
        "versao": "1.0.0"
    }

# 4. Rota de Recebimento (Dummy)
# Endpoint POST que simula o recebimento de um B.O.
# Por enquanto, ele apenas devolve o que recebeu (eco), provando que a API funciona.
@app.post("/analisar")
def receber_relato(bo: BoletimOcorrencia):
    # Log no terminal para o aluno ver a ação acontecendo
    print(f"Recebido relato da {bo.delegacia}: {bo.relato}")
    
    return {
        "mensagem": "Relato recebido com sucesso",
        "conteudo_original": bo.relato,
        "status_analise": "Pendente de Inteligência Artificial"
    }

# ==============================================================================
# COMO RODAR ESTE CÓDIGO:
# No terminal: uvicorn main_basico:app --reload
# ==============================================================================