# main3-1.py - O Indexador (Cria a Memória)
import chromadb
from chromadb.utils import embedding_functions

# 1. Configuração do Banco Vetorial (ChromaDB)
# Criamos uma pasta local './banco_vetorial' para salvar os dados
chroma_client = chromadb.PersistentClient(path="./banco_vetorial")

# 2. Configurar o Modelo de Embedding (Ollama)
# Usamos o 'nomic-embed-text' que baixamos. Ele traduz texto -> números.
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text"
)

# 3. Criar (ou carregar) a coleção (tabela)
collection = chroma_client.get_or_create_collection(
    name="inqueritos_pcdf",
    embedding_function=ollama_ef
)

# 4. Ler o arquivo de texto (Simulando o PDF)
with open("./data/inquerito_exemplo.txt", "r", encoding="utf-8") as f:
    texto_completo = f.read()

# 5. Chunking (Fatiamento)
# Não podemos salvar tudo de uma vez. Dividimos em parágrafos.
# Na prática, usamos bibliotecas como LangChain para isso, aqui faremos manual.
documentos = [p for p in texto_completo.split("\n") if p.strip()] # Divide por linha vazia
ids = [f"doc_{i}" for i in range(len(documentos))] # IDs: doc_0, doc_1...

print(f"Processando {len(documentos)} pedaços de informação...")

# 6. Salvar no Banco (Upsert)
collection.upsert(
    documents=documentos,
    ids=ids
)

print("Memória criada com sucesso! Dados vetorizados no ChromaDB.")