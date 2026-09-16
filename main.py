import os
import glob
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions

# ---------- Config ----------
OPENCODE_API_KEY = os.environ.get("OPENCODE_API_KEY", "")
OPENCODE_URL = "https://opencode.ai/zen/v1/chat/completions"
MODEL = "big-pickle"
DOCS_FOLDER = "docs"
CHUNK_SIZE = 500  # palavras por chunk
CHUNK_OVERLAP = 50
TOP_K = 4  # quantos trechos relevantes buscar por pergunta

# ---------- App ----------
app = FastAPI()

# libera o frontend acessar (ajuste em produção pro domínio do seu site)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Banco vetorial (Chroma, local, gratuito) ----------
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embed_fn = embedding_functions.DefaultEmbeddingFunction()  # roda local, sem custo
collection = chroma_client.get_or_create_collection(
    name="documentos", embedding_function=embed_fn
)


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def index_documents():
    """Lê todos os .txt da pasta docs/ e indexa no Chroma (roda uma vez)."""
    existing = collection.count()
    if existing > 0:
        print(f"Já existem {existing} chunks indexados. Pulando indexação.")
        return

    txt_files = glob.glob(f"{DOCS_FOLDER}/*.txt")
    print(f"Encontrados {len(txt_files)} arquivos .txt")

    ids, docs, metadatas = [], [], []
    for filepath in txt_files:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        chunks = chunk_text(content)
        filename = os.path.basename(filepath)
        for i, chunk in enumerate(chunks):
            ids.append(f"{filename}-{i}")
            docs.append(chunk)
            metadatas.append({"source": filename})

    if docs:
        collection.add(ids=ids, documents=docs, metadatas=metadatas)
        print(f"Indexados {len(docs)} chunks no total.")


def buscar_contexto(pergunta: str) -> str:
    resultados = collection.query(query_texts=[pergunta], n_results=TOP_K)
    trechos = resultados["documents"][0] if resultados["documents"] else []
    return "\n\n---\n\n".join(trechos)


def perguntar_big_pickle(pergunta: str, contexto: str) -> str:
    prompt_sistema = (
        "Você é um assistente que responde perguntas com base apenas nos "
        "documentos fornecidos abaixo. Se a resposta não estiver nos "
        "documentos, diga que não sabe.\n\n"
        f"DOCUMENTOS:\n{contexto}"
    )

    resposta = requests.post(
        OPENCODE_URL,
        headers={
            "Authorization": f"Bearer {OPENCODE_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": pergunta},
            ],
        },
        timeout=60,
    )
    resposta.raise_for_status()
    data = resposta.json()
    return data["choices"][0]["message"]["content"]


# ---------- Rotas ----------
class PerguntaRequest(BaseModel):
    pergunta: str


@app.on_event("startup")
def startup_event():
    index_documents()


@app.post("/perguntar")
def perguntar(req: PerguntaRequest):
    contexto = buscar_contexto(req.pergunta)
    resposta = perguntar_big_pickle(req.pergunta, contexto)
    return {"resposta": resposta}


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexados": collection.count()}


# serve o frontend estático (opcional, se quiser tudo junto)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
