import os
import glob
import pickle
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Config ----------
OPENCODE_API_KEY = os.environ.get("OPENCODE_API_KEY", "")
OPENCODE_URL = "https://opencode.ai/zen/v1/chat/completions"
MODEL = "big-pickle"
DOCS_FOLDER = "docs"
CHUNK_SIZE = 300  # palavras por chunk
CHUNK_OVERLAP = 50
TOP_K = 4  # quantos trechos relevantes buscar por pergunta
INDEX_FILE = "indice.pkl"

# ---------- App ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Estado do índice (em memória) ----------
chunks_texto = []
chunks_fonte = []
vectorizer = None
matriz_tfidf = None


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
    """Lê os .txt de docs/, quebra em chunks e monta o índice TF-IDF."""
    global chunks_texto, chunks_fonte, vectorizer, matriz_tfidf

    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "rb") as f:
            dados = pickle.load(f)
        chunks_texto = dados["chunks_texto"]
        chunks_fonte = dados["chunks_fonte"]
        vectorizer = dados["vectorizer"]
        matriz_tfidf = dados["matriz_tfidf"]
        print(f"[OK] Índice carregado do disco ({len(chunks_texto)} chunks).")
        return

    txt_files = glob.glob(f"{DOCS_FOLDER}/*.txt")
    print(f"Encontrados {len(txt_files)} arquivos .txt")

    chunks_texto, chunks_fonte = [], []
    for filepath in txt_files:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        filename = os.path.basename(filepath)
        for chunk in chunk_text(content):
            chunks_texto.append(chunk)
            chunks_fonte.append(filename)

    if not chunks_texto:
        print("[AVISO] Nenhum chunk pra indexar.")
        return

    vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True)
    matriz_tfidf = vectorizer.fit_transform(chunks_texto)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump(
            {
                "chunks_texto": chunks_texto,
                "chunks_fonte": chunks_fonte,
                "vectorizer": vectorizer,
                "matriz_tfidf": matriz_tfidf,
            },
            f,
        )
    print(f"[OK] Indexados {len(chunks_texto)} chunks no total.")


def buscar_contexto(pergunta: str) -> str:
    if vectorizer is None or matriz_tfidf is None or not chunks_texto:
        return ""

    vetor_pergunta = vectorizer.transform([pergunta])
    similaridades = cosine_similarity(vetor_pergunta, matriz_tfidf)[0]
    top_indices = similaridades.argsort()[::-1][:TOP_K]

    trechos = [chunks_texto[i] for i in top_indices if similaridades[i] > 0]
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
    return {"status": "ok", "chunks_indexados": len(chunks_texto)}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
