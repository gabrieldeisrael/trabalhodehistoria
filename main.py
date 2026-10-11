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
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"
DOCS_FOLDER = "docs"
CHUNK_SIZE = 200  # palavras por chunk
CHUNK_OVERLAP = 30
TOP_K = 3  # quantos trechos relevantes buscar por pergunta
MAX_TOKENS_RESPOSTA = 300  # limite de tamanho da resposta gerada
INDEX_FILE = "indice.pkl"

# cache simples em memória: (modo_debug, pergunta normalizada) -> resposta
cache_respostas = {}

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


PALAVRA_DEBUG = "debug"

PROMPT_BASE = (
    "Você é a Rita, assistente virtual sobre o Paraguai. "
    "Seu tom é leve, simpático e um pouco descontraído, sem exagero: "
    "pouco ou nenhum emoji e nenhuma piada fora de hora. "
    "Responda como alguém que conhece bem o Paraguai, usando apenas as "
    "informações fornecidas abaixo. Fale como se o conhecimento fosse seu: "
    "não mencione documentos, arquivos, textos, trabalho escolar nem de onde "
    "veio a informação, e considere que o usuário não vê nada disso. "
    "Se a resposta não estiver nas informações, diga com gentileza que não "
    "sabe e sugira outra pergunta sobre o Paraguai. "
    "Seja direta e breve."
)

PROMPT_DEBUG = (
    "Você é a Rita e está falando com o desenvolvedor do projeto. "
    "Pode mencionar o contexto, os documentos e como montou a resposta. "
    "Use apenas as informações fornecidas abaixo. Seja direta e breve."
)


def perguntar_groq(pergunta: str, contexto: str, modo_debug: bool = False) -> str:
    prompt_sistema = (
        (PROMPT_DEBUG if modo_debug else PROMPT_BASE)
        + f"\n\nINFORMAÇÕES:\n{contexto}"
    )

    resposta = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "max_tokens": MAX_TOKENS_RESPOSTA,
            "reasoning_effort": "low",
            "messages": [
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": pergunta},
            ],
        },
        timeout=60,
    )
    if not resposta.ok:
        print(f"[ERRO Groq] Status {resposta.status_code}: {resposta.text}")
    resposta.raise_for_status()
    data = resposta.json()
    return data["choices"][0]["message"]["content"] or ""


# ---------- Rotas ----------
class PerguntaRequest(BaseModel):
    pergunta: str


@app.on_event("startup")
def startup_event():
    index_documents()


@app.post("/perguntar")
def perguntar(req: PerguntaRequest):
    pergunta = req.pergunta.strip()
    modo_debug = pergunta.lower().endswith(PALAVRA_DEBUG)
    if modo_debug:
        pergunta = pergunta[: -len(PALAVRA_DEBUG)].strip()

    chave_cache = (modo_debug, pergunta.lower())
    if chave_cache in cache_respostas:
        return {"resposta": cache_respostas[chave_cache], "cache": True}

    contexto = buscar_contexto(pergunta)
    resposta = perguntar_groq(pergunta, contexto, modo_debug)
    cache_respostas[chave_cache] = resposta
    return {"resposta": resposta, "cache": False}


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexados": len(chunks_texto)}


app.mount("/", StaticFiles(directory="static", html=True), name="static")