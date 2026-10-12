import os
import re
import glob
import pickle
import requests
import edge_tts
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Config ----------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"
DOCS_FOLDER = "docs"
CHUNK_SIZE = 400  # palavras por chunk
CHUNK_OVERLAP = 30
TOP_K = 3  # quantos trechos relevantes buscar por pergunta
MAX_TOKENS_RESPOSTA = 600  # limite de tamanho da resposta gerada
TEMPERATURA = 0.3  # 0 = mais fiel e previsível; 1 = mais criativo (e mais propenso a inventar)
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

    txt_files = glob.glob(f"{DOCS_FOLDER}/*.txt")
    if os.path.exists(INDEX_FILE):
        index_mtime = os.path.getmtime(INDEX_FILE)
        if txt_files and all(os.path.getmtime(f) <= index_mtime for f in txt_files):
            with open(INDEX_FILE, "rb") as f:
                dados = pickle.load(f)
            chunks_texto = dados["chunks_texto"]
            chunks_fonte = dados["chunks_fonte"]
            vectorizer = dados["vectorizer"]
            matriz_tfidf = dados["matriz_tfidf"]
            print(f"[OK] Índice carregado do disco ({len(chunks_texto)} chunks).")
            return

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
    "Você é a Rita, assistente virtual especialista na independência e na formação do Paraguai. "
    "Diretrizes: "
    "1. Precisão factual absoluta e tolerância zero a invenções: afirme apenas o que está nas informações fornecidas abaixo. "
    "2. Tom leve, simpático e um pouco descontraído, com pouco ou nenhum emoji. "
    "3. Pode usar hipérbole épica leve para os feitos heroicos da independência, mas só na linguagem: nunca em números, datas ou nomes, e nunca em guerras, mortes, ditaduras ou repressão. "
    "4. Piadinhas brancas só raramente, no máximo uma por conversa, nunca nos temas pesados acima; a piada não pode afirmar fato novo nem atribuir ações ou falas a pessoas reais. Na dúvida, não faça piada. "
    "5. Fale como se o conhecimento fosse seu: não mencione documentos, arquivos, textos, trabalho escolar nem de onde veio a informação, e considere que o usuário não vê nada disso. "
    "6. Se a resposta não estiver nas informações, diga com gentileza que não sabe e sugira outra pergunta sobre o Paraguai; recuse com gentileza assuntos que não sejam o Paraguai. "
    "7. Não invente nem deduza rankings, comparações, apelidos ou superlativos (primeiro, segundo, maior, único) que não estejam escritos nas informações. "
    "8. Onde as fontes divergem (como o número de mortos da Guerra da Tríplice Aliança), diga que há divergência e não escolha um número. "
    "9. Seja direta e breve. Em pedidos abertos, como fatos curiosos, escolha de três a cinco fatos e responda de forma curta."
)

PROMPT_DEBUG = (
    "Você é a Rita e está falando com o desenvolvedor do projeto. "
    "Pode mencionar o contexto, os documentos e como montou a resposta. "
    "Use apenas as informações fornecidas abaixo e não invente nada: nada de "
    "apelidos, rankings ou comparações que não estejam escritos. Seja direta e breve."
)


# ---------- Busca na internet (Groq browser_search) ----------
# Desligue sem mexer no código: no Render, crie RITA_BUSCA_WEB=0
BUSCA_WEB = os.environ.get("RITA_BUSCA_WEB", "1") == "1"
MAX_TOKENS_COM_BUSCA = 1500  # com busca o modelo gasta mais tokens "pensando" e navegando

FONTES_WEB = [
    "https://www.britannica.com/place/Paraguay/History",
    "https://www.britannica.com/event/War-of-the-Triple-Alliance",
    "https://www.britannica.com/event/Chaco-War",
    "https://www.portalguarani.com/",
]

PROMPT_WEB = (
    " Exceção importante: se a resposta não estiver nas informações abaixo, em vez de dizer "
    "que não sabe, pesquise na internet, dando preferência a estas fontes: "
    + ", ".join(FONTES_WEB)
    + ". Só afirme o que encontrar de forma clara. Ao usar a internet, você pode citar em uma "
    "frase o nome da fonte pública (por exemplo, a Enciclopédia Britannica), mas continue sem "
    "mencionar documentos ou arquivos. Só diga que não sabe se também não encontrar nada "
    "confiável. Pesquise apenas sobre o Paraguai; outros assuntos você recusa com gentileza."
)


def _limpar_resposta(texto: str) -> str:
    """Tira as marcas de citação que a busca às vezes deixa no texto, tipo 【2†L55-L60】."""
    return re.sub(r"【[^】]*】", "", texto or "").strip()


def _chamar_groq(pergunta: str, contexto: str, modo_debug: bool, com_busca: bool):
    prompt_sistema = (
        (PROMPT_DEBUG if modo_debug else PROMPT_BASE)
        + (PROMPT_WEB if com_busca else "")
        + f"\n\nINFORMAÇÕES:\n{contexto}"
    )
    corpo = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS_RESPOSTA,
        "reasoning_effort": "low",
        "temperature": TEMPERATURA,
        "messages": [
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": pergunta},
        ],
    }
    if com_busca:
        corpo["tools"] = [{"type": "browser_search"}]
        corpo["max_tokens"] = MAX_TOKENS_COM_BUSCA
    return requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json=corpo,
        timeout=90,
    )


def perguntar_groq(pergunta: str, contexto: str, modo_debug: bool = False) -> str:
    resposta = _chamar_groq(pergunta, contexto, modo_debug, BUSCA_WEB)
    if not resposta.ok and BUSCA_WEB:
        # se a busca falhar (limite, erro do Groq...), responde só com o texto do site
        print(f"[ERRO Groq com busca] Status {resposta.status_code}: {resposta.text}")
        resposta = _chamar_groq(pergunta, contexto, modo_debug, False)
    if not resposta.ok:
        print(f"[ERRO Groq] Status {resposta.status_code}: {resposta.text}")
    resposta.raise_for_status()
    try:
        data = resposta.json()
        choices = data.get("choices", [])
        if not choices:
            return "Desculpe, não consegui gerar uma resposta no momento."
        content = choices[0].get("message", {}).get("content", "")
        return _limpar_resposta(content)
    except Exception as e:
        print(f"[ERRO Parse Groq] {e}")
        return "Desculpe, ocorreu um erro ao processar a resposta."


# ---------- Rotas ----------
class PerguntaRequest(BaseModel):
    pergunta: str


@app.on_event("startup")
def startup_event():
    index_documents()


@app.post("/perguntar")
def perguntar(req: PerguntaRequest):
    pergunta = req.pergunta.strip()
    # aceita pontuação depois do "debug" (a transcrição de voz costuma colocar ponto final)
    limpa = pergunta.rstrip(" .!?,;:")
    modo_debug = limpa.lower().endswith(PALAVRA_DEBUG)
    if modo_debug:
        pergunta = limpa[: -len(PALAVRA_DEBUG)].strip()

    chave_cache = (modo_debug, pergunta.lower())
    if chave_cache in cache_respostas:
        return {"resposta": cache_respostas[chave_cache], "cache": True}

    try:
        contexto = buscar_contexto(pergunta)
        resposta = perguntar_groq(pergunta, contexto, modo_debug)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERRO /perguntar] {e}")
        raise HTTPException(status_code=500, detail="Erro interno ao processar a pergunta.")

    cache_respostas[chave_cache] = resposta
    return {"resposta": resposta, "cache": False}


VOZ_TTS = "pt-BR-FranciscaNeural"


class TTSRequest(BaseModel):
    texto: str


@app.post("/tts")
async def tts(req: TTSRequest):
    """Gera o áudio da fala da Rita. Se falhar, o site continua só com texto."""
    texto = req.texto.replace("*", "").replace("#", "").replace("_", " ").strip()[:3000]
    if not texto:
        return Response(status_code=204)
    try:
        audio = b""
        async for parte in edge_tts.Communicate(texto, VOZ_TTS).stream():
            if parte["type"] == "audio":
                audio += parte["data"]
    except Exception as e:
        print(f"[ERRO TTS] {e}")
        return Response(status_code=503)
    return Response(content=audio, media_type="audio/mpeg")


STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
STT_MODEL = "whisper-large-v3-turbo"
STT_DICA = "Conversa sobre a história do Paraguai: Assunção, Francia, Yegros, Caballero, Belgrano, Chaco."


def _transcrever(audio: bytes, extensao: str) -> str:
    r = requests.post(
        STT_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": (f"fala.{extensao}", audio)},
        data={
            "model": STT_MODEL,
            "language": "pt",
            "response_format": "json",
            "temperature": 0,
            "prompt": STT_DICA,
        },
        timeout=60,
    )
    if not r.ok:
        print(f"[ERRO STT] Status {r.status_code}: {r.text}")
    r.raise_for_status()
    return r.json().get("text", "")


@app.post("/stt")
async def stt(request: Request):
    """Recebe o áudio gravado no navegador e devolve o texto falado."""
    audio = await request.body()
    if len(audio) < 1000:
        return {"texto": ""}
    tipo = request.headers.get("content-type", "")
    extensao = "ogg" if "ogg" in tipo else "mp4" if "mp4" in tipo else "webm"
    try:
        texto = await run_in_threadpool(_transcrever, audio, extensao)
    except Exception as e:
        print(f"[ERRO STT] {e}")
        return {"texto": ""}
    return {"texto": texto.strip()}


# ---------- Rita ao vivo com Gemini Live ----------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
LIVE_MODEL = "gemini-3.8-live"
LIVE_TOKEN_URL = "https://generativelanguage.googleapis.com/v1beta/auth_tokens"

PROMPT_LIVE = (
    PROMPT_BASE
    + " Você está conversando por voz. Responda sempre em português do Brasil, "
    "em frases curtas e naturais, sem listas, sem símbolos e sem formatação. "
    "Dê respostas de até três ou quatro frases, a menos que peçam mais detalhes. "
    "Você pode e deve usar a função 'mostrar_imagens' para exibir até 4 imagens históricas relevantes de docs/assets (como retratos de Pedro Juan Caballero, Fulgencio Yegros, Dr. Francia, Bernardo de Velasco ou a Casa da Independência) quando for oportuno para ilustrar a história."
)

FERRAMENTAS_LIVE = [
    {
        "functionDeclarations": [
            {
                "name": "mostrar_imagens",
                "description": "Mostra até 4 imagens históricas relevantes na tela para o usuário (retratos de líderes, Casa da Independência). Use quando for pertinente à conversa.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "imagens": {
                            "type": "ARRAY",
                            "items": {
                                "type": "STRING",
                                "enum": [
                                    "Bernardo_de_Velasco_retrato.jpg",
                                    "Bernardo_de_Velasco_corpo_inteiro.jpg",
                                    "Pedro-juan-caballero_retrato.jpg",
                                    "Pedro-juan-caballero_corpo_inteiro.jpg",
                                    "Fulgencio_Yegros_retrato.jpg",
                                    "José_Gaspar_de_Francia_sentado_em_uma_cadeira.jpg",
                                    "José_Gaspar_de_Francia_assinando_papeis_como_ditador.jpg",
                                    "Casa_de_la_independencia_hoje_em_dia.jpg"
                                ]
                            },
                            "description": "Lista com 1 a 4 nomes de arquivos de imagem de docs/assets."
                        }
                    },
                    "required": ["imagens"]
                }
            }
        ]
    }
]


def _texto_completo_docs() -> str:
    """Junta os .txt de docs/ inteiros (o Gemini Live aguenta bem o texto todo)."""
    partes = []
    for caminho in sorted(glob.glob(f"{DOCS_FOLDER}/*.txt")):
        with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
            partes.append(f.read())
    return "\n\n".join(partes)[:60000]


@app.post("/live-token")
def live_token():
    """Cria um token temporário (1 uso) para o navegador falar direto com o Gemini Live.
    A chave real e as instruções da Rita ficam só aqui no servidor."""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY não configurada")

    instrucao = {"parts": [{"text": PROMPT_LIVE + "\n\nINFORMAÇÕES:\n" + _texto_completo_docs()}]}
    agora = datetime.now(timezone.utc)
    formato = lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    base = {
        "uses": 1,
        "expireTime": formato(agora + timedelta(minutes=30)),
        "newSessionExpireTime": formato(agora + timedelta(minutes=2)),
    }
    modelo = f"models/{LIVE_MODEL}"

    # Duas formas de descrever a mesma configuração (a documentação do Google mostra as duas);
    # tenta a primeira e, se o Google recusar, tenta a segunda.
    tentativas = [
        {**base, "bidiGenerateContentSetup": {
            "model": modelo,
            "generationConfig": {"responseModalities": ["AUDIO"]},
            "systemInstruction": instrucao,
            "tools": FERRAMENTAS_LIVE,
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
        }},
        {**base, "liveConnectConstraints": {
            "model": modelo,
            "config": {
                "responseModalities": ["AUDIO"],
                "systemInstruction": instrucao,
                "tools": FERRAMENTAS_LIVE,
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
            },
        }},
    ]
    for corpo in tentativas:
        try:
            r = requests.post(
                LIVE_TOKEN_URL,
                headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
                json=corpo,
                timeout=30,
            )
        except requests.RequestException as e:
            print(f"[ERRO live-token] {e}")
            continue
        if r.ok:
            return {"token": r.json()["name"], "model": LIVE_MODEL}
        print(f"[ERRO live-token] Status {r.status_code}: {r.text}")
    raise HTTPException(status_code=502, detail="Não foi possível criar o token do Gemini Live")


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexados": len(chunks_texto)}


app.mount("/assets", StaticFiles(directory="docs/assets"), name="assets")
app.mount("/", StaticFiles(directory="static", html=True), name="static")
