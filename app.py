"""
ИИ-ассистент по документу (RAG) — веб-приложение (один файл).
Итоговая аттестация. Тема 3: Web-приложения — разработка, расширение
функционала и деплой. Автор: Азизов Игорь Алексеевич. Казань, 2026.

Как работает:
  1. Пользователь загружает PDF-документ (инструкцию, регламент, справочник).
  2. Приложение извлекает текст, делит его на фрагменты (чанки) и индексирует.
  3. По вопросу пользователя находятся релевантные фрагменты (retrieval).
  4. Языковая модель (Groq LLM) формирует ответ СТРОГО по найденным фрагментам.
  Если ответа в документе нет — ассистент честно об этом сообщает.

Весь код собран в одном файле намеренно: так Streamlit при каждом обновлении
гарантированно перечитывает свежую версию (отдельный модуль он кэширует в памяти).
"""
import re
import io

import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

# Метка версии — видно на странице, чтобы всегда точно знать, какой код запущен.
APP_VERSION = "v3 (softer prompt)"

# ==========================================================================
#  ЧАСТЬ 1. ЛОГИКА RAG
# ==========================================================================

# ---------- Извлечение текста из PDF ----------
def extract_text_from_pdf(file_obj) -> str:
    reader = PdfReader(file_obj)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


# ---------- Деление текста на фрагменты (чанки) ----------
HEADING_RE = re.compile(r"(?m)^\s*\d{1,2}\.\s+\S")  # заголовок пункта: "1. Название"
MAX_SECTION_LEN = 1200


def _split_by_sentences(text, chunk_size, overlap):
    """Запасной способ: деление по предложениям (для текста без нумерации)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= chunk_size:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if overlap and current else ""
            current = (tail + " " + sent).strip()
    if current:
        chunks.append(current)
    return chunks


def split_into_chunks(text, chunk_size=350, overlap=80):
    """
    Делит документ на смысловые фрагменты. Основной режим — по нумерованным
    пунктам (каждый пункт целиком). Если нумерации нет — по предложениям.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    if not text:
        return []

    starts = [m.start() for m in HEADING_RE.finditer(text)]
    if len(starts) < 2:
        return _split_by_sentences(text, chunk_size, overlap)

    chunks = []
    preamble = text[:starts[0]].strip()
    if preamble:
        chunks.append(preamble)

    borders = starts + [len(text)]
    for i in range(len(starts)):
        section = " ".join(text[borders[i]:borders[i + 1]].split())
        if not section:
            continue
        if len(section) <= MAX_SECTION_LEN:
            chunks.append(section)
        else:
            heading = section.split(".")[0] + "."
            for part in _split_by_sentences(section, chunk_size, overlap):
                chunks.append(part if part.startswith(heading) else f"{heading} {part}")
    return chunks


# ---------- Поиск релевантных фрагментов ----------
class Retriever:
    """
    Индексирует фрагменты через TF-IDF и ищет самые похожие на вопрос.
    Два взгляда на текст: по словам (точные термины) и по символам
    (устойчивость к русским словоформам: «функцию» находит «функция»).
    """
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = FeatureUnion([
            ("words", TfidfVectorizer(
                lowercase=True, analyzer="word",
                ngram_range=(1, 2), token_pattern=r"(?u)\b\w\w+\b")),
            ("chars", TfidfVectorizer(
                lowercase=True, analyzer="char_wb", ngram_range=(3, 5))),
        ])
        self.matrix = self.vectorizer.fit_transform(chunks)

    def search(self, query, top_k=3):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).ravel()
        top_idx = sims.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_idx]


# ---------- Генерация ответа через Groq ----------
SYSTEM_PROMPT = (
    "Ты — ассистент, который отвечает на вопросы по документу.\n"
    "Правила:\n"
    "1. Отвечай на основе КОНТЕКСТА: используй определения, примеры кода, "
    "формулировки и данные, приведённые в контексте. Собственных знаний, "
    "которых нет в контексте, добавлять не нужно.\n"
    "2. Если пользователь просит показать пример или что-то написать, а в "
    "контексте есть подходящий пример кода — приведи его полностью, "
    "как он записан в документе, оформив блоком кода. Не сокращай и не "
    "переписывай код по-своему.\n"
    "3. Если тема вопроса вообще не встречается в контексте — ответь одной "
    "фразой: «В документе нет информации по этому вопросу». Правило "
    "применяется только к вопросам совсем не по теме документа, не к тем, "
    "где ответ хотя бы частично есть.\n"
    "4. Не выдумывай факты, цифры, сроки и названия, которых нет в контексте.\n"
    "5. Отвечай кратко, по-русски, деловым языком."
)

PRIMARY_MODEL = "qwen/qwen3.6-27b"
FALLBACK_MODEL = "openai/gpt-oss-20b"
MAX_ANSWER_TOKENS = 800


def build_user_prompt(context, question):
    return (f"КОНТЕКСТ (фрагменты документа):\n{context}\n\n"
            f"ВОПРОС: {question}\n\nОтветь только по контексту выше.")


def clean_answer(raw):
    """Убирает блок размышлений <think>...</think> на случай, если он появится."""
    text = (raw or "").strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    elif "<think>" in text:
        text = text.split("<think>", 1)[0].strip()
    return text


def generate_answer(groq_client, question, retrieved):
    """Две попытки: основная модель с выключенными размышлениями, затем запасная."""
    context = "\n\n---\n\n".join(chunk for chunk, score in retrieved)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(context, question)},
    ]

    # Попытка 1: Qwen с полностью выключенным режимом размышлений
    try:
        resp = groq_client.chat.completions.create(
            messages=messages, model=PRIMARY_MODEL,
            temperature=0.1, max_tokens=MAX_ANSWER_TOKENS,
            reasoning_effort="none",
        )
        answer = clean_answer(resp.choices[0].message.content)
        if answer:
            return answer
    except Exception:
        pass

    # Попытка 2: запасная модель (не «размышляет» по умолчанию)
    resp = groq_client.chat.completions.create(
        messages=messages, model=FALLBACK_MODEL,
        temperature=0.1, max_tokens=MAX_ANSWER_TOKENS,
    )
    answer = clean_answer(resp.choices[0].message.content)
    return answer or "Не удалось получить ответ от модели. Попробуйте ещё раз."


# Порог релевантности: если ни один фрагмент не набрал столько — вопрос явно
# не по документу. Подобран по замерам: реальные вопросы от 0.136, посторонние
# до 0.184; порог занижен, чтобы не блокировать осмысленные вопросы.
RELEVANCE_THRESHOLD = 0.08


# ==========================================================================
#  ЧАСТЬ 2. ИНТЕРФЕЙС
# ==========================================================================
st.set_page_config(page_title="ИИ-ассистент по документу (RAG)", page_icon="🤖", layout="centered")

st.title("🤖 ИИ-ассистент по документу")
st.caption(f"RAG-агент: отвечает на вопросы строго по загруженному документу. "
           f"Итоговая аттестация · Азизов И. А. · Казань, 2026 · {APP_VERSION}")


def get_groq_client():
    api_key = None
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None
    if not api_key:
        st.error("Не найден GROQ_API_KEY. Добавьте его в Settings → Secrets: "
                 'GROQ_API_KEY = "ваш_ключ"')
        st.stop()
    return Groq(api_key=api_key)


with st.sidebar:
    st.header("Как пользоваться")
    st.markdown(
        "1. Загрузите PDF-документ.\n"
        "2. Дождитесь сообщения «Документ обработан».\n"
        "3. Задайте вопрос по содержанию документа.\n\n"
        "Ассистент отвечает **только по документу** и честно говорит, "
        "если информации в нём нет."
    )
    st.divider()
    top_k = st.slider("Сколько фрагментов искать (top-k)", 1, 6, 3)
    st.caption(f"Модель: Qwen 3.6-27B через Groq API · {APP_VERSION}")


@st.cache_data(show_spinner=False)
def process_document(file_bytes):
    text = extract_text_from_pdf(io.BytesIO(file_bytes))
    return text, split_into_chunks(text)


uploaded = st.file_uploader("📄 Загрузите PDF-документ", type=["pdf"])

if uploaded is not None:
    with st.spinner("Обрабатываю документ…"):
        text, chunks = process_document(uploaded.getvalue())

    if not chunks:
        st.error("Не удалось извлечь текст из PDF. Возможно, это скан без текстового слоя.")
        st.stop()

    retriever = Retriever(chunks)
    st.success(f"Документ обработан ✅  Символов: {len(text)}, фрагментов: {len(chunks)}")

    question = st.text_input("❓ Ваш вопрос по документу:",
                             placeholder="Например: как сбросить пароль?")

    if st.button("Спросить", type="primary") and question.strip():
        client = get_groq_client()
        with st.spinner("Ищу ответ в документе…"):
            retrieved = retriever.search(question, top_k=top_k)
            best_score = retrieved[0][1] if retrieved else 0.0
            if best_score < RELEVANCE_THRESHOLD:
                answer = ("В документе нет информации по этому вопросу. "
                          "Задайте вопрос по содержанию загруженного документа.")
            else:
                answer = generate_answer(client, question, retrieved)

        st.markdown("### 💬 Ответ")
        st.write(answer)

        with st.expander("🔎 На основе каких фрагментов документа (источники)"):
            for i, (chunk, score) in enumerate(retrieved, 1):
                st.markdown(f"**Фрагмент {i}** (релевантность: {score:.3f})")
                st.info(chunk)
else:
    st.info("👆 Загрузите PDF-документ, чтобы начать. "
            "Например: инструкцию, регламент техподдержки или справочник.")
