"""
Ядро RAG-ассистента: извлечение текста из PDF, деление на чанки,
поиск релевантных фрагментов (TF-IDF) и генерация ответа через Groq LLM.

Логика вынесена отдельно от интерфейса (app.py), чтобы её можно было
тестировать и переиспользовать.
"""
import re
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion


# ---------- 1. Извлечение текста из PDF ----------
def extract_text_from_pdf(file_obj) -> str:
    """Читает PDF и возвращает весь текст одной строкой."""
    reader = PdfReader(file_obj)
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n".join(parts)


# ---------- 2. Деление текста на чанки ----------
# Заголовок нумерованного пункта: строка вида "1. Доступ в PDM"
HEADING_RE = re.compile(r"(?m)^\s*\d{1,2}\.\s+\S")

# Если один пункт длиннее этого значения — он дополнительно дробится
MAX_SECTION_LEN = 1200


def _split_by_sentences(text: str, chunk_size: int, overlap: int) -> list:
    """
    Запасной способ: деление по границам предложений.
    Используется для документов без нумерованных пунктов.
    """
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

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


def split_into_chunks(text: str, chunk_size: int = 350, overlap: int = 80) -> list:
    """
    Делит документ на смысловые фрагменты (чанки).

    Основной режим — деление ПО ПУНКТАМ документа: если найдены нумерованные
    заголовки ("1. ...", "2. ..."), каждый пункт становится отдельным цельным
    фрагментом. Это важно для регламентов и инструкций: условия одного пункта
    (сроки, приоритет, ответственный) не смешиваются с условиями другого.

    Если нумерованных пунктов нет (произвольный текст) — используется
    запасной режим: деление по предложениям с перекрытием.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    if not text:
        return []

    # Позиции, с которых начинаются нумерованные пункты
    starts = [m.start() for m in HEADING_RE.finditer(text)]

    # Меньше двух пунктов — структуры нет, работаем по предложениям
    if len(starts) < 2:
        return _split_by_sentences(text, chunk_size, overlap)

    chunks = []

    # Текст до первого пункта (заголовок документа, преамбула)
    preamble = text[:starts[0]].strip()
    if preamble:
        chunks.append(preamble)

    # Каждый пункт — от своего заголовка до начала следующего
    borders = starts + [len(text)]
    for i in range(len(starts)):
        section = text[borders[i]:borders[i + 1]].strip()
        section = " ".join(section.split())  # схлопываем переносы строк внутри пункта
        if not section:
            continue

        if len(section) <= MAX_SECTION_LEN:
            chunks.append(section)
        else:
            # Слишком длинный пункт дробим, но к каждой части добавляем
            # его заголовок — иначе часть потеряет контекст
            heading = section.split(".")[0] + "."
            for part in _split_by_sentences(section, chunk_size, overlap):
                chunks.append(part if part.startswith(heading) else f"{heading} {part}")

    return chunks


# ---------- 3. Поиск релевантных чанков (retrieval) ----------
class Retriever:
    """
    Индексирует чанки через TF-IDF и находит наиболее похожие
    на вопрос пользователя (косинусная близость).
    TF-IDF выбран сознательно: он лёгкий, быстрый, не требует GPU
    и тяжёлых моделей — приложение мгновенно разворачивается в облаке.
    """
    def __init__(self, chunks: list):
        self.chunks = chunks

        # Два взгляда на текст одновременно:
        #  • по словам   — точные совпадения терминов (PDM, VPN, DataFrame);
        #  • по символам — устойчивость к русской морфологии: «функцию» находит
        #    «функция», хотя это разные словоформы. Без этого поиск промахивался.
        # Сравнение вариантов на тестовых документах: только слова — 15/18 точных
        # попаданий, комбинация — 17/18.
        self.vectorizer = FeatureUnion([
            ("words", TfidfVectorizer(
                lowercase=True, analyzer="word",
                ngram_range=(1, 2), token_pattern=r"(?u)\b\w\w+\b",
            )),
            ("chars", TfidfVectorizer(
                lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
            )),
        ])
        self.matrix = self.vectorizer.fit_transform(chunks)

    def search(self, query: str, top_k: int = 3):
        """Возвращает список (чанк, оценка_похожести) для top_k лучших."""
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).ravel()
        # индексы top_k по убыванию похожести
        top_idx = sims.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_idx]


# ---------- 4. Генерация ответа через Groq LLM ----------
SYSTEM_PROMPT = (
    "Ты — ассистент, который отвечает на вопросы СТРОГО по приведённому "
    "КОНТЕКСТУ из документа.\n"
    "Правила:\n"
    "1. Используй только сведения из контекста. Никаких собственных знаний.\n"
    "2. Если в контексте нет прямого ответа на заданный вопрос — ответь ровно "
    "одной фразой: «В документе нет информации по этому вопросу». "
    "Не пытайся угадать, не пересказывай похожие места документа и не отвечай "
    "частично.\n"
    "3. Не выдумывай факты, цифры, сроки и названия.\n"
    "4. Если в контексте есть пример кода, подходящий к вопросу — приведи его "
    "полностью, как он записан в документе, оформив блоком кода. Не сокращай "
    "код и не переписывай его по-своему.\n"
    "5. Отвечай кратко, по-русски, деловым языком."
)


def build_user_prompt(context: str, question: str) -> str:
    return (
        f"КОНТЕКСТ (фрагменты документа):\n{context}\n\n"
        f"ВОПРОС: {question}\n\n"
        "Ответь только по контексту выше."
    )


# Qwen — «размышляющая» модель: по умолчанию она сначала пишет длинный ход
# рассуждений в теге <think> и лишь потом отвечает. На практике рассуждения
# съедали весь лимит токенов, и до ответа модель не доходила.
# Параметр reasoning_effort="none" полностью выключает этот режим —
# проверено экспериментально на реальном API.
PRIMARY_MODEL = "qwen/qwen3.6-27b"     # основная модель
FALLBACK_MODEL = "openai/gpt-oss-20b"  # запасная, если основная недоступна
MAX_ANSWER_TOKENS = 800                # без размышлений этого с запасом хватает


def clean_answer(raw: str) -> str:
    """
    Убирает блок рассуждений <think>...</think>, который добавляют
    «размышляющие» модели (Qwen и подобные).

    Разбираются три случая:
      1) блок закрыт  — берём всё, что идёт после последнего </think>;
      2) блок не закрыт (ответ обрезали) — отбрасываем всё от <think> до конца;
      3) блока нет     — возвращаем текст как есть.
    """
    text = (raw or "").strip()

    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    elif "<think>" in text:
        text = text.split("<think>", 1)[0].strip()

    return text


def generate_answer(groq_client, question: str, retrieved: list,
                    model: str = PRIMARY_MODEL) -> str:
    """
    Собирает контекст из найденных фрагментов и просит модель ответить по нему.

    Работает в две попытки:
      1) основная модель с выключенным режимом размышлений;
      2) если она недоступна или вернула пустоту — запасная модель.
    """
    context = "\n\n---\n\n".join(chunk for chunk, score in retrieved)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(context, question)},
    ]

    # Попытка 1: основная модель, режим размышлений выключен
    try:
        response = groq_client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.1,
            max_tokens=MAX_ANSWER_TOKENS,
            reasoning_effort="none",
        )
        answer = clean_answer(response.choices[0].message.content)
        if answer:
            return answer
    except Exception:
        pass  # переходим к запасному варианту

    # Попытка 2: запасная модель (она не «размышляет» по умолчанию)
    response = groq_client.chat.completions.create(
        messages=messages,
        model=FALLBACK_MODEL,
        temperature=0.1,
        max_tokens=MAX_ANSWER_TOKENS,
    )
    answer = clean_answer(response.choices[0].message.content)

    return answer or ("Не удалось получить ответ от модели. "
                      "Попробуйте задать вопрос ещё раз.")
