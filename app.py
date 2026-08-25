"""
ИИ-ассистент по документу (RAG) — веб-приложение.
Итоговая аттестация. Тема 4: ИИ-агенты и no-code автоматизация.
Автор: Азизов Игорь Алексеевич. Казань, 2026.

Как работает:
  1. Пользователь загружает PDF-документ (инструкцию, регламент).
  2. Приложение извлекает текст, делит его на фрагменты (чанки) и индексирует (TF-IDF).
  3. По вопросу пользователя находятся самые релевантные фрагменты (retrieval).
  4. Языковая модель (Groq LLM) формирует ответ СТРОГО по найденным фрагментам (generation).
  Если ответа в документе нет — ассистент честно об этом сообщает.
"""
import streamlit as st
from groq import Groq

from rag_core import (
    extract_text_from_pdf,
    split_into_chunks,
    Retriever,
    generate_answer,
)

# Минимальная релевантность фрагмента, при которой имеет смысл обращаться
# к языковой модели. Если ни один фрагмент не набрал столько — вопрос явно
# не по документу, и ассистент говорит об этом сразу, не тратя запрос.
#
# Значение подобрано по замерам на тестовых документах: настоящие вопросы
# набирали от 0.136, посторонние — до 0.184. Диапазоны частично перекрываются,
# поэтому порог намеренно занижен: он служит страховкой от совсем чужих
# вопросов и НИКОГДА не блокирует осмысленный. Основную защиту от выдумывания
# несёт системный промпт, который требует отвечать только по контексту.
RELEVANCE_THRESHOLD = 0.08

# ---------- Настройка страницы ----------
st.set_page_config(page_title="ИИ-ассистент по документу (RAG)", page_icon="🤖", layout="centered")

st.title("🤖 ИИ-ассистент по документу")
st.caption("RAG-агент: отвечает на вопросы строго по загруженному документу. "
           "Итоговая аттестация · Азизов И. А. · Казань, 2026")

# ---------- Получение API-ключа Groq ----------
# Ключ хранится в секретах Streamlit (Settings → Secrets), а НЕ в коде.
def get_groq_client():
    api_key = None
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None
    if not api_key:
        st.error("Не найден GROQ_API_KEY. Добавьте его в Settings → Secrets вашего приложения "
                 "в формате:  GROQ_API_KEY = \"ваш_ключ\"")
        st.stop()
    return Groq(api_key=api_key)

# ---------- Боковая панель: как пользоваться ----------
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
    st.caption("Модель: Qwen 3.6-27B через Groq API")

# ---------- Загрузка документа ----------
uploaded = st.file_uploader("📄 Загрузите PDF-документ", type=["pdf"])

# Кэшируем обработку документа, чтобы не пересчитывать при каждом вопросе.
@st.cache_data(show_spinner=False)
def process_document(file_bytes):
    import io
    text = extract_text_from_pdf(io.BytesIO(file_bytes))
    chunks = split_into_chunks(text)
    return text, chunks

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    with st.spinner("Обрабатываю документ…"):
        text, chunks = process_document(file_bytes)

    if not chunks:
        st.error("Не удалось извлечь текст из PDF. Возможно, это скан (картинка) без текстового слоя.")
        st.stop()

    retriever = Retriever(chunks)
    st.success(f"Документ обработан ✅  Символов: {len(text)}, фрагментов: {len(chunks)}")

    # ---------- Вопрос пользователя ----------
    question = st.text_input("❓ Ваш вопрос по документу:",
                             placeholder="Например: как сбросить пароль?")

    if st.button("Спросить", type="primary") and question.strip():
        client = get_groq_client()
        with st.spinner("Ищу ответ в документе…"):
            retrieved = retriever.search(question, top_k=top_k)

            # Защита от галлюцинаций: если ни один фрагмент не релевантен
            # вопросу, не обращаемся к модели вообще — сразу честный ответ.
            best_score = retrieved[0][1] if retrieved else 0.0
            if best_score < RELEVANCE_THRESHOLD:
                answer = ("В документе нет информации по этому вопросу. "
                          "Задайте вопрос по содержанию загруженного документа.")
            else:
                answer = generate_answer(client, question, retrieved)

        st.markdown("### 💬 Ответ")
        st.write(answer)

        # Показываем, на каких фрагментах основан ответ (прозрачность RAG)
        with st.expander("🔎 На основе каких фрагментов документа (источники)"):
            for i, (chunk, score) in enumerate(retrieved, 1):
                st.markdown(f"**Фрагмент {i}** (релевантность: {score:.3f})")
                st.info(chunk)
else:
    st.info("👆 Загрузите PDF-документ, чтобы начать. "
            "Например: инструкцию, регламент техподдержки или методичку.")
