import re

import faiss
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()  # reads ANTHROPIC_API_KEY from .env
MODEL = "claude-sonnet-4-6"  # change here if you want another model
client = Anthropic()

st.set_page_config(page_title="AI Placement Coach", page_icon="🎯", layout="wide")


# ---------- helpers ----------
def read_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, size: int = 200, overlap: int = 40):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)] or [""]


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def build_index(chunks):
    vectors = load_embedder().encode(chunks, normalize_embeddings=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def ask_llm(prompt: str, system: str = "You are a helpful placement coach.") -> str:
    reply = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return reply.content[0].text


# ---------- tools (the 3 functions) ----------
def search_resume(query: str, k: int = 3) -> str:
    q = load_embedder().encode([query], normalize_embeddings=True)
    _, ids = st.session_state.index.search(q, k)
    return "\n---\n".join(st.session_state.chunks[i] for i in ids[0] if i >= 0)


def generate_questions(topic: str, n: int = 5) -> list:
    text = ask_llm(
        f"Resume:\n{st.session_state.resume[:4000]}\n\n"
        f"Job description:\n{st.session_state.jd[:3000]}\n\n"
        f"Write {n} interview questions about: {topic}. "
        "Return one question per line, no numbering, no extra text."
    )
    return [q.strip("-• ").strip() for q in text.splitlines() if q.strip()][:n]


def evaluate_answer(question: str, answer: str):
    text = ask_llm(
        f"Interview question: {question}\nCandidate answer: {answer}\n\n"
        "Reply in this exact format:\nSCORE: <number 0-10>\n"
        "FEEDBACK: <2-3 short sentences>\nBETTER ANSWER: <3 sentences>"
    )
    match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", text)
    score = min(float(match.group(1)), 10) if match else 0
    return score, text


# ---------- state ----------
for key, default in {
    "resume": "", "jd": "", "chunks": [], "index": None,
    "questions": [], "history": [],
}.items():
    st.session_state.setdefault(key, default)

# ---------- sidebar ----------
with st.sidebar:
    st.header("1. Upload")
    resume_file = st.file_uploader("Your resume (PDF)", type="pdf")
    jd_text = st.text_area("Paste the job description", height=200)
    if st.button("Process", type="primary"):
        if resume_file is None:
            st.error("Upload your resume first.")
        else:
            with st.spinner("Reading and indexing..."):
                st.session_state.resume = read_pdf(resume_file)
                st.session_state.jd = jd_text
                st.session_state.chunks = chunk_text(st.session_state.resume)
                st.session_state.index = build_index(st.session_state.chunks)
                st.session_state.questions = []
            st.success("Ready!")

# ---------- main ----------
st.title("🎯 AI Placement Coach")

if st.session_state.index is None:
    st.info("Upload your resume and job description in the sidebar, then click Process.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["💬 Ask my resume", "🎤 Mock interview", "📊 Job match"])

with tab1:
    question = st.text_input("Ask anything about your resume", placeholder="What projects do I have?")
    if question:
        with st.spinner("Thinking..."):
            context = search_resume(question)
            answer = ask_llm(f"Resume excerpts:\n{context}\n\nQuestion: {question}\n"
                             "Answer only from the excerpts.")
        st.write(answer)
        with st.expander("Resume parts used"):
            st.write(context)

with tab2:
    topic = st.text_input("Topic", value="my projects and technical skills")
    if st.button("Generate questions"):
        with st.spinner("Writing questions..."):
            st.session_state.questions = generate_questions(topic)
    if st.session_state.questions:
        q = st.selectbox("Pick a question", st.session_state.questions)
        ans = st.text_area("Your answer", height=150)
        if st.button("Score my answer") and ans.strip():
            with st.spinner("Scoring..."):
                score, feedback = evaluate_answer(q, ans)
            st.session_state.history.append(score)
            st.metric("Score", f"{score}/10")
            st.write(feedback)
    if st.session_state.history:
        st.subheader("Score history")
        st.line_chart(st.session_state.history)

with tab3:
    if not st.session_state.jd.strip():
        st.warning("Paste a job description in the sidebar and click Process again.")
    elif st.button("Check match"):
        with st.spinner("Comparing..."):
            result = ask_llm(
                f"Resume:\n{st.session_state.resume[:4000]}\n\n"
                f"Job description:\n{st.session_state.jd[:3000]}\n\n"
                "Give: 1) match score out of 100, 2) missing skills, "
                "3) three suggestions to improve the resume."
            )
        st.write(result)
