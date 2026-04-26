import streamlit as st
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import wikipedia

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="GAAL System", layout="wide")
st.title("GAAL: Ephemeral Curiosity System")

MEMORY_TTL = 300  # seconds (5 min)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -----------------------------
# INIT SESSION MEMORY
# -----------------------------
if "memory" not in st.session_state:
    st.session_state.memory = [
        {"q": "What is AI?", "a": "AI is the simulation of human intelligence."},
        {"q": "What is ML?", "a": "ML learns patterns from data."}
    ]
    st.session_state.timestamps = [time.time(), time.time()]

# -----------------------------
# AUTO CLEAR OLD MEMORY
# -----------------------------
def clean_memory():
    current_time = time.time()
    new_memory = []
    new_time = []

    for item, t in zip(st.session_state.memory, st.session_state.timestamps):
        if current_time - t < MEMORY_TTL:
            new_memory.append(item)
            new_time.append(t)

    st.session_state.memory = new_memory
    st.session_state.timestamps = new_time

clean_memory()

# -----------------------------
# EMBEDDINGS
# -----------------------------
def get_embeddings(data):
    questions = [d["q"] for d in data]
    return model.encode(questions), questions

# -----------------------------
# EXTERNAL FETCH (CURIOSITY)
# -----------------------------
def explore_external(query):
    try:
        results = wikipedia.search(query)
        summaries = []
        for r in results[:2]:
            try:
                summaries.append(wikipedia.summary(r, sentences=2))
            except:
                pass
        return summaries
    except:
        return []

# -----------------------------
# GAAL PROCESS
# -----------------------------
def gaal(query):

    kb = st.session_state.memory
    embeddings, questions = get_embeddings(kb)

    q_vec = model.encode([query])
    scores = cosine_similarity(q_vec, embeddings)[0]

    max_score = np.max(scores)
    gap = 1 - max_score

    # STATE
    if max_score > 0.7:
        state = "CONFIDENT"
    elif max_score > 0.4:
        state = "UNCERTAIN"
    else:
        state = "CURIOUS"

    actions = []

    # BEHAVIOR
    if state == "CONFIDENT":
        idx = np.argmax(scores)
        answer = kb[idx]["a"]
        actions.append("Used internal memory")

    elif state == "UNCERTAIN":
        idxs = scores.argsort()[-2:][::-1]
        answer = " ".join([kb[i]["a"] for i in idxs])
        actions.append("Combined partial knowledge")

    else:
        external = explore_external(query)

        if external:
            answer = " ".join(external)
            actions.append("Gap detected")
            actions.append("Explored external knowledge")

            # TEMP LEARNING (with timestamp)
            if not any(d["q"].lower() == query.lower() for d in kb):
                st.session_state.memory.append({"q": query, "a": answer})
                st.session_state.timestamps.append(time.time())

        else:
            answer = "I couldn't understand this yet."
            actions.append("Exploration failed")

    return state, answer, gap, actions, scores, questions

# -----------------------------
# UI INPUT
# -----------------------------
query = st.text_input(" Ask something")

# -----------------------------
# MAIN
# -----------------------------
if query:
    state, answer, gap, actions, scores, questions = gaal(query)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Response")
        st.write(answer)

        st.subheader(" State")
        if state == "CONFIDENT":
            st.success("CONFIDENT")
        elif state == "UNCERTAIN":
            st.warning("UNCERTAIN")
        else:
            st.error("CURIOUS")

    with col2:
        st.metric("Gap", f"{gap:.2f}")
        st.progress(float(gap))

    st.subheader("Actions")
    for a in actions:
        st.write(f" {a}")

    st.subheader(" Similarity")
    for i, q in enumerate(questions):
        st.write(f"{q} → {scores[i]:.3f}")

    st.subheader(" Active Memory (Auto-Expiring)")
    st.write(f"{len(st.session_state.memory)} items (clears in {MEMORY_TTL}s)")

else:
    st.info("Ask a question to activate GAAL")