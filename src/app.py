"""Interface Streamlit de demonstration de l'assistant RAG."""

import streamlit as st

from src.generate import RagAssistant

st.set_page_config(page_title="Assistant RAG", page_icon="💬")
st.title("Assistant documentaire (RAG)")
st.caption("Pose une question sur la base de connaissances indexee.")


@st.cache_resource
def load_assistant() -> RagAssistant:
    return RagAssistant()


assistant = load_assistant()

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ecris ta question ici...")

for role, message in st.session_state.history:
    with st.chat_message(role):
        st.markdown(message)

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans la base documentaire..."):
            answer = assistant.answer(question)
        st.markdown(answer)
    st.session_state.history.append(("assistant", answer))
