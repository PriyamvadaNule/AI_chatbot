import streamlit as st
import wikipedia
from transformers import pipeline
from gtts import gTTS
import os

st.title("🌍 Web Search AI Chatbot")
query = st.text_input("❓ Ask a question")

if query:
    try:
        context = wikipedia.summary(query, sentences=5)
        st.markdown("**📚 Wikipedia Context:**")
        st.write(context)
    except:
        st.error("❌ Couldn't find Wikipedia info.")
        context = None

    if context:
        qa_pipeline = pipeline("question-answering")
        answer = qa_pipeline(question=query, context=context)["answer"]

        st.markdown("### ✅ Answer:")
        st.success(answer)

        tts = gTTS(text=answer, lang='en')
        tts.save("answer.mp3")
        st.audio("answer.mp3")
        os.remove("answer.mp3")
