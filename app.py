import streamlit as st
import wikipedia
from transformers import pipeline
from gtts import gTTS
import os

st.title("📚 WikiTalk AI Chatbot")
st.write("Ask me anything. I’ll search Wikipedia and speak the answer!")

query = st.text_input("🔍 Your Question:")

if query:
    try:
        context = wikipedia.summary(query, sentences=5)
        st.markdown("**Wikipedia Summary:**")
        st.write(context)
    except:
        st.error("❌ Couldn't find that topic. Try again.")
        context = None

    if context:
        qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        answer = qa_pipeline(question=query, context=context)["answer"]

        st.markdown("### ✅ Answer:")
        st.success(answer)

        tts = gTTS(text=answer, lang='en')
        tts.save("answer.mp3")
        audio_file = open("answer.mp3", "rb")
        st.audio(audio_file.read(), format="audio/mp3")
        audio_file.close()
        os.remove("answer.mp3")
