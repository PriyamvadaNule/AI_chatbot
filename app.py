import streamlit as st
import wikipedia
from gtts import gTTS
import os

try:
    import numpy as np  # Needed by transformers
except ImportError:
    st.error("🚨 Numpy not available. Please check your environment.")
    st.stop()

try:
    from transformers import pipeline
except ImportError:
    st.error("🚨 Transformers not available. Check your requirements.txt")
    st.stop()

st.title("📚 WikiTalk AI Chatbot")
st.write("Ask me anything. I’ll search Wikipedia and speak the answer!")

query = st.text_input("🔍 Your Question:")

if query:
    try:
        # Wikipedia summary
        context = wikipedia.summary(query, sentences=5)
        st.markdown("**📖 Wikipedia Summary:**")
        st.write(context)

        # QA model
        qa_pipeline = pipeline("question-answering")
        result = qa_pipeline(question=query, context=context)
        answer = result["answer"]

        # Display answer
        st.markdown("### ✅ Answer:")
        st.success(answer)

        # Text to Speech
        tts = gTTS(text=answer, lang='en')
        tts.save("answer.mp3")
        with open("answer.mp3", "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/mp3")
        os.remove("answer.mp3")

    except wikipedia.exceptions.DisambiguationError:
        st.error("⚠️ Your query is too broad. Please ask something more specific.")
    except wikipedia.exceptions.PageError:
        st.error("❌ Couldn't find that topic. Try a different question.")
    except Exception as e:
        st.error(f"🚨 Unexpected error: {e}")
