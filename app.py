import streamlit as st
import wikipedia
from transformers import pipeline
from gtts import gTTS
import os
import numpy as np  # Required by transformers

# Title and description
st.title("📚 WikiTalk AI Chatbot")
st.write("Ask me anything. I’ll search Wikipedia and speak the answer!")

# Input from user
query = st.text_input("🔍 Your Question:")

if query:
    try:
        # Get summary from Wikipedia
        context = wikipedia.summary(query, sentences=5)
        st.markdown("**📖 Wikipedia Summary:**")
        st.write(context)

        # Load the question-answering pipeline
        qa_pipeline = pipeline("question-answering")

        # Get answer from model
        result = qa_pipeline(question=query, context=context)
        answer = result["answer"]

        # Display the answer
        st.markdown("### ✅ Answer:")
        st.success(answer)

        # Text-to-Speech
        tts = gTTS(text=answer, lang='en')
        tts.save("answer.mp3")

        # Play the audio
        with open("answer.mp3", "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/mp3")

        # Cleanup
        os.remove("answer.mp3")

    except wikipedia.exceptions.DisambiguationError:
        st.error("⚠️ Your query is too broad. Please ask something more specific.")
    except wikipedia.exceptions.PageError:
        st.error("❌ Couldn't find that topic. Try a different question.")
    except Exception as e:
        st.error(f"🚨 Unexpected error: {e}")
