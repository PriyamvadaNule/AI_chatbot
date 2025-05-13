import streamlit as st
from transformers import pipeline
import wikipedia

# Set the title of the app
st.set_page_config(page_title="AI Chatbot (Wikipedia QA)", layout="centered")
st.title("🤖 AI Chatbot Based on Wikipedia")
st.markdown("Ask me anything about a topic! I’ll fetch data from Wikipedia and answer using a QA model.")

# Load the question-answering pipeline
@st.cache_resource
def load_model():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa_pipeline = load_model()

# User input
topic = st.text_input("📚 Enter a topic:", placeholder="e.g. Python programming")
query = st.text_input("❓ Ask a question about the topic:", placeholder="e.g. What is Python used for?")

# When both topic and query are entered
if topic and query:
    try:
        # Get summary from Wikipedia
        context = wikipedia.summary(topic)
        # Truncate context to fit model's token limit
        context = " ".join(context.split()[:450])  # ~512 tokens max for distilBERT

        # Get the answer
        result = qa_pipeline(question=query, context=context)
        answer = result["answer"]

        st.markdown("### ✅ Answer:")
        st.success(answer)

    except wikipedia.exceptions.DisambiguationError as e:
        st.error("The topic you entered is too broad. Please be more specific.")
    except wikipedia.exceptions.PageError:
        st.error("The topic was not found on Wikipedia.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
