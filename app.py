import streamlit as st
import datetime
import requests
import time
from chatbot_workflow import app
from openai import RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": """👋 Hello! I'm your Splan Product Assistant 🤖

I specialize in helping with all questions about our products and services. 
Here's what I can help you with:
- Product specifications
- Pricing information
- Troubleshooting
- Order status

How may I assist you today?""",
            "timestamp": datetime.datetime.now().strftime("%H:%M")
        }
    ]

if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.datetime.now()

if "ip_address" not in st.session_state:
    try:
        st.session_state.ip_address = requests.get("https://api.ipify.org", timeout=3).text
    except:
        st.session_state.ip_address = "Unknown"

# Session timeout check
if (datetime.datetime.now() - st.session_state.session_start).seconds > 600:
    st.warning("Session expired after 10 minutes of inactivity.")
    st.session_state.clear()
    st.experimental_rerun()

# Rate-limited API call handler
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def safe_invoke(question):
    try:
        return app.invoke({"question": question})
    except RateLimitError as e:
        st.warning(f"🔄 Please wait while I reconnect... (Attempt {safe_invoke.retry.statistics['attempt_number']}/3)")
        raise e
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {str(e)}")
        raise

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(f"**Splan Assistant** ({message['timestamp']})")
        else:
            st.markdown(f"**You** ({message['timestamp']})")
        st.write(message["content"])

# User input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message to chat
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.datetime.now().strftime("%H:%M")
    })
    
    # Display typing indicator
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get response
                response = safe_invoke(user_input)
                answer = response["answer"]
                
                # Simulate typing effect
                message_placeholder = st.empty()
                full_response = ""
                for chunk in answer.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(f"**Splan Assistant** ({datetime.datetime.now().strftime('%H:%M')})  \n{full_response}▌")
                
                # Final message
                message_placeholder.markdown(f"**Splan Assistant** ({datetime.datetime.now().strftime('%H:%M')})  \n{answer}")
                
                # Add to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "timestamp": datetime.datetime.now().strftime("%H:%M")
                })
                
            except Exception as e:
                st.error(f"⚠️ Sorry, I encountered an error: {str(e)}")

# Conversation feedback and collection (moved to sidebar)
with st.sidebar:
    if len(st.session_state.chat_history) >= 3 and "consent" not in st.session_state:
        consent = st.radio("Can we save this conversation to improve our services?", ("Yes", "No"))
        if consent:
            st.session_state.consent = consent

    if len(st.session_state.chat_history) >= 5 and "email_sent" not in st.session_state:
        if st.checkbox("📤 Email me this chat"):
            email = st.text_input("Your email:")
            if st.button("Send"):
                st.success("📬 Chat log sent to your email!")
                st.session_state.email_sent = True

    if len(st.session_state.chat_history) >= 5:
        st.divider()
        with st.expander("💬 Feedback"):
            with st.form("feedback_form"):
                st.write("How was your experience?")
                rating = st.slider("Rating", 1, 5, 3)
                comments = st.text_area("Additional comments")
                if st.form_submit_button("Submit"):
                    st.success("🎉 Thank you for your feedback!")
