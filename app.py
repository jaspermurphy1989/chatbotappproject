import streamlit as st
import datetime
import time
import requests
from chatbot_workflow import app
from openai import OpenAI, RateLimitError

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.datetime.now()

if (datetime.datetime.now() - st.session_state.session_start).seconds > 600:
    st.warning("Session expired after 10 minutes of inactivity.")
    st.session_state.clear()
    st.experimental_rerun()

if "ip_address" not in st.session_state:
    try:
        st.session_state.ip_address = requests.get("https://api.ipify.org").text
    except:
        st.session_state.ip_address = "Unknown"

with st.chat_message("assistant"):
    st.write("👋 Hello! I'm your Splan Product Assistant 🤖")
    st.write("I'm here to help you with any questions about our products and services.")
    st.write("How may I assist you today?")

user_input = st.text_input("Ask our Expert about our Splan Products:")

if st.button("Submit"):
    if user_input:
        st.info("Generating your answer, please wait...")
        time.sleep(3)
        response = app.invoke({"question": user_input})
        answer = response["answer"]
        st.write("Answer:")
        st.write(answer)
        st.session_state.chat_history.append({"user": user_input, "bot": answer})
    else:
        st.warning("Please enter a question.")

if len(st.session_state.chat_history) >= 3 and "consent" not in st.session_state:
    consent = st.radio("Can we save this conversation to improve our services?", ("Yes", "No"))
    if consent:
        st.session_state.consent = consent

if len(st.session_state.chat_history) >= 5 and "email_sent" not in st.session_state:
    if st.checkbox("📤 Would you like to receive this chat by email?"):
        email = st.text_input("Enter your email to receive chat log:")
        if st.button("Send Email"):
            st.success("📬 Chat log will be sent to your email (simulation).")
            st.session_state.email_sent = True

if len(st.session_state.chat_history) >= 5 and st.button("✅ Yes, I'm satisfied"):
    with st.form("user_details"):
        first = st.text_input("First Name")
        last = st.text_input("Last Name")
        phone = st.text_input("Phone")
        email = st.text_input("Email")
        submit = st.form_submit_button("Submit")
        if submit:
            st.success("🎉 Thanks! Your feedback has been recorded.")
