import streamlit as st  # import Streamlit for the web UI

# Setup configuration for the app
st.set_page_config(page_title="JomeInvoice: SME E-Invoicing Guideline Assistant", page_icon="🤖", layout="wide")

# --- DEFINISIKAN PAGES (Only chatbot remains) ---
chatbot_page = st.Page("views/chatbot.py", title="Guideline Chatbot", icon="💬", default=True)

# --- SETUP NAVIGATION BAR (SIDEBAR) ---
pg = st.navigation([chatbot_page])
pg.run()