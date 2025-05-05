import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from streamlit_cookies_manager import EncryptedCookieManager
from PyPDF2 import PdfReader
import datetime
import uuid
import time

# === Setup MongoDB ===
client = MongoClient("mongodb+srv://streetnoshery:Sumit%40Godwan%401062@streetnoshery.g7ufm.mongodb.net/")
db = client["street_noshery"]
users_col = db["chat_users"]
pdfs_col = db["chat_pdfs"]

# === Cookie Manager ===
cookies = EncryptedCookieManager(
    prefix="chatbot/",
    password="your-secure-password"  # Use env vars in production
)

if not cookies.ready():
    st.stop()

# === Session from Cookies ===
session_token = cookies.get("session_token")
user = users_col.find_one({"session_token": session_token}) if session_token else None
if user:
    st.session_state.logged_in = True
    st.session_state.user = user

# === Auth Logic ===
def register_user(username, password):
    if users_col.find_one({"username": username}):
        return False, "Username already exists."
    hashed_pw = generate_password_hash(password)
    session_token = str(uuid.uuid4())
    users_col.insert_one({
        "username": username,
        "password": hashed_pw,
        "session_token": session_token,
        "registered_at": datetime.datetime.utcnow()
    })
    cookies["session_token"] = session_token
    cookies.save()
    return True, users_col.find_one({"username": username})

def authenticate_user(username, password):
    user = users_col.find_one({"username": username})
    if not user:
        return False, None
    if check_password_hash(user["password"], password):
        session_token = str(uuid.uuid4())
        users_col.update_one({"_id": user["_id"]}, {"$set": {"session_token": session_token}})
        cookies["session_token"] = session_token
        cookies.save()
        user["session_token"] = session_token
        return True, user
    return False, None

# === Sidebar Auth UI ===
if not st.session_state.get("logged_in", False):
    st.sidebar.title("Authentication")
    auth_mode = st.sidebar.radio("Choose", ["Login", "Register"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Submit"):
        if auth_mode == "Register":
            success, result = register_user(username, password)
        else:
            success, result = authenticate_user(username, password)

        if success:
            st.session_state.logged_in = True
            st.session_state.user = result
            st.sidebar.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.sidebar.error(result or "Invalid credentials")

# === Main App ===
if st.session_state.get("logged_in", False):
    user = st.session_state["user"]
    user_id = str(user["_id"])
    st.title(f"Welcome {user['username']} 👋")

    st.sidebar.title("Your Documents")
    pdfs = list(pdfs_col.find({"user_id": user_id}, {"filename": 1, "_id": 0}).sort("uploaded_at", -1))
    filenames = [p["filename"] for p in pdfs]
    selected_filename = st.sidebar.selectbox("Choose a PDF", ["-- Select --"] + filenames)
    file = None
    if selected_filename == "-- Select --" or not filenames:
        file = st.sidebar.file_uploader("Or upload a new PDF", type="pdf")

    text, chunks = "", []
    if selected_filename and selected_filename != "-- Select --":
        doc = pdfs_col.find_one({"filename": selected_filename, "user_id": user_id})
        if doc:
            text, chunks = doc["full_text"], doc["chunks"]
    elif file is not None:
        if pdfs_col.find_one({"filename": file.name, "user_id": user_id}):
            st.error("This file has already been uploaded.")
        else:
            reader = PdfReader(file)
            text = "".join(page.extract_text() or "" for page in reader.pages).replace("\n", " ").strip()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=15)
            chunks = splitter.split_text(text)
            pdfs_col.insert_one({
                "filename": file.name,
                "user_id": user_id,
                "uploaded_at": datetime.datetime.utcnow(),
                "full_text": text,
                "chunks": chunks
            })
            st.success(f"{file.name} uploaded!")

    # === Chatbot Q&A Section ===
    if chunks:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vector_store = FAISS.from_texts(chunks, embeddings)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.chat_input("Ask something about your document...")

        if user_input:
            match = vector_store.similarity_search(user_input, k=5)

            @st.cache_resource
            def get_answer_pipeline():
                from transformers import pipeline
                return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

            context = " ".join([doc.page_content for doc in match])
            qa_pipeline = get_answer_pipeline()
            answer_raw = qa_pipeline(question=user_input, context=context)['answer']

            # Simulate streaming
            bot_message_placeholder = st.empty()
            streamed_text = ""
            for word in answer_raw.split():
                streamed_text += word + " "
                bot_message_placeholder.markdown(f"**{streamed_text}**")
                time.sleep(0.05)

            # Save messages
            st.session_state.chat_history.append({"role": "user", "text": user_input})
            st.session_state.chat_history.append({"role": "bot", "text": streamed_text.strip()})

        # Display history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").markdown(message["text"])
            else:
                st.chat_message("assistant").markdown(f"**{message['text']}**")

    # === Logout ===
    if st.sidebar.button("Logout"):
        cookies["session_token"] = ""
        cookies.save()
        st.session_state.logged_in = False
        st.session_state.user = None
        st.rerun()
