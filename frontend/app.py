import streamlit as st
import requests
import uuid

API_URL = "https://rag-xfpz.onrender.com"

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide"
)

#Session state 

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


#Sidebar

with st.sidebar:
    st.title("Document Q&A")
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

    st.divider()

    # Upload
    st.subheader("Upload PDF")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded:
        if st.button("Ingest", use_container_width=True):
            with st.spinner("Ingesting..."):
                response = requests.post(
                    f"{API_URL}/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                )
                if response.ok:
                    data = response.json()
                    st.success(f"{data['filename']} — {data['pages']} pages, {data['chunks']} chunks")
                    st.caption(f"Summary: {data['summary']}")
                else:
                    st.error(f"Failed: {response.text}")

    st.divider()

    # Documents list
    st.subheader("Ingested Documents")
    try:
        docs = requests.get(f"{API_URL}/documents").json()["documents"]
        if docs:
            for doc in docs:
                with st.expander(doc["filename"]):
                    st.caption(f"Pages: {doc['pages']} | Chunks: {doc['chunks']}")
                    st.write(doc["summary"])
        else:
            st.caption("No documents yet")
    except:
        st.caption("API not reachable")

    st.divider()

    # Health
    try:
        health = requests.get(f"{API_URL}/health").json()
        st.caption(f"📡 API: online | Docs: {health['documents']} | Sessions: {health['active_sessions']}")
    except:
        st.caption("📡 API: offline")

    # New session
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()


# Chat 

st.title("Chat with your Documents")

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.caption(f"{s}")
        if msg.get("searched_docs"):
            with st.expander("Searched"):
                for d in msg["searched_docs"]:
                    st.caption(f"🗂 {d}")

# Input
if query := st.chat_input("Ask something about your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "query": query,
                        "session_id": st.session_state.session_id
                    },
                    timeout=120
                )
                if response.ok:
                    data = response.json()
                    st.write(data["answer"])

                    if data.get("sources"):
                        with st.expander("Sources"):
                            for s in data["sources"]:
                                st.caption(f"📎 {s}")

                    if data.get("searched_docs"):
                        with st.expander("Searched documents"):
                            for d in data["searched_docs"]:
                                st.caption(f"🗂 {d}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources", []),
                        "searched_docs": data.get("searched_docs", [])
                    })
                else:
                    st.error(f"Error: {response.text}")
            except requests.exceptions.Timeout:
                st.error("Request timed out. LLM taking too long.")
            except Exception as e:
                st.error(f"Error: {e}")