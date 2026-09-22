
import tempfile
import os
import streamlit as st
import streamlit.components.v1 as components
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="REACT JS",
    layout="centered"
)

st.title("📝 React JS (Agent)")
st.write("📚 PDF-powered React Assistant with Web Search Fallback")

# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# Initialize Web Search Tool
web_search = DuckDuckGoSearchRun()



# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ System Config")


   # Set your API key here directly, or fetch it from OS environment variables
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

    # Automatically store it in session state
    st.session_state.api_key = GROQ_API_KEY
    user_api_key = st.session_state.api_key

    st.info("Your Groq API key is configured to wake the AI Brain.")

    st.divider()

    if st.button("🗑️ New Chat ➔", type = "secondary"):
      st.write("Resetting.....")
      st.session_state.messages = []
      st.rerun()

    st.divider()

    st.header("📚 React Documents")

    uploaded_files = st.file_uploader(
        "Upload React documents (Optional)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("📖 Process PDFs"):
        if not uploaded_files:
            st.warning("⚠️ Please upload at least one React PDF.")
        else:
            all_documents = []
            progress = st.progress(0)

            for index, uploaded_file in enumerate(uploaded_files):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_pdf_path = temp_file.name

                    loader = PyPDFLoader(temp_pdf_path)
                    documents = loader.load()
                    all_documents.extend(documents)
                    os.unlink(temp_pdf_path)

                    progress.progress((index + 1) / len(uploaded_files))

                except Exception as e:
                    st.error(f"❌ Error reading {uploaded_file.name}: {e}")

            if all_documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )
                chunks = text_splitter.split_documents(all_documents)

                with st.spinner("🧠 Creating document knowledge base..."):
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    vectorstore = FAISS.from_documents(chunks, embeddings)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.pdf_names = [file.name for file in uploaded_files]

                st.success(f"✅ {len(uploaded_files)} PDF(s) processed successfully!")

    if st.session_state.pdf_names:
        st.divider()
        st.subheader("📄 Loaded Documents")
        for pdf_name in st.session_state.pdf_names:
            st.write(f"✅ {pdf_name}")

    st.divider()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================================================
# CHAT INPUT & GENERATION FLOW
# ============================================================

if user_query := st.chat_input("💬 Ask a question or chat casually..."):
    if not user_api_key:
        st.error("❌ Please enter your Groq API key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        llm = ChatGroq(
            temperature=0.7,
            model="openai/gpt-oss-120b",
            api_key=user_api_key
        )

        # --------------------------------------------------------
        # STEP 1: CASUAL CONVERSATION CHECK
        # --------------------------------------------------------
        casual_check_prompt = f"""
Determine if the following message is purely casual conversation or small talk (e.g. greetings, 'how are you', 'hi', 'who are you', 'thanks').
Respond strictly with ONLY 'YES' if it is casual conversation, or 'NO' if it requires looking up technical or document information.

User message: {user_query}
"""
        is_casual = "YES" in llm.invoke(casual_check_prompt).content.strip().upper()

        if is_casual:
            with st.spinner("🧠 Responding..."):
                bot_answer = llm.invoke(f"You are a friendly AI assistant specializing in React JS. Respond warmly to: {user_query}").content

        else:
            # --------------------------------------------------------
            # STEP 2: DOCUMENT AVAILABILITY VALIDATION
            # --------------------------------------------------------
            if st.session_state.vectorstore is None:
                bot_answer = "⚠️ No PDF documents uploaded. Please upload a React document in the sidebar to proceed with document-based queries."
            else:
                # --------------------------------------------------------
                # STEP 3: RETRIEVE PDF CONTEXT & VERIFY ANSWER AVAILABILITY
                # --------------------------------------------------------
                relevant_docs = st.session_state.vectorstore.similarity_search(user_query, k=4)
                pdf_context = "\n\n".join([doc.page_content for doc in relevant_docs])

                has_pdf_answer = False
                if pdf_context.strip():
                    verification_prompt = f"""
Given the context below, determine if it contains sufficient information to answer the query.
Respond ONLY with 'YES' or 'NO'.

Context:
{pdf_context}

Query: {user_query}
"""
                    has_pdf_answer = "YES" in llm.invoke(verification_prompt).content.strip().upper()

                # --------------------------------------------------------
                # STEP 4: WEB SEARCH FALLBACK (IF PDF DOES NOT HAVE DATA)
                # --------------------------------------------------------
                search_context = ""
                if not has_pdf_answer:
                    with st.spinner("🌐 Document didn't contain the full answer. Searching the web..."):
                        try:
                            search_context = web_search.run(user_query)
                        except Exception as e:
                            search_context = f"Web search failed: {e}"

                # --------------------------------------------------------
                # STEP 5: CONSTRUCT PROMPT & GENERATE RESPONSE
                # --------------------------------------------------------
                source_type = "PDF Documentation" if has_pdf_answer else "Web Search"
                active_context = pdf_context if has_pdf_answer else search_context

                system_prompt = f"""
You are the React JS Expert AI Assistant.

BEHAVIOR INSTRUCTIONS:
1. ACCURATE ANSWERS: Answer the user's question clearly based on the provided CONTEXT.
2. REACTJS / DOCUMENT QUESTIONS: If the user asks specific questions about uploaded documents or InGrid operations, answer using the PDF CONTEXT provided below.
3. UNCERTAINTY: If documents are uploaded but the answer cannot be found in the provided context or general context, politely state that the information isn't present in the documents.
4. SOURCE ALERT: State explicitly at the beginning of your response whether the details were retrieved from uploaded PDF documents or Web Search.

=========================================================
ACTIVE CONTEXT (Source: {source_type})
=========================================================
{active_context if active_context else "No context available."}
"""

                prompt = f"{system_prompt}\n\nUser Question: {user_query}\nAnswer:"

                with st.spinner("🧠 AI is thinking..."):
                    response = llm.invoke(prompt)
                    bot_answer = response.content

        # --------------------------------------------------------
        # RECORD & DISPLAY ANSWER
        # --------------------------------------------------------
        st.session_state.messages.append({"role": "assistant", "content": bot_answer})

        with st.chat_message("assistant"):
            st.markdown(bot_answer)



# ============================================================
# FLOATING SCROLL-DOWN BUTTON
# ============================================================

components.html(
    """
    <script>
    const doc = window.parent.document;

    // Remove old button if Streamlit reruns
    const oldButton = doc.getElementById("scrollDownButton");
    if (oldButton) {
        oldButton.remove();
    }

    // Create button
    const button = doc.createElement("button");

    button.id = "scrollDownButton";
    button.innerHTML = "↓";
    button.title = "Go to last message";

    // Button style
    button.style.position = "fixed";
    button.style.right = "25px";
    button.style.bottom = "50px";
    button.style.width = "45px";
    button.style.height = "45px";
    button.style.borderRadius = "50%";
    button.style.border = "1px solid #cccccc";
    button.style.backgroundColor = "green";
    button.style.color = "white";
    button.style.fontSize = "28px";
    button.style.fontWeight = "bold";
    button.style.cursor = "pointer";
    button.style.zIndex = "999999";
    button.style.boxShadow = "0 2px 8px rgba(0,0,0,0.3)";

    // Click event
    button.addEventListener("click", function() {

        // Find Streamlit's main scroll container
        const main = doc.querySelector(
            '[data-testid="stAppViewContainer"]'
        );

        if (main) {
            main.scrollTo({
                top: main.scrollHeight,
                behavior: "smooth"
            });
        }

        // Also try the main section
        const section = doc.querySelector(
            '[data-testid="stMain"]'
        );

        if (section) {
            section.scrollTo({
                top: section.scrollHeight,
                behavior: "smooth"
            });
        }

        // Scroll to last chat message
        const messages = doc.querySelectorAll(
            '[data-testid="stChatMessage"]'
        );

        if (messages.length > 0) {

            messages[messages.length - 1].scrollIntoView({
                behavior: "smooth",
                block: "end"
            });
        }
    });

    // Add button to Streamlit page
    doc.body.appendChild(button);
    </script>
    """,
    height=0,
)
