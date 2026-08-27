import streamlit as st
import tempfile
from pathlib import Path
import importlib

import self_rag


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Self-RAG PDF Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 0px;
}

.subtitle {
    color: #777;
    font-size: 17px;
    margin-bottom: 25px;
}

.chat-user {
    background-color: #e8f0fe;
    padding: 12px 16px;
    border-radius: 12px;
    margin: 8px 0;
}

.chat-assistant {
    background-color: #f1f3f4;
    padding: 12px 16px;
    border-radius: 12px;
    margin: 8px 0;
}

.status-box {
    padding: 10px;
    border-radius: 8px;
    background-color: #f8f9fa;
    border: 1px solid #ddd;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_paths" not in st.session_state:
    st.session_state.pdf_paths = []

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

if "retriever_ready" not in st.session_state:
    st.session_state.retriever_ready = False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 PDF Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload multiple PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.write("### Selected PDFs")

        for file in uploaded_files:
            st.write(f"📄 {file.name}")

    if st.button(
        "🚀 Build Knowledge Base",
        use_container_width=True
    ):

        if not uploaded_files:

            st.warning("Please upload at least one PDF.")

        else:

            with st.spinner(
                "Loading PDFs and creating FAISS vector store..."
            ):

                temp_dir = Path(
                    tempfile.mkdtemp()
                )

                pdf_paths = []

                for uploaded_file in uploaded_files:

                    pdf_path = temp_dir / uploaded_file.name

                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    pdf_paths.append(pdf_path)

                try:

                    # Build vector store
                    self_rag.vector_store = (
                        self_rag.build_vector_store(pdf_paths)
                    )

                    # Create retriever
                    self_rag.retriever = (
                        self_rag.vector_store.as_retriever(
                            search_kwargs={"k": 4}
                        )
                    )

                    st.session_state.pdf_paths = pdf_paths
                    st.session_state.documents_loaded = True
                    st.session_state.retriever_ready = True

                    st.success(
                        f"Successfully loaded {len(pdf_paths)} PDFs."
                    )

                except Exception as e:

                    st.error(
                        f"Error while building knowledge base:\n\n{e}"
                    )


    st.divider()

    # ========================================================
    # BACKEND GRAPH
    # ========================================================

    st.header("🧠 Backend Architecture")

    graph_path = Path("backend_graph.png")

    if graph_path.exists():

        st.image(
            str(graph_path),
            caption="Self-RAG LangGraph Pipeline",
            use_container_width=True
        )

    else:

        st.warning(
            "backend_graph.png not found."
        )


    st.divider()

    # ========================================================
    # DOCUMENT STATUS
    # ========================================================

    st.header("📊 Knowledge Base")

    if st.session_state.documents_loaded:

        st.success("Knowledge base ready")

        st.write(
            f"**PDFs:** {len(st.session_state.pdf_paths)}"
        )

        for pdf in st.session_state.pdf_paths:

            st.write(
                f"📄 {pdf.name}"
            )

    else:

        st.info(
            "Upload PDFs and build the knowledge base."
        )


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 Self-RAG PDF Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Multi-PDF conversational RAG powered by LangGraph + FAISS + Groq'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SHOW CURRENT PDF STATUS
# ============================================================

if st.session_state.documents_loaded:

    st.success(
        f"📚 {len(st.session_state.pdf_paths)} PDF(s) loaded. "
        "You can now ask questions."
    )

else:

    st.info(
        "👈 Upload one or more PDFs from the sidebar and "
        "click **Build Knowledge Base**."
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message["role"]

    if role == "user":

        with st.chat_message("user"):
            st.markdown(message["content"])

    else:

        with st.chat_message("assistant"):
            st.markdown(message["content"])

            # Show metadata if available
            if message.get("metadata"):

                metadata = message["metadata"]

                with st.expander("🔍 RAG Execution Details"):

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Retrieval",
                            str(
                                metadata.get(
                                    "need_retrieval",
                                    "N/A"
                                )
                            )
                        )

                    with col2:
                        st.metric(
                            "Retrieved Docs",
                            metadata.get(
                                "retrieved_docs",
                                0
                            )
                        )

                    with col3:
                        st.metric(
                            "Relevant Docs",
                            metadata.get(
                                "relevant_docs",
                                0
                            )
                        )

                    st.write(
                        "**Grounding:**",
                        metadata.get(
                            "issup",
                            "N/A"
                        )
                    )

                    st.write(
                        "**Usefulness:**",
                        metadata.get(
                            "isuse",
                            "N/A"
                        )
                    )

                    if metadata.get("evidence"):

                        st.write("### Evidence")

                        for evidence in metadata["evidence"]:

                            st.write(
                                f"- {evidence}"
                            )

                    if metadata.get("sources"):

                        st.write("### Sources")

                        for source in metadata["sources"]:

                            st.write(
                                f"- {source}"
                            )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask something about your uploaded PDFs..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # ========================================================
    # CHECK KNOWLEDGE BASE
    # ========================================================

    if not st.session_state.retriever_ready:

        answer = (
            "Please upload your PDF files and click "
            "**Build Knowledge Base** first."
        )

        with st.chat_message("assistant"):

            st.warning(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    else:

        # ====================================================
        # RUN LANGGRAPH
        # ====================================================

        with st.chat_message("assistant"):

            with st.spinner(
                "🧠 Thinking through the Self-RAG pipeline..."
            ):

                try:

                    initial_state = {

                        "question": question,

                        "retrival_query": question,

                        "rewrite_tries": 0,

                        "need_retrieval": True,

                        "docs": [],

                        "relevant_docs": [],

                        "context": "",

                        "answer": "",

                        "issup": "",

                        "evidence": [],

                        "retries": 0,

                        "isuse": "not_useful",

                        "use_reason": "",
                    }


                    result = self_rag.app.invoke(

                        initial_state,

                        config={
                            "recursion_limit": 80
                        }

                    )


                    answer = result.get(
                        "answer",
                        "No answer found."
                    )


                    # =================================================
                    # COLLECT SOURCES
                    # =================================================

                    sources = []

                    relevant_docs = (
                        result.get(
                            "relevant_docs",
                            []
                        ) or []
                    )

                    for doc in relevant_docs:

                        metadata = (
                            doc.metadata
                            or {}
                        )

                        source = metadata.get(
                            "source",
                            "Unknown"
                        )

                        page = metadata.get(
                            "page"
                        )

                        if page is not None:

                            source_text = (
                                f"{Path(source).name} "
                                f"(page {page + 1})"
                            )

                        else:

                            source_text = (
                                Path(source).name
                            )

                        if source_text not in sources:

                            sources.append(
                                source_text
                            )


                    # =================================================
                    # METADATA
                    # =================================================

                    execution_metadata = {

                        "need_retrieval":
                            result.get(
                                "need_retrieval"
                            ),

                        "retrieved_docs":
                            len(
                                result.get(
                                    "docs",
                                    []
                                ) or []
                            ),

                        "relevant_docs":
                            len(
                                relevant_docs
                            ),

                        "issup":
                            result.get(
                                "issup"
                            ),

                        "isuse":
                            result.get(
                                "isuse"
                            ),

                        "evidence":
                            result.get(
                                "evidence",
                                []
                            ),

                        "sources":
                            sources,

                        "rewrite_tries":
                            result.get(
                                "rewrite_tries",
                                0
                            ),

                        "support_retries":
                            result.get(
                                "retries",
                                0
                            ),

                        "use_reason":
                            result.get(
                                "use_reason",
                                ""
                            )
                    }


                    # =================================================
                    # DISPLAY ANSWER
                    # =================================================

                    st.markdown(answer)


                    # =================================================
                    # SOURCES
                    # =================================================

                    if sources:

                        st.write("### 📚 Sources")

                        for source in sources:

                            st.write(
                                f"📄 {source}"
                            )


                    # =================================================
                    # DEBUG / RAG DETAILS
                    # =================================================

                    with st.expander(
                        "🔍 RAG Execution Details"
                    ):

                        col1, col2, col3 = st.columns(3)

                        with col1:

                            st.metric(
                                "Retrieved",
                                execution_metadata[
                                    "retrieved_docs"
                                ]
                            )

                        with col2:

                            st.metric(
                                "Relevant",
                                execution_metadata[
                                    "relevant_docs"
                                ]
                            )

                        with col3:

                            st.metric(
                                "Rewrites",
                                execution_metadata[
                                    "rewrite_tries"
                                ]
                            )


                        st.write(
                            "**Retrieval required:**",
                            execution_metadata[
                                "need_retrieval"
                            ]
                        )

                        st.write(
                            "**IsSUP:**",
                            execution_metadata[
                                "issup"
                            ]
                        )

                        st.write(
                            "**IsUSE:**",
                            execution_metadata[
                                "isuse"
                            ]
                        )

                        st.write(
                            "**Support revisions:**",
                            execution_metadata[
                                "support_retries"
                            ]
                        )

                        st.write(
                            "**Usefulness reason:**",
                            execution_metadata[
                                "use_reason"
                            ]
                        )


                        if execution_metadata[
                            "evidence"
                        ]:

                            st.write(
                                "### Evidence"
                            )

                            for evidence in (
                                execution_metadata[
                                    "evidence"
                                ]
                            ):

                                st.write(
                                    f"• {evidence}"
                                )


                    # =================================================
                    # SAVE MESSAGE
                    # =================================================

                    st.session_state.messages.append(

                        {
                            "role": "assistant",

                            "content": answer,

                            "metadata":
                                execution_metadata
                        }

                    )


                except Exception as e:

                    error_message = (
                        f"❌ Error while running "
                        f"Self-RAG:\n\n{e}"
                    )

                    st.error(error_message)

                    st.session_state.messages.append(

                        {
                            "role": "assistant",

                            "content": error_message
                        }

                    )