import shutil
import chromadb
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory, ConversationTokenBufferMemory
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_pinecone import Pinecone as PineconeStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from chromadb.errors import InvalidDimensionException
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import tempfile
import time
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

load_dotenv('.env')

# Initialize Pinecone
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Set environment variables for LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")

class Chatbot:
    def __init__(self):
        self.pdf_caricato = False
        self.chat_history = []
        self.memory = ConversationTokenBufferMemory(memory_key="chat_history", return_messages=True, llm=ChatCohere(), max_token_limit=1200)
        self.vector_store = self.load_vector_store()
        self.qa_chain = self.create_qa_chain()

    def load_vector_store(self):
        embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
        directory = 'Documenti/docs/chroma/'
        if self.pdf_caricato:
            return self.load_pdf()
        else:
            return self.load_all_directory("./Documenti")

    def create_qa_chain(self):
        llm = ChatCohere(model="command-r-plus", temperature=0.5, max_tokens=1024, timeout=None, max_retries=2)

        condense_question_system_template = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        condense_question_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", condense_question_system_template),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, self.vector_store.as_retriever(), condense_question_prompt
        )

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise. Answer in Italian only."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )

        qa_chain = create_stuff_documents_chain(llm, qa_prompt)

        return create_retrieval_chain(history_aware_retriever, qa_chain)

    
    def load_all_directory(self, pdf_directory):
        directory = 'Documenti/docs/chroma/'
        embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

        # Check if the Chroma directory is empty
        if not os.listdir(directory):
            pdf_files = [file for file in os.listdir(pdf_directory) if file.endswith('.pdf')]

            docs = []

            for pdf_file in pdf_files:
                file_path = os.path.join(pdf_directory, pdf_file)

                loader = PyPDFLoader(file_path)
                loaded_docs = loader.load()

                for doc in loaded_docs:
                    doc.metadata["source"] = pdf_file
                    docs.append(doc)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024,
                chunk_overlap=40
            )
            documents_chunks = text_splitter.split_documents(docs)
            
            print(f"Now you have {len(documents_chunks)} chunks.")

            self.vector_store = Chroma.from_documents(documents_chunks, embeddings, persist_directory=directory)
        else:
            print("Vector store already exists in the Chroma directory. Skipping loading and splitting.")
            self.vector_store = Chroma(embedding_function=embeddings, persist_directory=directory)
            
        return self.vector_store

    def load_pdf(self, uploaded_file):
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())

        pdf_loader = PyPDFLoader(path)
        pdf_text = pdf_loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
        docs = text_splitter.split_documents(pdf_text)

        embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
        self.vector_store = FAISS.from_documents(docs, embeddings)

        try:
            self.vector_store = FAISS.from_documents(docs, embeddings)
        except InvalidDimensionException as e:
            st.error(f"Error loading PDF: {e}")
            return None, None

        self.qa_chain = self.create_qa_chain()
        self.pdf_caricato = True

        return st.success("Documento PDF caricato con successo!"), self.vector_store

# Streamlit code
st.title('Chatbot Bandi in corso Sistema Puglia')
chatbot_instance = Chatbot()

uploaded_file = st.file_uploader("Carica un file", type="pdf")
if uploaded_file:
    chatbot_instance.load_pdf(uploaded_file)

# Welcome message
with st.chat_message('assistant'):
    st.write("Ciao, sono il tuo assistente personale per rispondere a domande relative ai bandi in corso della regione Puglia presenti sul sito [link](https://www.sistema.puglia.it/)!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Inserisci la tua domanda:"):
    with st.chat_message("user"):
        st.markdown(user_input)

    result = chatbot_instance.qa_chain.invoke({
        "input": user_input,
        "chat_history": chatbot_instance.chat_history
    })

    chatbot_instance.chat_history.append({"question": user_input, "answer": result["answer"]})

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = result["answer"]
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    chatbot_instance.memory.save_context({"input": user_input}, {"output": full_response})
    st.session_state.messages.append({"role": "assistant", "content": full_response})
