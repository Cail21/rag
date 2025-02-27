import streamlit as st
from langchain_community.document_loaders import TextLoader
from pypdf import PdfReader
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
load_dotenv()

print("bot.py è stato eseguito")


def read_pdf(file):
    document = ""

    reader = PdfReader(file)
    for page in reader.pages:
        document += page.extract_text()

    return document


import codecs

import codecs

def read_txt(file):
    # Leggi i byte dal file
    raw_bytes = file.getvalue()
    try:
        s = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        s = raw_bytes.decode("latin1")
    # Se nella stringa sono presenti sequenze di escape, prova a correggerle
    if "\\x" in s:
        try:
            # Primo passaggio: decodifica delle sequenze di escape
            s = codecs.decode(s, "unicode_escape")
            # Secondo passaggio: corregge l'encoding (da latin1 a utf-8)
            s = s.encode("latin1").decode("utf-8")
        except Exception as e:
            # In caso di errore, lascia il testo così com'è
            pass
    # Sostituisci newline e return, se necessario
    s = s.replace("\n", " \\n ").replace("\r", " \\r ")
    return s







def split_doc(document, chunk_size, chunk_overlap):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split = splitter.split_text(document)
    split = splitter.create_documents(split)

    return split


def embedding_storing( split, create_new_vs, existing_vector_store, new_vs_name):
    if create_new_vs is not None:
        instructor_embeddings =HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        # Implement embeddings
        db = FAISS.from_documents(split, instructor_embeddings)

        if create_new_vs == True:
            # Save db
            db.save_local("vectorstore/" + new_vs_name)
        else:
            # Load existing db
            load_db = FAISS.load_local(
                "vectorstore/" + existing_vector_store,
                instructor_embeddings,
                allow_dangerous_deserialization=True
            )
            # Merge two DBs and save
            load_db.merge_from(db)
            load_db.save_local("vectorstore/" + new_vs_name)
            
        st.success("The document has been saved.")
        
        

def prepare_rag_llm(token, vector_store_list, temperature, max_length):
    # Controlla se il token è presente; se non lo è, solleva un errore.
    if not token:
        raise ValueError("HuggingFace API token is missing. Please set the HUGGINGFACEHUB_API_TOKEN environment variable or pass it as a parameter.")
        
    instructor_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    # Carica il vector store
    loaded_db = FAISS.load_local(
        f"vectorstore/{vector_store_list}",
        instructor_embeddings,
        allow_dangerous_deserialization=True
    )

    # Prompt rafforzato per forzare il ruolo di JFK
    qa_template = """
[System: You are a psychological assiant that helps people quit smoking.]

Question: {question}
Context: {context}

Risposta: 
"""
    qa_prompt = PromptTemplate(template=qa_template, input_variables=["question", "context"])

    # Carica l'LLM passando esplicitamente il token
    llm = HuggingFaceHub(
        repo_id='tiiuae/falcon-7b-instruct',
        model_kwargs={"temperature": temperature, "max_length": max_length},
        huggingfacehub_api_token=token  # il token viene passato direttamente qui
    )

    memory = ConversationBufferWindowMemory(
        k=0,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    # Passa il prompt personalizzato direttamente con combine_docs_chain_kwargs
    qa_conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=loaded_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

    return qa_conversation






def generate_answer(question, token):
    
    if question.lower().strip() in ["chi sei", "who are you"]:
        return "Sono il Presidente John F. Kennedy. Come posso aiutarti oggi?", []
    
    if not token:
        return "Nessun token fornito, impossibile generare la risposta.", []

    response = st.session_state.conversation({"question": question})
    answer = response.get("answer", "").strip()

    marker = "Risposta:"
    if marker in answer:
        answer = answer.split(marker, 1)[-1].strip()

    # Se la risposta contiene sequenze esadecimali, prova a correggerle
    if "\\x" in answer:
        try:
            answer = codecs.decode(answer, "unicode_escape")
            answer = answer.encode("latin1").decode("utf-8")
        except Exception as e:
            pass

    explanation = response.get("source_documents", [])
    doc_source = [d.page_content for d in explanation]
    return answer, doc_source





