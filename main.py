import openai
from langchain_community.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
import streamlit as st

# pdf reader 
def read_pdf(file):
    try:
        pdf_reader = PdfReader(file)
        raw_text = ''
        
        for i, page in enumerate(pdf_reader.pages):
            content = page.extract_text()
            if content:
                raw_text += content
        
        return raw_text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# splitting text
def split_text(text):
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        return text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return []

# embeddings 
def create_faiss_vector_store(texts, openai_api_key):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        return FAISS.from_texts(texts, embeddings)
    except Exception as e:
        if isinstance(e, openai.OpenAIError) and e.status_code == 401:
            st.error("Incorrect API key provided. Please check your API key.")
        else:
          st.error(f"Error creating FAISS vector store: {e}")
        return None

# get answers
def get_answer(query, pdf_search, chain):
    try:
        pdf = pdf_search.similarity_search(query)
        return chain.run(input_documents=pdf, question=query)
    except Exception as e:
        st.error(f"Error getting answer: {e}")
        return None

#clear chat
def clear_chat():
    if "messages" in st.session_state:
        st.session_state.messages = []  
     
def main():
    st.title('📑💭ASK YOUR PDF')
    
    # Sidebar for additional options
    st.sidebar.title('📄PDF Management')
    
    # Ask the user to enter their OpenAI API key
    api_key = st.text_input(
        "🗝️Enter your OpenAI API key:", type="password"
    )

    if not api_key:
        st.markdown("You can get your OpenAI API key from [here](https://platform.openai.com/account/api-keys).")
        return

    # Initialize OpenAI components with the provided API key
    openai_api_key = api_key

    try:
        # Load QA chain 
        chain = load_qa_chain(OpenAI(openai_api_key=openai_api_key), chain_type='stuff')
    except openai.OpenAIError as e:
        if e.status_code == 401:
            st.error("Incorrect API key provided. Please check your API key.")
        else:
            st.error(f"Error loading QA chain: {e}")
        return
    except Exception as e:
        st.error(f"Error loading QA chain: {e}")
        return

    uploaded_file = st.file_uploader("📥 Please upload a PDF file to get started", type="pdf")
    if 'previous_file' not in st.session_state:
        st.session_state.previous_file = None

    if uploaded_file:
        try:
            if uploaded_file != st.session_state.previous_file:
               st.session_state.previous_file = uploaded_file
               clear_chat()
               
            with st.spinner('Processing PDF...'):
                # Read and split the PDF
                raw_text = read_pdf(uploaded_file)
                if raw_text is None or raw_text.strip() == "":
                    st.warning("The uploaded PDF file appears to be empty or contains no readable text.")
                    return

                texts = split_text(raw_text)
                if not texts:
                  st.warning("Text splitting encountered an issue. Please try again with a different PDF.")
                  return
                
                # Create FAISS vector store
                pdf_search = create_faiss_vector_store(texts, openai_api_key)
                if pdf_search is None:
                    return  # Exit if there was an error creating the vector store

            st.success("Your PDF has been successfully uploaded✅")
            
            # Display PDF status
            st.sidebar.subheader("📌Current PDF Status:")
            st.sidebar.markdown(f"**File Name:** {uploaded_file.name}")
            st.sidebar.markdown(f"**File Size:** {round(uploaded_file.size / 1024, 2)} KB")
            st.sidebar.markdown(f"**Number of Pages:** {len(PdfReader(uploaded_file).pages)}")
            
            # Chat history management
            if st.sidebar.button("Start New Chat"):
                clear_chat()

            
            # Chat history 
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display existing chat history
            for message in st.session_state.messages:
                if message["role"] == "user":
                    with st.chat_message(message["role"], avatar='🐼'):
                        st.markdown(message["content"])
                elif message["role"] == "assistant":
                    with st.chat_message(message["role"], avatar='🤖'):
                        st.markdown(message["content"])

                    
            # User prompt        
            if prompt := st.chat_input("Ask a Question for your PDF"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message in chat message container
                with st.chat_message("user",avatar='🐼'):
                    st.markdown(prompt)
                
                # Get response
                response_text = get_answer(prompt, pdf_search, chain)
                if response_text is not None:
                    # Display assistant response in chat message container
                    with st.chat_message("assistant",avatar='🤖'):
                        st.markdown(response_text)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response_text})  

        except Exception as e:
            st.error(f"General error: {e}")


main()
