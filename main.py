import openai
from langchain_community.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
import streamlit as st
from typing import List,Optional


#PDF Reader Function
def read_pdf(file) -> Optional[str]:
    '''Reads text from a PDF file'''
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

#Text Splitting Function
def split_text(text:str) -> List[str]:
    '''Splits text into chuncks for processing'''
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

#Text Embedding Function
def create_faiss_vector_store(texts: List[str], openai_api_key: str) -> Optional[FAISS]:
    '''Creates a FAISS vector store for the given texts'''
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        return FAISS.from_texts(texts, embeddings)
    
    except Exception as e:
        if isinstance(e, openai.OpenAIError) and e.status_code == 401:
            st.error("Incorrect API key provided. Please check your API key.")
        else:
          st.error(f"Error creating FAISS vector store: {e}")
        return None

#Get Answers From PDF
def get_answer(query: str, pdf_search: FAISS, chain: load_qa_chain) -> Optional[str]:
    '''Gets an answer to the query from the PDF content'''
    try:
        pdf = pdf_search.similarity_search(query)
        return chain.run(input_documents=pdf, question=query)
    
    except Exception as e:
        st.error(f"Error getting answer: {e}")
        return None

#Clear Chat History Function
def clear_chat() -> None:
    '''Clears the chat history'''
    if "messages" in st.session_state:
        st.session_state.messages = []  
     
def main() -> None:
    
    st.title('📑💭ASK YOUR PDF')
    st.sidebar.title('📄PDF Management')
    
    #Get openAi API KEY
    api_key = st.text_input(
        "🗝️Enter your OpenAI API key:", type="password"
    )
    if not api_key:
        st.markdown("You can get your OpenAI API key from [here](https://platform.openai.com/account/api-keys).")
        return
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

    #pdf file uploader 
    uploaded_file = st.file_uploader("📥 Please upload a PDF file to get started", type="pdf")
    if 'previous_file' not in st.session_state:
        st.session_state.previous_file = None

    if uploaded_file:
        try:
            #clear chat if new file is uploaded
            if uploaded_file != st.session_state.previous_file:
               st.session_state.previous_file = uploaded_file
               clear_chat()
               
            with st.spinner('Processing PDF...'):
                #Read and split the PDF
                raw_text = read_pdf(uploaded_file)
                if raw_text is None or raw_text.strip() == "":
                    st.warning("The uploaded PDF file appears to be empty or contains no readable text.")
                    return

                texts = split_text(raw_text)
                if not texts:
                  st.warning("Text splitting encountered an issue. Please try again with a different PDF.")
                  return
                
                #Create FAISS vector store
                pdf_search = create_faiss_vector_store(texts, openai_api_key)
                if pdf_search is None:
                    return  #Exit if there was an error creating the vector store

            st.success("Your PDF has been successfully uploaded✅")
            
            #display PDF status
            st.sidebar.subheader("📌Current PDF Status:")
            st.sidebar.markdown(f"**File Name:** {uploaded_file.name}")
            st.sidebar.markdown(f"**File Size:** {round(uploaded_file.size / 1024, 2)} KB")
            st.sidebar.markdown(f"**Number of Pages:** {len(PdfReader(uploaded_file).pages)}")
            
            #chat history management
            if st.sidebar.button("Start New Chat"):
                clear_chat()

            
            #chat history 
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            #display existing chat history
            for message in st.session_state.messages:
                if message["role"] == "user":
                    with st.chat_message(message["role"], avatar='👤'):
                        st.markdown(message["content"])
                elif message["role"] == "assistant":
                    with st.chat_message(message["role"], avatar='🤖'):
                        st.markdown(message["content"])

                    
            #user prompt        
            if prompt := st.chat_input("Ask a Question for your PDF"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                #display user message in chat message container
                with st.chat_message("user",avatar='👤'):
                    st.markdown(prompt)
                
                #get response
                response_text = get_answer(prompt, pdf_search, chain)
                if response_text is not None:
                    #display assistant response in chat message container
                    with st.chat_message("assistant",avatar='🤖'):
                        st.markdown(response_text)
                    
                    #add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response_text})  

        except Exception as e:
            st.error(f"General error: {e}")


main()
