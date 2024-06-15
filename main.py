import os 
import openai
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
from typing_extensions import Concatenate
import faiss
import tiktoken 
import streamlit as st


#get api key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

#pdf reader 
def read_pdf(file):
  pdf_reader = PdfReader(file)
  raw_text = ''
  
  for i,page in enumerate(pdf_reader.pages):
    content = page.extract_text()
    if content:
      raw_text+=content
  
  return raw_text    
      
#splitting text
def split_text(text):
  text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size = 800,
    chunk_overlap = 200,
    length_function = len,
  )
  return text_splitter.split_text(text)

#Embeddings 
def create_faiss_vector_store(texts):
  embeddings = OpenAIEmbeddings()
  return FAISS.from_texts(texts,embeddings)

#get answers
def get_answer(query, pdf_search, chain):
  pdf = pdf_search.similarity_search(query)
  return chain.run(input_documents = pdf, question=query)

def main():
  st.title('📑📌 ASK YOUR PDF')
  uploaded_file = st.file_uploader("choose a PDF file", type="pdf")
  
  if uploaded_file:
    #read and split the pdf
    raw_text = read_pdf(uploaded_file)
    texts = split_text(raw_text)
    st.write(f"⏳ Loaded {len(texts)} chunks of text from a PDF.")
    
    #create FAISS vector store
    pdf_search = create_faiss_vector_store(texts)
    st.write("🧮 FAISS vector store is created")
    
    #load QA chain 
    chain = load_qa_chain(OpenAI(),chain_type='stuff')
    st.write("🔗 Question Answering chain is loaded")
    
    #user input prompt
    query = st.text_input("Ask a Question for the PDF: ")
    if query:
      response = get_answer(query,pdf_search,chain)
      st.write(response)
  
  st.write("📥 Please uploade a pdf file to get Started")    
    
    
main()    
    
    
  




 
