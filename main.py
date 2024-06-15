# PDF Query
# document loaders *langchain documentation*

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

#open ai api key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

#pdf reader 
pdf_reader = PdfReader('ml.pdf')
raw_text = ''
for i,page in enumerate(pdf_reader.pages):
  content = page.extract_text()
  if content:
    raw_text+=content

#splitting text
text_splitter = CharacterTextSplitter(
  separator="\n",
  chunk_size = 800,
  chunk_overlap = 200,
  length_function = len,
)
texts = text_splitter.split_text(raw_text)
print(len(texts))

#Embeddings 
embeddings = OpenAIEmbeddings()
pdf_search = FAISS.from_texts(texts,embeddings)

#Q/A Chains
chain = load_qa_chain(OpenAI(),chain_type='stuff')
query = "in which language the pdf is written"
pdf = pdf_search.similarity_search(query)
response = chain.run(input_documents = pdf, question=query)
print(query)
print(response)




 
