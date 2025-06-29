import os
import google as genai
from google.genai import types
import io
import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer


def read_pdf(pdf_path):
   texts = []
   for path in pdf_path:
      doc = fitz.open(path)
      for page in doc:
         texts.append(page.get_text())
      doc.close()
   return texts

def document_chunking(texts, chunking_size = 200):
   chunks = []
   for text in texts:
      words = text.split()
      for i in range(0,len(words),chunking_size):
         chunk = " ".join(words[i:i+chunking_size])
         chunks.append(chunk)
   return chunks

def build_faiss_index(chunks, embedding_model):
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, chunks

def retrieve_query (question,index,embedding_model,chunks,top_chunk = 3):
   query = embedding_model.encode([question],convert_to_numpy = True)
   _,top_index = index.search(query,top_chunk)
   return [chunks[i] for i in top_chunk[0]] #This gives  the top k relevant text chunks for your question.


def LLM (questions,relevant_chunks):
   # Setup
   api_key = "GOOGLE API KEY"
   genai.configure(api_key = api_key)
   model = genai.GenerativeModel("gemini-2.0-flash")

   context = "\n".join(relevant_chunks)
   prompt = f"""
      You are a professional Cyber Security guard.
      Use the following context to answer the question truthfully and avoid hallucination.
      Context:
      {context}
      Question:
      {questions}
   
   """
   
   response = model.generate_content(prompt)
   return response.text

def RAG_pipeline(pdf_path,question,embedding_model):
   texts = read_pdf(pdf_path)
   chunks = document_chunking(texts)
   embedder = SentenceTransformer("all-MiniLM-L6-v2")
   index,_,chunks_map = build_faiss_index(chunks,embedding_model=embedder)
   relevant_chunks = retrieve_query(question,index,embedder,chunks_map)
   answer = LLM(question,relevant_chunks)
   
   return answer
