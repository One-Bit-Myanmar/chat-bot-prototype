import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import google as genai

class RAGPipeline:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", api_key="GOOGLE API KEY"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def read_pdf(self, pdf_paths):
        texts = []
        for path in pdf_paths:
            doc = fitz.open(path)
            for page in doc:
                texts.append(page.get_text())
            doc.close()
        return texts

    def document_chunking(self, texts, chunking_size=200):
        chunks = []
        for text in texts:
            words = text.split()
            for i in range(0, len(words), chunking_size):
                chunk = " ".join(words[i:i+chunking_size])
                chunks.append(chunk)
        return chunks

    def build_faiss_index(self, chunks):
        embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, embeddings, chunks

    def retrieve_query(self, question, index, chunks, top_k=3):
        query = self.embedding_model.encode([question], convert_to_numpy=True)
        _, top_indices = index.search(query, top_k)
        return [chunks[i] for i in top_indices[0]]

    def LLM(self, question, relevant_chunks):
        context = "\n".join(relevant_chunks)
        prompt = f"""
            You are a professional Cyber Security guard.
            Use the following context to answer the question truthfully and avoid hallucination.
            Context:
            {context}
            Question:
            {question}
        """
        response = self.model.generate_content(prompt)
        return response.text

    def run(self, pdf_paths, question):
        texts = self.read_pdf(pdf_paths)
        chunks = self.document_chunking(texts)
        index, _, chunks_map = self.build_faiss_index(chunks)
        relevant_chunks = self.retrieve_query(question, index, chunks_map)
        answer = self.LLM(question, relevant_chunks)
        return answer