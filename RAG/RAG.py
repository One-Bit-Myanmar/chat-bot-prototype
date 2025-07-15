import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

class LangChainRAG:
    def __init__(self, api_key, pdf_dir="../books", model_name="gemini-2.5-pro", embedding_model="all-MiniLM-L6-v2", cache_dir="cache"):
        os.environ["GOOGLE_API_KEY"] = api_key
        self.llm = ChatGoogleGenerativeAI(model=model_name)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.cache_dir = cache_dir
        self.pdf_dir = pdf_dir
        self.vectorstore = None

    def load_and_split(self, chunk_size=500, overlap=50):
        docs = []
        pdf_paths = [os.path.join(self.pdf_dir, f) for f in os.listdir(self.pdf_dir) if f.endswith(".pdf")]
        for path in pdf_paths:
            loader = PyMuPDFLoader(path)
            docs.extend(loader.load())
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        return splitter.split_documents(docs)

    def build_or_load_vectorstore(self):
        index_path = os.path.join(self.cache_dir, "faiss_index")
        if os.path.exists(index_path):
            print("[INFO] Loading vectorstore from cache.")
            self.vectorstore = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            print("[INFO] Building new vectorstore from PDFs...")
            docs = self.load_and_split()
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            self.vectorstore.save_local(index_path)

    def ask(self, question, top_k=4):
        if self.vectorstore is None:
            self.build_or_load_vectorstore()
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        context = retriever.get_relevant_documents(question)
        context_text = "\n\n".join(doc.page_content for doc in context)

        prompt = f"""
        Your name is DevGeek and you dont have to introduce yourself unless asked specifically.
        You are a cybersecurity expert.
        Answer the following question clearly and professionally using your knowledge base.
        Do not mention the context or that it came from a document.
        {context_text}
        Question:
        {question}
        """
        response = self.llm.invoke(prompt)
        
        docs_and_scores = self.vectorstore.similarity_search_with_score(question, k=top_k)
        for i, (doc, score) in enumerate(docs_and_scores):
            print(f"[Chunk {i+1}] Score: {score:.4f}")
            print(doc.page_content[:300] + "...\n")
        return response.content
