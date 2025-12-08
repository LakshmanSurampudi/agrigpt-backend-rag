import os
from typing import List, Tuple, Optional, Any, Dict
from dotenv import load_dotenv

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# LangSmith imports
from langsmith import traceable

# PDF processing
from pypdf import PdfReader

# Pinecone
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

class RAGService:
    def __init__(self):
        self.embeddings = None
        self.vectorstore_citrus = None
        self.vectorstore_schemes = None
        self.llm = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    async def initialize(self):
        """Initialize all components"""
        try:
            print("Step 1: Checking environment variables...")
            google_api_key = os.getenv("GOOGLE_API_KEY")
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            if not pinecone_api_key:
                raise ValueError("PINECONE_API_KEY not found in environment variables")
        
            print(f"✅ Environment variables loaded (Google key: {google_api_key[:10]}..., Pinecone key exists)")
        
            print("Step 2: Initializing embeddings...")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=google_api_key
            )
            print("✅ Embeddings initialized successfully")
        
            print("Step 3: Initializing LLM...")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=google_api_key,
                temperature=0.3,
                convert_system_message_to_human=True
            )
            print("✅ LLM initialized successfully")
        
            print("Step 4: Initializing Pinecone...")
            pc = Pinecone(api_key=pinecone_api_key)
            print("✅ Pinecone client created")
        
            index_name = os.getenv("PINECONE_INDEX", "agrigpt-backend-rag-index")
            print(f"Step 5: Checking index: {index_name}")
        
            # Check if index exists, create if not
            existing_indexes = [index.name for index in pc.list_indexes()]
            print(f"Existing indexes: {existing_indexes}")
        
            if index_name not in existing_indexes:
                print(f"Creating new index: {index_name}")
                pc.create_index(
                    name=index_name,
                    dimension=768,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print("✅ Index created")
            else:
                print(f"✅ Index {index_name} already exists")
        
            print("Step 6: Initializing vector store for citrus...")
            self.vectorstore_citrus = PineconeVectorStore(
                index_name=index_name,
                embedding=self.embeddings,
                namespace="citrus_crop",
                pinecone_api_key=pinecone_api_key
            )
            print("✅ Citrus vector store initialized")
        
            print("Step 7: Initializing vector store for schemes...")
            self.vectorstore_schemes = PineconeVectorStore(
                index_name=index_name,
                embedding=self.embeddings,
                namespace="government_schemes",
                pinecone_api_key=pinecone_api_key
            )   
            print("✅ Schemes vector store initialized")
        
            if self.vectorstore_citrus is None or self.vectorstore_schemes is None:
                raise ValueError("Vector store initialization failed")
        
            print("✅✅✅ ALL COMPONENTS INITIALIZED SUCCESSFULLY ✅✅✅")
        
        except Exception as e:
            print(f"❌ Initialization failed")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

    async def remove_existing_file(self, filename: str, document_type: str) -> int:
        """
        Check if vectors with the given filename exist and delete them.
        Returns the number of vectors deleted.
        """
        try:
            index_name = os.getenv("PINECONE_INDEX", "agrigpt-backend-rag-index")
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(index_name)
            
            # Determine namespace based on document type
            namespace = "citrus_crop" if document_type == "citrus" else "government_schemes"
            
            # Query to find all vectors with this filename in metadata
            query_response = index.query(
                vector=[0.0] * 768,  # Dummy vector (must match dimension)
                filter={"source": {"$eq": filename}},
                top_k=10000,  # Max results to find all matches
                include_metadata=True,
                namespace=namespace
            )
            
            # Extract IDs of vectors to delete
            ids_to_delete = [match['id'] for match in query_response['matches']]
            
            if ids_to_delete:
                # Delete in batches (Pinecone limit is 1000 per batch)
                batch_size = 1000
                for i in range(0, len(ids_to_delete), batch_size):
                    batch = ids_to_delete[i:i + batch_size]
                    index.delete(ids=batch, namespace=namespace)
                
                print(f"Deleted {len(ids_to_delete)} vectors for file: {filename} in namespace: {namespace}")
                return len(ids_to_delete)
            
            return 0
            
        except Exception as e:
            raise Exception(f"Error removing existing file: {str(e)}")
    
    async def process_pdf(self, file_path: str, filename: str, document_type: str) -> int:
        """Process PDF and add to vector store"""
        try:
            # First, remove any existing vectors for this filename
            deleted_count = await self.remove_existing_file(filename, document_type)
            if deleted_count > 0:
                print(f"Removed {deleted_count} existing chunks for {filename}")
            
            # Read PDF
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "chunk": i,
                        "total_chunks": len(chunks),
                        "document_type": document_type
                    }
                )
                documents.append(doc)
            
            # Add documents to the appropriate vector store
            if document_type == "citrus":
                self.vectorstore_citrus.add_documents(documents)
            else:
                self.vectorstore_schemes.add_documents(documents)
            
            return len(documents)
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    @traceable(run_type="retriever")
    def retrieve_documents(self, query: str, document_type: str) -> List[Dict[str, Any]]:
        """Retrieve documents relevant to the query from the appropriate namespace"""
        print(f"begin retrieve_documents for {document_type}")
        import time
        retries = 3
        docs = []
        
        # Select the appropriate vectorstore
        vectorstore = self.vectorstore_citrus if document_type == "citrus" else self.vectorstore_schemes
        
        for attempt in range(retries):
            try:
                docs = vectorstore.similarity_search(query, k=5)
                break
            except Exception as e:
                if "429" in str(e) and attempt < retries - 1:
                    print(f"Rate limit hit, retrying in {2 * (attempt + 1)} seconds...")
                    time.sleep(2 * (attempt + 1))  # Exponential backoff
                    continue
                raise e
        print(f"end retrieve_documents for {document_type}")
        return [
            {
                "page_content": doc.page_content,
                "type": "Document",
                "metadata": doc.metadata
            }
            for doc in docs
        ]

    @traceable(run_type="prompt")
    def create_prompt(self, query: str, context: List[Dict[str, Any]], chat_history: List[dict] = None, document_type: str = "citrus") -> List[Any]:
        """Create the prompt for the LLM with strict scope enforcement"""
        
        # Format context
        context_str = "\n\n".join([doc["page_content"] for doc in context])
        
        # Determine the topic name
        topic_name = "Citrus Crop cultivation and management" if document_type == "citrus" else "Government agricultural schemes and programs"
        
        # STRICT system prompt with guardrails
        system_prompt = (
            f"You are a specialized AI assistant that ONLY answers questions about {topic_name}. "
            f"You have access to a knowledge base containing information about {topic_name}.\n\n"
            
            "STRICT RULES:\n"
            "1. ONLY answer questions that are directly related to the provided context below.\n"
            "2. If the question is NOT related to the context or topic, respond EXACTLY with: "
            f"\"I can only answer questions about {topic_name}. Please ask a relevant question.\"\n"
            "3. DO NOT provide general knowledge, casual conversation, or information outside the context.\n"
            "4. If the context doesn't contain enough information to answer, say: "
            "\"I don't have enough information in my knowledge base to answer that question.\"\n"
            "5. DO NOT make up or infer information that isn't in the context.\n"
            "6. Stay focused and professional. No small talk.\n\n"
            
            f"Context from knowledge base:\n{context_str}\n\n"
            
            "Remember: If the question is off-topic, politely redirect the user. Only answer what's in the context."
        )
        
        messages = [("system", system_prompt)]
        
        if chat_history:
            for item in chat_history:
                if item.get("role") == "user":
                    messages.append(("human", item.get("content", "")))
                elif item.get("role") == "assistant":
                    messages.append(("ai", item.get("content", "")))
        
        messages.append(("human", query))
        
        # Create prompt template to format messages properly
        prompt_template = ChatPromptTemplate.from_messages(messages)
        return prompt_template.format_messages()

    @traceable(run_type="llm")
    def call_llm(self, messages: List[Any]) -> str:
        """Call the LLM with the messages"""
        print("begin call_llm")
        import time
        retries = 3
        
        for attempt in range(retries):
            try:
                response = self.llm.invoke(messages)
                print("end call_llm")
                return response.content
            except Exception as e:
                if "429" in str(e) and attempt < retries - 1:
                    print(f"Rate limit hit on LLM, retrying in {2 * (attempt + 1)} seconds...")
                    time.sleep(2 * (attempt + 1))
                    continue
                raise e

    @traceable(run_type="chain")
    async def query(self, query: str, document_type: str, chat_history: List[dict] = None) -> Tuple[str, List[str]]:
        """Query the RAG system with the appropriate document type"""
        try:
            print(f"Query for {document_type}: ", query)
            
            # 1. Retrieve documents from the appropriate namespace
            retrieved_docs = self.retrieve_documents(query, document_type)
            print("Retrieved documents: ", len(retrieved_docs))
            
            # 2. Create prompt with strict scope
            messages = self.create_prompt(query, retrieved_docs, chat_history, document_type)
            
            # 3. Call LLM
            answer = self.call_llm(messages)
            print("Answer: ", answer)
            
            # Extract sources
            sources = []
            for doc in retrieved_docs:
                metadata = doc.get("metadata", {})
                source = metadata.get("source", "Unknown")
                chunk = metadata.get("chunk", 0)
                sources.append(f"{source} (chunk {chunk + 1})")
            
            return answer, sources
            
        except Exception as e:
            raise Exception(f"Error querying RAG system: {str(e)}")
    
    async def clear_knowledge_base(self, document_type: Optional[str] = None):
        """Clear all documents from the vector store or specific namespace"""
        try:
            index_name = os.getenv("PINECONE_INDEX", "agrigpt-backend-rag-index")
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(index_name)
            
            if document_type:
                # Clear specific namespace
                namespace = "citrus_crop" if document_type == "citrus" else "government_schemes"
                index.delete(delete_all=True, namespace=namespace)
                print(f"Cleared namespace: {namespace}")
            else:
                # Clear both namespaces
                index.delete(delete_all=True, namespace="citrus_crop")
                index.delete(delete_all=True, namespace="government_schemes")
                print("Cleared all namespaces")
            
        except Exception as e:
            raise Exception(f"Error clearing knowledge base: {str(e)}")