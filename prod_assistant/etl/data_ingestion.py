from asyncio.log import logger
import os
from exception.document_exception import DocumentPortalException
from fastapi import Path
from logger.custom_logger import CustomLogger
import pandas as pd
from dotenv import load_dotenv
from typing import Iterable, List
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from prod_assistant.utils.model_loader import ModelLoader
from prod_assistant.utils.config_loader import load_config
from astrapy import DataAPIClient

from langchain_community.document_loaders  import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    JSONLoader
    
)
from pathlib import Path
import tempfile

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

log = CustomLogger().get_logger(__name__)
class DataIngestion:
    """
    Class to handle data transformation and ingestion into AstraDB vector store.
    """

    def __init__(self):
        """
        Initialize environment variables, embedding model, and set CSV file path.
        """
        print("Initializing DataIngestion pipeline...")
        self.model_loader=ModelLoader()
        self._load_env_variables()
        self.csv_path = self._get_csv_path()
        self.product_data = self._load_csv()
        self.config=load_config()

    def _load_env_variables(self):
        """
        Load and validate required environment variables.
        """
        load_dotenv()
        
        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
        
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")
        
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")

       

    def _get_csv_path(self):
        """
        Get path to the CSV file located inside 'data' folder.
        """
        current_dir = os.getcwd()
        csv_path = os.path.join(current_dir,'data', 'product_reviews.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")

        return csv_path

    def _load_csv(self):
        """
        Load product data from CSV.
        """
        df = pd.read_csv(self.csv_path)
        expected_columns = {'product_id','product_title', 'rating', 'total_reviews','price', 'top_reviews'}

        if not expected_columns.issubset(set(df.columns)):
            raise ValueError(f"CSV must contain columns: {expected_columns}")

        return df

    def transform_data(self):
        """
        Transform product data into list of LangChain Document objects.
        """
        product_list = []

        for _, row in self.product_data.iterrows():
            product_entry = {
                    "product_id": row["product_id"],
                    "product_title": row["product_title"],
                    "rating": row["rating"],
                    "total_reviews": row["total_reviews"],
                    "price": row["price"],
                    "top_reviews": row["top_reviews"]
                }
            product_list.append(product_entry)

        documents = []
        for entry in product_list:
            metadata = {
                    "product_id": entry["product_id"],
                    "product_title": entry["product_title"],
                    "rating": entry["rating"],
                    "total_reviews": entry["total_reviews"],
                    "price": entry["price"]
            }
            doc = Document(page_content=entry["top_reviews"], metadata=metadata)
            documents.append(doc)

        print(f"Transformed {len(documents)} documents.")
        return documents

    def store_in_vector_db(self, documents: List[Document]):
        """
        Store documents into AstraDB vector store.
        """      
        
        vstore = self.setup_vector_store()

        inserted_ids = vstore.add_documents(documents)
        print(f"Successfully inserted {len(inserted_ids)} documents into AstraDB.")
        return vstore, inserted_ids

    def run_pipeline(self):
        """
        Run the full data ingestion pipeline: transform data and store into vector DB.
        """
        documents = self.transform_data()
        vstore, _ = self.store_in_vector_db(documents)

        #Optionally do a quick search
        query = "Can you tell me the low budget iphone?"
        results = vstore.similarity_search(query)

        print(f"\nSample search results for query: '{query}'")
        for res in results:
            print(f"Content: {res.page_content}\nMetadata: {res.metadata}\n")

   
   

    def setup_vector_store(self):
        """Setup vector store with correct dimension"""
        
        # Load embeddings
        embeddings = self.model_loader.load_embeddings()
        
        
        # # Test dimension
        # test_vector = embeddings.embed_query("test")
        # vector_dimension = len(test_vector)
        # print(f"Embedding dimension: {vector_dimension}")  # Should be 768 for Gemini
        
        collection_name = self.config["astra_db"]["collection_name"]
        
       
        vstore = AstraDBVectorStore(
            embedding=embeddings,
            collection_name=collection_name,
            api_endpoint=self.db_api_endpoint,
            token=self.db_application_token,
            namespace=self.db_keyspace,
        )
        
        return vstore
    def save_data(self, files: Iterable):  # Accept UploadedFile objects
        """
        Read uploaded files in-memory, split content, and store in vector DB.
        """
        try:
            raw_documents = []

            for file in files:
                ext = Path(file.name).suffix.lower()

                # Read file content based on extension
                if ext in [".txt", ".md", ".csv", ".json"]:
                    content = file.getvalue().decode("utf-8", errors="ignore")
                elif ext == ".pdf":
                    # Optional: use PyMuPDF or pdfplumber to extract text from file.read()
                    raise NotImplementedError("PDF in-memory parsing not yet supported here.")
                else:
                    continue  # Skip unsupported formats

                # Wrap into Document object
                doc = Document(page_content=content, metadata={"source": file.name})
                raw_documents.append(doc)

            if not raw_documents:
                raise ValueError("No supported files were uploaded.")

            # Split documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""],
            )
            split_documents = splitter.split_documents(raw_documents)

            # Store in vector DB
            _vstore = self.setup_vector_store()
            # vstore, _ = _vstore.add_documents(split_documents)
              # Add documents - returns only list of IDs
            document_ids = _vstore.add_documents(split_documents)
            
            logger.info(
                f"Successfully ingested {len(document_ids)} documents",
                extra={
                    "document_count": len(document_ids),
                    "collection": _vstore.collection_name
                }
            )
            
            # Return vector store and IDs if needed
            return {
                "vector_store": _vstore,
                "document_ids": document_ids,
                "count": len(document_ids)
            }
           

        except Exception as e:
            log.error("Failed processing uploaded files", error=str(e))
            raise DocumentPortalException("Error processing uploaded files", e) from e
# Run if this file is executed directly
if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run_pipeline()