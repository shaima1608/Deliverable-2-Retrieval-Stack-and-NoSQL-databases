from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    pdf_dir: str = os.getenv('PDF_DIR', './data/pdfs')
    questions_path: str = os.getenv('QUESTIONS_PATH', './data/question.xlsx')
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = os.getenv('MONGO_DB', 'csai415_d2')
    mongo_docs_collection: str = os.getenv('MONGO_DOCS_COLLECTION', 'documents')
    mongo_chunks_collection: str = os.getenv('MONGO_CHUNKS_COLLECTION', 'chunks')
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = os.getenv('QDRANT_COLLECTION', 'paper_chunks')
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password: str = os.getenv('NEO4J_PASSWORD', 'password123')
    embedding_model: str = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    hybrid_alpha: float = float(os.getenv('HYBRID_ALPHA', '0.55'))
    chunk_size: int = int(os.getenv('CHUNK_SIZE', '900'))
    chunk_overlap: int = int(os.getenv('CHUNK_OVERLAP', '180'))

settings = Settings()
