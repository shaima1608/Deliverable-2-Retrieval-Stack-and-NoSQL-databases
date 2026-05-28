from typing import Dict, List

import numpy as np
from pymongo import MongoClient, ASCENDING
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from .config import settings


def get_mongo():
    client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    db = client[settings.mongo_db]
    db[settings.mongo_docs_collection].create_index([('paper_id', ASCENDING)], unique=True)
    db[settings.mongo_chunks_collection].create_index([('chunk_id', ASCENDING)], unique=True)
    db[settings.mongo_chunks_collection].create_index([('paper_id', ASCENDING)])
    return db


def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def get_qdrant() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, timeout=30)


def reset_qdrant_collection(client: QdrantClient, vector_size: int):
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection in collections:
        client.delete_collection(settings.qdrant_collection)
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def seed_mongo(documents: List[Dict], chunks: List[Dict]):
    db = get_mongo()
    db[settings.mongo_docs_collection].delete_many({})
    db[settings.mongo_chunks_collection].delete_many({})
    if documents:
        db[settings.mongo_docs_collection].insert_many(documents)
    if chunks:
        db[settings.mongo_chunks_collection].insert_many(chunks)
    return {'documents': len(documents), 'chunks': len(chunks)}


def seed_qdrant(chunks):

    client = get_qdrant()
    model = SentenceTransformer(settings.embedding_model)

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    vector_size = len(embeddings[0])

    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

    points = []
    for i, chunk in enumerate(chunks):
        payload = {
            k: str(v) if k == "_id" else v
            for k, v in chunk.items()
        }

        points.append(
            PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload=payload
            )
        )

    batch_size = 256

    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=batch
        )
        print(f"Uploaded Qdrant points {start + len(batch)}/{len(points)}")

    return {
        "collection": settings.qdrant_collection,
        "points": len(points)
    }