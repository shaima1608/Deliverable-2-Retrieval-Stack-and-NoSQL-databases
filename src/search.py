import pickle
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

from .config import settings
from .stores import get_mongo, get_qdrant, load_embedding_model

INDEX_PATH = Path('reports/bm25_index.pkl')


def build_bm25_index(chunks: List[Dict], output_path: str = str(INDEX_PATH)):
    tokenized = [c['text'].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    data = {'bm25': bm25, 'chunks': chunks}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    return output_path


def load_bm25_index(path: str = str(INDEX_PATH)):
    with open(path, 'rb') as f:
        return pickle.load(f)


def normalize_scores(score_dict: Dict[str, float]) -> Dict[str, float]:
    if not score_dict:
        return {}
    vals = np.array(list(score_dict.values()), dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return {k: 1.0 for k in score_dict}
    return {k: (v - lo) / (hi - lo) for k, v in score_dict.items()}


def hybrid_search(query: str, top_k: int = 5, alpha: float = None, bm25_path: str = str(INDEX_PATH)) -> Dict:
    alpha = settings.hybrid_alpha if alpha is None else float(alpha)
    started = time.time()
    data = load_bm25_index(bm25_path)
    bm25, chunks = data['bm25'], data['chunks']

    # Lexical candidates.
    raw_bm25 = np.array(bm25.get_scores(query.lower().split()), dtype=float)
    bm25_candidates = raw_bm25.argsort()[-max(50, top_k * 10):][::-1]
    bm25_scores = {chunks[i]['chunk_id']: float(raw_bm25[i]) for i in bm25_candidates}
    bm25_scores = normalize_scores(bm25_scores)

    # Dense candidates from Qdrant.
    model = load_embedding_model()
    qvec = model.encode([query], normalize_embeddings=True)[0].tolist()
    qdrant = get_qdrant()
    dense_hits = qdrant.search(collection_name=settings.qdrant_collection, query_vector=qvec, limit=max(50, top_k * 10))
    dense_scores = {h.payload['chunk_id']: float(h.score) for h in dense_hits}
    dense_scores = normalize_scores(dense_scores)

    chunk_by_id = {c['chunk_id']: c for c in chunks}
    all_ids = set(bm25_scores) | set(dense_scores)
    fused = []
    for cid in all_ids:
        score = alpha * dense_scores.get(cid, 0.0) + (1.0 - alpha) * bm25_scores.get(cid, 0.0)
        ch = chunk_by_id.get(cid, {})
        fused.append({
            'score': float(score),
            'chunk_id': cid,
            'paper_id': ch.get('paper_id'),
            'filename': ch.get('filename'),
            'title': ch.get('title'),
            'page_start': ch.get('page_start'),
            'page_end': ch.get('page_end'),
            'citation': ch.get('citation'),
            'text': ch.get('text', '')[:700],
            'bm25_score': float(bm25_scores.get(cid, 0.0)),
            'dense_score': float(dense_scores.get(cid, 0.0)),
        })
    fused.sort(key=lambda x: x['score'], reverse=True)
    return {'query': query, 'latency_ms': round((time.time() - started) * 1000, 2), 'results': fused[:top_k]}


def mongo_lookup(chunk_id: str) -> Dict:
    db = get_mongo()
    return db[settings.mongo_chunks_collection].find_one({'chunk_id': chunk_id}, {'_id': 0}) or {}
