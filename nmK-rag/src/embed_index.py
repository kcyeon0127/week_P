import os, json, argparse
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DEF_EMB = "BAAI/bge-m3"  # 한국어/다국어 강한 임베딩

def main(chunks_jsonl: str, persist_dir: str, collection: str, emb_model: str = DEF_EMB):
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=True))
    try:
        client.delete_collection(collection)
    except:
        pass
    col = client.create_collection(collection, metadata={"hnsw:space":"cosine"})

    model = SentenceTransformer(emb_model, trust_remote_code=True)
    ids, texts, metas = [], [], []
    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="index"):
            item = json.loads(line)
            ids.append(item["chunk_id"])
            texts.append(item["text"])
            metas.append({
                "doc_id": item["doc_id"],
                "title": item["title"],
                "url": item.get("url") or "",
                "doctype": item.get("doctype") or "web",
                "lang": item.get("lang") or "ko",
                "chunk_index": item["chunk_index"],
            })
            if len(ids) >= 256:
                embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
                col.add(ids=ids, embeddings=embs, documents=texts, metadatas=metas)
                ids, texts, metas = [], [], []
    if ids:
        embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        col.add(ids=ids, embeddings=embs, documents=texts, metadatas=metas)

    print(f"[done] indexed to {persist_dir} collection='{collection}' with model {emb_model}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data_curated/chunks.jsonl")
    ap.add_argument("--persist", default="index/chroma")
    ap.add_argument("--collection", default="nmK")
    ap.add_argument("--model", default=DEF_EMB)
    args = ap.parse_args()
    main(args.chunks, args.persist, args.collection, args.model)
