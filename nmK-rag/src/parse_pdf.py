import os, glob, argparse, pdfplumber, json
from tqdm import tqdm
from src.schema import Doc, make_id

def pdf_to_text(fp: str) -> str:
    parts = []
    with pdfplumber.open(fp) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()

def main(in_glob: str, out_dir: str, url_prefix: str = None):
    os.makedirs(out_dir, exist_ok=True)
    files = glob.glob(in_glob, recursive=True)
    for fp in tqdm(files):
        text = pdf_to_text(fp)
        title = os.path.basename(fp)
        url = f"{url_prefix.rstrip('/')}/{title}" if url_prefix else None
        doc = Doc(
            doc_id=make_id(fp),
            url=url,
            title=title,
            doctype="pdf",
            text=text,
            lang="ko"
        )
        with open(os.path.join(out_dir, f"{doc.doc_id}.json"), "w", encoding="utf-8") as f:
            f.write(doc.model_dump_json(ensure_ascii=False))
    print(f"[done] {len(files)} PDFs parsed to {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("in_glob", help="e.g., data_raw/**/*.pdf")
    ap.add_argument("-o","--out", default="data_curated")
    ap.add_argument("--url_prefix", default=None)
    args = ap.parse_args()
    main(args.in_glob, args.out, args.url_prefix)
