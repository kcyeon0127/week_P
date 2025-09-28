import os, json, requests
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from threading import Thread

def use_ollama() -> bool:
    return os.getenv("OLLAMA_MODEL") is not None

def chat_ollama(system: str, user: str, context_snippets: List[Dict]) -> str:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL")
    assert model, "Set OLLAMA_MODEL to use Ollama."
    ctx = "\n\n".join([f"[{i+1}] {s['title']} — {s.get('url','')}\n{s['text']}" for i, s in enumerate(context_snippets)])
    prompt = f"{system}\n\n[컨텍스트]\n{ctx}\n\n[질문]\n{user}\n\n위 컨텍스트만 근거로 한국어로 답하세요. 답 뒤에 (출처: [번호]) 형태로 인용표기."
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}}
    r = requests.post(f"{url}/api/generate", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["response"]

_pipe_cache = {}

def chat_hf(system: str, user: str, context_snippets: List[Dict]) -> str:
    model_name = os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    key = model_name
    if key not in _pipe_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
        )
        _pipe_cache[key] = (tokenizer, model)
    tokenizer, model = _pipe_cache[key]
    ctx = "\n\n".join([f"[{i+1}] {s['title']} — {s.get('url','')}\n{s['text']}" for i, s in enumerate(context_snippets)])
    messages = [
        {"role":"system","content": system},
        {"role":"user","content": f"[컨텍스트]\n{ctx}\n\n[질문]\n{user}\n\n규칙: 컨텍스트 근거 없는 내용은 쓰지 말고, 답 뒤에 (출처: [번호]) 인용표기."}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.0
    )
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    out = []
    for t in streamer:
        out.append(t)
    return "".join(out).strip()

def chat(system: str, user: str, context_snippets: List[Dict]) -> str:
    if use_ollama():
        return chat_ollama(system, user, context_snippets)
    return chat_hf(system, user, context_snippets)
