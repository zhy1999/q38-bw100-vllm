#!/usr/bin/env python3
"""ShareGPT serving benchmark: zero-framework client for OpenAI-compatible vLLM.

Measures output throughput (tok/s), TTFT, TPOT, latency per request.
Streaming (SSE) is used so TTFT is measured on the first token.
"""
import argparse, json, random, threading, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import requests

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base-url', default='http://127.0.0.1:18680/v1')
    p.add_argument('--served-model-name', default='qwen38-vllm021')
    p.add_argument('--tokenizer-dir', default='/data/tools/vllm-lunch/qwen38-tokenizer')
    p.add_argument('--dataset-path', required=True)
    p.add_argument('--sharegpt-context-len', type=int, default=131072)
    p.add_argument('--num-prompts', type=int, default=1000)
    p.add_argument('--max-concurrency', type=int, default=1)
    p.add_argument('--max-tokens', type=int, default=1024)
    p.add_argument('--warmup-requests', type=int, default=0)
    p.add_argument('--seed', type=int, default=20260912)
    p.add_argument('--output-file', required=True)
    return p.parse_args()

def load_prompts(args, tokenizer):
    rows = []
    with open(args.dataset_path, encoding='utf-8') as f:
        first = f.read(1)
    objs = []
    if first == '[':
        # pretty-printed JSON array (one conversation per object)
        with open(args.dataset_path, encoding='utf-8') as f:
            objs = json.load(f)
    else:
        # JSONL: one conversation per line
        with open(args.dataset_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                objs.append(json.loads(line))
    for obj in objs:
        conv = obj.get('conversations') or []
        texts = [m.get('value', '') for m in conv if m.get('from') == 'human']
        if texts:
            rows.append(texts[0])
    rng = random.Random(args.seed)
    rows = rng.sample(rows, min(args.num_prompts, len(rows)))
    # truncate with tokenizer to context len
    prompts = []
    for t in rows:
        ids = tokenizer(t, add_special_tokens=False)['input_ids']
        if len(ids) > args.sharegpt_context_len:
            ids = ids[:args.sharegpt_context_len]
            t = tokenizer.decode(ids)
        prompts.append(t)
    return prompts

def send_one(base, model, prompt, max_tokens, out):
    url = base.rstrip('/') + '/chat/completions'
    payload = dict(model=model, messages=[{'role': 'user', 'content': prompt}],
                   max_tokens=max_tokens, temperature=0.0, stream=True, seed=20260912)
    t0 = time.perf_counter()
    ttft = latency = None
    n_out = 0
    try:
        with requests.post(url, json=payload, stream=True, timeout=1800) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                chunk = json.loads(data)
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000.0
                usage = chunk.get('usage')
                if usage:
                    n_out = usage.get('completion_tokens', n_out)
            latency = (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        out.append(dict(error=str(e)))
        return
    if ttft is None:
        ttft = latency
    out.append(dict(ttft_ms=ttft, tpot_ms=(latency - ttft) / n_out if n_out else None,
                    latency_ms=latency, output_tokens=n_out))

def main():
    args = parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    prompts = load_prompts(args, tok)
    results = []
    lock = threading.Lock()
    def worker(p):
        send_one(args.base_url, args.served_model_name, p, args.max_tokens, results)
    for _ in range(args.warmup_requests):
        send_one(args.base_url, args.served_model_name, prompts[0], args.max_tokens, [])
        print(f'[warmup] {_+1}/{args.warmup_requests}', flush=True)
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.max_concurrency) as ex:
        list(ex.map(worker, prompts))
    wall = time.perf_counter() - t_start
    ok = [r for r in results if 'error' not in r]
    n_out = sum(r['output_tokens'] or 0 for r in ok)
    def pct(lst, q):
        s = sorted(lst); i = min(len(s) - 1, int(q * len(s))); return s[i]
    ttfts = [r['ttft_ms'] for r in ok]
    tpots = [r['tpot_ms'] for r in ok if r['tpot_ms']]
    lats = [r['latency_ms'] for r in ok]
    meta = dict(started_at=datetime.now(timezone.utc).isoformat(), wall_s=round(wall, 2),
                requests=len(prompts), ok=len(ok), output_tokens=n_out,
                throughput_tps=round(n_out / wall, 2),
                ttft_ms_mean=round(sum(ttfts)/len(ttfts), 2) if ttfts else None,
                ttft_ms_p50=round(pct(ttfts, .5), 2) if ttfts else None,
                ttft_ms_p95=round(pct(ttfts, .95), 2) if ttfts else None,
                tpot_ms_mean=round(sum(tpots)/len(tpots), 2) if tpots else None,
                tpot_ms_p50=round(pct(tpots, .5), 2) if tpots else None,
                tpot_ms_p95=round(pct(tpots, .95), 2) if tpots else None,
                latency_ms_mean=round(sum(lats)/len(lats), 2) if lats else None)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

if __name__ == '__main__':
    main()
