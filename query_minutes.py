#!/usr/bin/env python3
import argparse

import app as m


def main():
    ap = argparse.ArgumentParser(description='Query meeting minutes vector DB without GUI')
    ap.add_argument('query', help='Question or keyword query')
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--no-llm', action='store_true', help='Return retrieved records only, no final LLM answer')
    args = ap.parse_args()

    records = m.search_records(args.query, top_k=args.top_k)
    print(f"Top {len(records)} records")
    for i, r in enumerate(records, 1):
        md = r.get('metadata', {})
        sim = r.get('similarity')
        print(f"\n[{i}] sim={sim:.3f} | pdf={md.get('pdf_name')} | minit={md.get('minit_number')} | date={md.get('date')}")
        print(f"tag={md.get('tag')} impact={md.get('impact')}")
        print((md.get('summary') or '')[:500])

    if not args.no_llm:
        print('\n--- Answer ---')
        ans = m.generate_chatgpt_answer(args.query, records)
        print(ans)


if __name__ == '__main__':
    main()
