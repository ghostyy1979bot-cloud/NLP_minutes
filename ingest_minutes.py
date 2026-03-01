#!/usr/bin/env python3
import argparse
import os
import shutil
import uuid

import app as m


def save_processed_result(filename: str, result: dict, dry_run: bool = False):
    header_info = result['header_info']
    minit_summaries = result['minit_summaries']

    saved = 0
    for minit_number, payload in minit_summaries.items():
        summary = (payload.get('summary') or '').strip()
        tag = (payload.get('tag') or '').strip()
        impact = (payload.get('impact') or '').strip()

        if not summary:
            continue

        embedding = m.embedding_model.encode(summary).tolist()
        metadata = {
            'pdf_name': filename,
            'committee_name': header_info.get('Committee_Name', ''),
            'minit_number': str(minit_number),
            'summary': summary,
            'tag': tag,
            'date': header_info.get('Date', ''),
            'impact': impact,
        }

        if dry_run:
            print(f"[DRY-RUN] minit={minit_number} tag={tag} impact={impact} summary={summary[:120]}")
            saved += 1
            continue

        existing = m.collection.get(
            where={"$and": [
                {"pdf_name": {"$eq": filename}},
                {"minit_number": {"$eq": str(minit_number)}}
            ]},
            include=["metadatas", "documents"]
        )

        if existing.get('metadatas'):
            record_id = existing['ids'][0]
            m.collection.update(ids=[record_id], embeddings=[embedding], metadatas=[metadata])
        else:
            m.collection.add(ids=[str(uuid.uuid4())], embeddings=[embedding], metadatas=[metadata])
        saved += 1

    return saved


def main():
    ap = argparse.ArgumentParser(description='Ingest a meeting minutes PDF into Chroma without GUI')
    ap.add_argument('pdf', help='Path to PDF file')
    ap.add_argument('--dry-run', action='store_true', help='Parse/summarize only; do not write DB')
    args = ap.parse_args()

    pdf_path = os.path.abspath(args.pdf)
    if not os.path.isfile(pdf_path):
        raise SystemExit(f"File not found: {pdf_path}")

    filename = os.path.basename(pdf_path)
    target = os.path.join(m.app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(m.app.config['UPLOAD_FOLDER'], exist_ok=True)

    if os.path.abspath(target) != pdf_path:
        shutil.copy2(pdf_path, target)

    task_id = str(uuid.uuid4())
    m.process_pdf_task(filename, task_id)
    result = m.progress_dict.get(task_id)

    if not isinstance(result, dict) or result.get('status') != 'Done':
        raise SystemExit(f"Ingest failed: {result}")

    print('Header info:')
    print(result['header_info'])
    print(f"Minutes parsed: {len(result['minit_summaries'])}")

    saved = save_processed_result(filename, result, dry_run=args.dry_run)
    mode = 'previewed' if args.dry_run else 'saved'
    print(f"Done: {mode} {saved} minute records for {filename}")


if __name__ == '__main__':
    main()
