import gzip
import json
import os
from datetime import datetime, timedelta
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data", "bulk")

def parse_and_filter(filename, cutoff_date):
    papers = []
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            paper = json.loads(line)
            
            submitted_str = paper.get('submitted') or paper.get('created')
            if not submitted_str:
                continue
            try:
                submitted_dt = datetime.strptime(submitted_str, "%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                continue

            if submitted_dt < cutoff_date:
                continue

            title = paper.get('title', '').replace('\n', ' ').strip()
            abstract = paper.get('abstract', '').replace('\n', ' ').strip()
            #categories_str = paper.get('categories', '')
            #categories = categories_str.split()
            primary_cat = paper.get('primar_category').strip()

            papers.append({
                'title': title,
                'abstract': abstract,
                'primary_category': primary_cat,
                'submitted': submitted_str,
            })

    return papers

def main():
    cutoff_date = datetime.now() - timedelta(days=365)
    all_papers = []

    for month_offset in range(1):
        dt = datetime.now() - timedelta(days=30*month_offset)
        filename = os.path.join(DATA_DIR, f"arXiv_src_{dt.strftime('%Y%m')}.json.gz")
        if not os.path.exists(filename):
            print(f"File not found: {filename}, skipping...")
            continue
        print(f"Parsing {filename} ...")
        papers = parse_and_filter(filename, cutoff_date)
        print(f"  Found {len(papers)} papers from last 12 months in this file")
        all_papers.extend(papers)

    df = pd.DataFrame(all_papers)
    print(f"Total papers collected: {len(df)}")

    output_path = os.path.join(DATA_DIR, "arxiv_last_12_months.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved all papers to {output_path}")

if __name__ == "__main__":
    main()