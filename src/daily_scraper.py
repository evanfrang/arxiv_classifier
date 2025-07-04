import feedparser
import pandas as pd
import urllib.parse
from datetime import datetime, timezone, timedelta, date
import time
import os

def read_last_run(last_run_path):
    last_run_path_txt = last_run_path + "/last_run.txt"
    if os.path.exists(last_run_path_txt):
        with open(last_run_path_txt, "r") as f:
            return datetime.fromisoformat(f.read().strip())
    else:
        return datetime(2000, 1, 1, tzinfo=timezone.utc)

def write_last_run(dt, last_run_path):
    last_run_path_txt = last_run_path + "/last_run.txt"
    with open(last_run_path_txt, "w") as f:
        f.write(dt.isoformat())

def fetch_yesterday_arxiv_papers(since_dt, max_total, batch_size=100):
    
    base_url = "http://export.arxiv.org/api/query?"
    query = "cat:physics*"
    encoded_query = urllib.parse.quote(query)

    papers = []
    start = 0
    keep_fetching = True

    while keep_fetching and len(papers) < max_total:
        print(f"Fetching records {start} to {start + batch_size}")
        search_query = (
            f"search_query={encoded_query}"
            f"&start={start}&max_results={batch_size}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        feed_url = base_url + search_query
        feed = feedparser.parse(feed_url)

        if not feed.entries:
            print("No more entries returned. Ending pagination.")
            break

        for entry in feed.entries:
            time.sleep(3)
            published_dt = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

            if published_dt <= since_dt:
                keep_fetching = False
                print(f"Stopping early — encountered article from {published_dt.isoformat()}")
                break

            primary_cat = entry.arxiv_primary_category['term']
            paper = {
                "title": entry.title.strip().replace('\n', ' '),
                "abstract": entry.summary.strip().replace('\n', ' '),
                "primary_category": primary_cat,
                "published": entry.published
            }
            papers.append(paper)

            if len(papers) >= max_total:
                print(f"Reached max_total of {max_total} papers.")
                keep_fetching = False
                break

        start += batch_size

    return pd.DataFrame(papers)

    

def main():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")

    last_run_dt = read_last_run(DATA_DIR)
    print(f"Last run was at {last_run_dt.isoformat()}")

    df = fetch_yesterday_arxiv_papers(last_run_dt, max_total=2000)
    print(f"Fetched {len(df)} new papers since last run.")

    if not df.empty:
        today_str = date.today().strftime("%Y-%m-%d")
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
        os.makedirs(DATA_DIR, exist_ok=True)

        out_path = os.path.join(DATA_DIR, f"arxiv_{today_str}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")

        now = datetime.now(timezone.utc)
        write_last_run(now, DATA_DIR)
        print(f"Updated last run time to {now.isoformat()}")

if __name__ == '__main__':
    main()