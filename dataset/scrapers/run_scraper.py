import argparse
import os
import sys

if __package__ is None or __package__ == "":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from dataset.scrapers.core import Scraper
from dataset.utils.util import write_jsonl 

def main():
    parser = argparse.ArgumentParser(description="Run targeted web scraping respecting robots and rate limits.")
    parser.add_argument("--base-url", default="https://github.com/jingyaogong/minimind", help="Base site URL, e.g., https://example.com")
    parser.add_argument("--paths", default='/', help="Comma-separated relative paths to crawl, e.g., /,/posts,/about")
    parser.add_argument("--urls", default='', help="Comma-separated full URLs to crawl, e.g., https://example.com/page1,https://example.com/page2")
    parser.add_argument("--rate", type=float, default=1.0, help="Rate limit per request in seconds")
    parser.add_argument("--out", default="dataset/out/scraper_out/scraped.jsonl", help="Output JSONL path")
    parser.add_argument("--source", default="web", help="Source label for records")
    args = parser.parse_args()

    scraper = Scraper(base_url=args.base_url, rate_limit_sec=args.rate)
    records_iter = []
    paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    if paths:
        records_iter.append(scraper.crawl_paths(paths, source_name=args.source))


    def recs():
        for it in records_iter:
            for s in it:
                yield {"source": s.source, "title": s.title, "text": s.text, "url": s.url}

    write_jsonl(args.out, recs())


if __name__ == "__main__":
    main()

# python dataset/scrapers/run_scraper.py \
#   --base-url https://en.wikipedia.org/wiki/Nvidia \
#   --paths / \
#   --out dataset/dataset_out/scraped.jsonl