import time
import re
import urllib.parse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import requests
from bs4 import BeautifulSoup


USER_AGENT = "MiniMindScraper/0.1 (+https://github.com/DiracSeas/minimind)"


@dataclass
class ScrapeResult:
    url: str
    title: Optional[str]
    text: str
    source: str


class RobotsCache:
    def __init__(self):
        self.cache: Dict[str, Optional[requests.Response]] = {}

    def allowed(self, base: str, path: str) -> bool:
        # Simple robots.txt respect: fetch robots once and deny disallowed paths via regex
        # For robust compliance, integrate robotexclusionrulesparser; keep lightweight here.
        try:
            parsed = urllib.parse.urlparse(base)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            if robots_url not in self.cache:
                resp = requests.get(robots_url, timeout=10, headers={"User-Agent": USER_AGENT})
                self.cache[robots_url] = resp if resp.status_code == 200 else None
            resp = self.cache.get(robots_url)
            if not resp or not resp.text:
                return True
            # naive block: lines like "Disallow: /path" for all agents
            disallows = []
            for line in resp.text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("user-agent:"):
                    # ignore agent filters: act as generic agent
                    continue
                if line.lower().startswith("disallow:"):
                    rule = line.split(":", 1)[1].strip()
                    disallows.append(rule)
            for rule in disallows:
                if rule and path.startswith(rule):
                    return False
            return True
        except Exception:
            return True


class Scraper:
    def __init__(self, base_url: str, rate_limit_sec: float = 1.0):
        self.base_url = base_url.rstrip("/")
        self.rate_limit_sec = rate_limit_sec
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.robots = RobotsCache()
        self._last_fetch = 0.0

    def _throttle(self):
        now = time.time()
        wait = self.rate_limit_sec - (now - self._last_fetch)
        if wait > 0:
            time.sleep(wait)
        self._last_fetch = time.time()

    def fetch(self, url_path: str) -> Optional[requests.Response]:
        if not url_path.startswith("/"):
            url_path = "/" + url_path
        if not self.robots.allowed(self.base_url, url_path):
            return None
        self._throttle()
        full_url = f"{self.base_url}{url_path}"
        try:
            resp = self.session.get(full_url, timeout=20, allow_redirects=True)
            ctype = resp.headers.get("Content-Type", "")
            if resp.status_code == 200 and ("text/html" in ctype or "text" in ctype or ctype == ""):
                return resp
            return None
        except Exception:
            return None

    @staticmethod
    def extract_text(html: str) -> Dict[str, Optional[str]]:
        soup = BeautifulSoup(html, "lxml")
        # Remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        title = None
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        # Prefer article-like containers if present
        article = soup.find("article") or soup.find(id=re.compile("content|main", re.I))
        body = article.get_text(" ", strip=True) if article else soup.get_text(" ", strip=True)
        return {"title": title, "text": body}

    def crawl_paths(self, paths: Iterable[str], source_name: str) -> Iterable[ScrapeResult]:
        for p in paths:
            resp = self.fetch(p)
            if not resp:
                continue
            extracted = self.extract_text(resp.text)
            text = extracted.get("text") or ""
            if len(text) < 64:
                continue
            yield ScrapeResult(url=f"{self.base_url}{p}", title=extracted.get("title"), text=text, source=source_name)

