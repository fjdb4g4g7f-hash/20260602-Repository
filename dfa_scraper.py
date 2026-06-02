"""
DFA (Department of Foreign Affairs) News Scraper - Python version

Uses Wayback Machine archives to access dfa.gov.ph articles about WPS/SCS.
Direct access is blocked by Cloudflare, but Wayback Machine archives work.

Usage: python3 dfa_scraper.py
Output: dfa_WPS_data.json
"""

import json
import os
import re
import time
import requests
from datetime import datetime
from collections import defaultdict

# Configuration
CONFIG = {
    'dateFrom': '2022-07-01',
    'dateTo': '2026-05-31',
    'outputFile': os.path.join(os.path.dirname(__file__), 'dfa_WPS_data.json'),
    'delay': 1.0,  # seconds between Wayback Machine requests
}

# WPS/SCS keywords for URL filtering
URL_WPS_KEYWORDS = [
    'west-philippine', 'west-philippine', 'west_philippine',
    'south-china-sea', 'south+china+sea',
    '/wps', '-wps', 'wps.', 'wps-',
    'maritime', 'arbitral', 'unclos',
    'kalayaan', 'pag-asa', 'ayungin',
    'fisheries', 'coast-guard', 'coastguard',
    'reef', 'sovereignty', 'territorial',
    'chinese-vessel', 'china-sea',
    'naval', 'exclusive-economic', 'continental-shelf',
    'submission', 'extended-continental',
    'benham', 'philippine-rise',
    'scarborough', 'spratly', 'pancake',
]

# Primary content keywords that indicate WPS/SCS relevance
CONTENT_PRIMARY = [
    'west philippine sea', 'south china sea',
    'west philippine sea (wps)', 'wps',
]

# Secondary content keywords (weaker signal)
CONTENT_SECONDARY = [
    'arbitral award', 'unclos', 'ayungin shoal',
    'pagtindig para sa west philippine sea',
    'defending west philippine',
    'chinese coast guard', 'fishing ban',
    'bajo de masinloc', 'scarborough shoal',
    'kalayaan island', 'pag-asa island',
    'extended continental shelf',
    'maritime security',
]


def is_wps_url(url):
    """Check if URL slug suggests WPS/SCS content."""
    slug = url.lower()
    return any(kw in slug for kw in URL_WPS_KEYWORDS)


def has_wps_content(text):
    """Check if text contains WPS/SCS keywords."""
    lower = text.lower()
    primary_matches = [kw for kw in CONTENT_PRIMARY if kw in lower]
    secondary_matches = [kw for kw in CONTENT_SECONDARY if kw in lower]
    return primary_matches, secondary_matches


def extract_date(text):
    """Extract article date from text content."""
    # Look for "DD Month YYYY" pattern
    patterns = [
        r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
        r'(\d{4}-\d{2}-\d{2})',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            try:
                if '-' in match.group(1):
                    return match.group(1)
                else:
                    dt = datetime.strptime(match.group(1), '%d %B %Y')
                    return dt.strftime('%Y-%m-%d')
            except:
                pass
    return ''


def fetch_wayback_article(url, timestamp):
    """Fetch an archived DFA article from Wayback Machine."""
    wb_url = f"https://web.archive.org/web/{timestamp}/{url}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    try:
        r = requests.get(wb_url, timeout=30, headers=headers)
        if r.status_code != 200:
            return None

        html = r.text

        # Skip if blocked by Cloudflare
        if 'Just a moment' in html:
            return None

        # Extract title from HTML
        title_match = re.search(r'<title>(.*?)</title>', html, re.DOTALL)
        title = title_match.group(1).strip() if title_match else ''

        # Strip HTML tags to get text
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove Wayback Machine specific text
        text = re.sub(r'The Wayback Machine - https://web\.archive\.org/.*?\s', '', text)
        text = re.sub(r'__wm\.\w+\([^)]+\)', '', text)

        # Check if we got meaningful content
        if len(text) < 100:
            return None

        # Extract date
        article_date = extract_date(text)

        return {
            'title': title,
            'text': text,
            'articleDate': article_date,
            'wbTimestamp': timestamp,
        }

    except Exception as e:
        return None


def load_urls(filepath):
    """Load URLs from a tab-separated file (timestamp\turl)."""
    urls = {}
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    ts, url = line.split('\t', 1)
                    clean = url.split('?')[0].rstrip('/')
                    # Filter out listing pages
                    slug = clean.split('/')[-1] if '/' in clean else ''
                    if slug and len(slug) > 5 and not slug.startswith('start='):
                        if clean not in urls or ts > urls[clean]:
                            urls[clean] = ts
    return urls


def main():
    print('=== DFA News Scraper (via Wayback Machine) ===')
    print(f'Date range: {CONFIG["dateFrom"]} ~ {CONFIG["dateTo"]}')
    print()

    # Step 1: Load all URL sources
    script_dir = os.path.dirname(__file__)
    all_urls = {}

    for fname in ['dfa_article_urls.txt', 'dfa_statements_urls.txt']:
        fpath = os.path.join(script_dir, fname)
        urls = load_urls(fpath)
        print(f'Loaded {len(urls)} URLs from {fname}')
        for url, ts in urls.items():
            if url not in all_urls or ts > all_urls[url]:
                all_urls[url] = ts

    print(f'Total combined articles: {len(all_urls)}')

    # Step 2: Filter by URL keywords
    candidates = []
    for url, ts in all_urls.items():
        if is_wps_url(url):
            # Skip known false positive patterns
            slug = url.lower()
            if 'west-edmonton' in slug or 'western-australia' in slug or 'western-china' in slug:
                continue
            candidates.append({'url': url, 'timestamp': ts, 'slug': url.split('/')[-1]})

    print(f'URL-filtered candidates: {len(candidates)}')
    print()

    # Step 3: Fetch each candidate from Wayback Machine
    results = []
    total_wps = 0

    for i, cand in enumerate(candidates):
        slug_short = cand['slug'][:60]
        print(f'[{i+1}/{len(candidates)}] {slug_short}... ', end='', flush=True)

        data = fetch_wayback_article(cand['url'], cand['timestamp'])

        if data and len(data['text']) > 100:
            primary_kw, secondary_kw = has_wps_content(data['text'])
            if primary_kw or secondary_kw:
                results.append({
                    'title': data['title'].replace(' | Department of Foreign Affairs', '').replace(' - DFA', '').strip(),
                    'url': cand['url'],
                    'text': data['text'][:50000],
                    'date': data['articleDate'],
                    'wbTimestamp': data['wbTimestamp'],
                    'keywords': primary_kw + secondary_kw,
                    'source': 'DFA',
                    'country': 'Philippines',
                })
                total_wps += 1
                print(f'✓ WPS ({len(primary_kw + secondary_kw)} kw)')
            else:
                print(f'✗ no WPS')
        else:
            print(f'✗ empty/failed')

        time.sleep(CONFIG['delay'])

    # Step 4: Filter by date range
    from_date = datetime.fromisoformat(CONFIG['dateFrom'])
    to_date = datetime.fromisoformat(CONFIG['dateTo'])
    in_range = []

    for article in results:
        if article['date']:
            try:
                d = datetime.fromisoformat(article['date']) if 'T' in article['date'] else datetime.strptime(article['date'], '%Y-%m-%d')
                if from_date <= d <= to_date:
                    in_range.append(article)
            except:
                in_range.append(article)
        else:
            in_range.append(article)

    # Step 5: Save results
    print(f'\n=== Results ===')
    print(f'Total candidates: {len(candidates)}')
    print(f'With WPS content: {len(results)}')
    print(f'In date range: {len(in_range)}')

    output = {
        'config': CONFIG,
        'scrapedAt': datetime.now().isoformat(),
        'totalCandidates': len(candidates),
        'totalWithWps': len(results),
        'inRangeCount': len(in_range),
        'articles': in_range,
    }

    with open(CONFIG['outputFile'], 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f'\nSaved to {CONFIG["outputFile"]}')

    # Print year distribution
    year_dist = defaultdict(int)
    for a in in_range:
        d = a.get('date', '')
        if d and len(d) >= 4:
            year_dist[d[:4]] += 1
    print(f'Year distribution: {dict(sorted(year_dist.items()))}')

    # Print some sample titles
    print(f'\nSample articles:')
    for a in in_range[:10]:
        print(f'  {a["date"][:10] if a["date"] else "N/A"} - {a["title"][:70]}')


if __name__ == '__main__':
    main()
