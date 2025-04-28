import nltk
import argparse
import csv
import hashlib
import logging
import time
from datetime import datetime
import feedparser
import dateparser
import numpy as np
import pytz
import requests
import requests_cache
from bs4 import BeautifulSoup
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from newspaper import Article
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('punkt_tab')

class NewsAnalyzer:
    def __init__(self):
        self.setup_logging()
        self.setup_cache()
        self.sources = self.initialize_sources()
        self.analyzer = SentimentIntensityAnalyzer()

    def setup_logging(self):
        """Configure colored logging output."""

        class YellowFormatter(logging.Formatter):
            YELLOW = "\033[93m"
            RESET = "\033[0m"

            def format(self, record):
                return f"{self.YELLOW}{super().format(record)}{self.RESET}"

        handler = logging.StreamHandler()
        handler.setFormatter(YellowFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    def setup_cache(self):
        """Enable HTTP caching to reduce requests."""
        requests_cache.install_cache('news_cache', expire_after=1800)

    def initialize_sources(self):
        """Define news sources and their configurations."""
        return {
            'cnn': {
                'page_url': 'https://edition.cnn.com/world',
                'base_url': 'https://edition.cnn.com',
                'filter': lambda l: l.startswith('/world'),
                'timezone': 'UTC'
            },
            'gulfnews': {
                'page_url': 'https://gulfnews.com',
                'base_url': 'https://gulfnews.com',
                'filter': lambda l: any(p in l for p in ['/business/', '/world/', '/uae/', '/life/']),
                'timezone': 'Asia/Dubai'
            },
            'khaleejtimes': {
                'page_url': 'https://www.khaleejtimes.com',
                'base_url': 'https://www.khaleejtimes.com',
                'filter': lambda l: l.startswith('/uae/') or l.startswith('/world/') or
                                  l.startswith('/business/') or l.startswith('/sports/') or
                                  l.startswith('/entertainment/'),
                'timezone': 'Asia/Dubai'
            },
            'bbc': {
                'page_url': 'https://www.bbc.com/news',
                'base_url': 'https://www.bbc.com',
                'filter': lambda l: '/news/' in l and not l.endswith('.live'),
                'timezone': 'Europe/London'
            },
            'nytimes': {
                'page_url': 'https://rss.nytimes.com/services/xml/rss/nyt/World.xml',
                'base_url': '',
                'filter': lambda l: l.endswith('.html'),
                'timezone': 'America/New_York',
                'rss': True
            }
        }

    def extract_links(self, soup, base, flt):
        """Extract relevant links from HTML content."""
        out = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            if not flt(href):
                continue
            url = href if href.startswith('http') else base + href
            out.add(url)
        return out

    def get_recent_news(self, src_key):
        """Fetch recent news links from a source."""
        cfg = self.sources[src_key]

        try:
            # Handle RSS feeds
            if cfg.get('rss'):
                feed = feedparser.parse(cfg['page_url'])
                links = {
                    entry.link for entry in feed.entries
                    if cfg['filter'](entry.link)
                }
                logging.info(f"Found {len(links)} RSS links from {src_key}.")
                return links

            # Handle HTML scraping
            r = requests.get(cfg['page_url'],
                           headers={'User-Agent': 'Mozilla/5.0'},
                           timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')
            links = self.extract_links(soup, cfg['base_url'], cfg['filter'])
            logging.info(f"Found {len(links)} links on {src_key}.")
            return links
        except Exception as e:
            logging.error(f"Fetch fail {src_key}: {e}")
            return set()

    def parse_date(self, ds, tz):
        """Parse date string with timezone handling."""
        if not ds:
            return datetime.now(pytz.timezone(tz))
        dt = dateparser.parse(ds)
        if dt and not dt.tzinfo:
            return pytz.timezone(tz).localize(dt)
        return dt or datetime.now(pytz.timezone(tz))

    def extract_gulfnews_date(self, soup):
        """Special handling for Gulf News date extraction."""
        time_tag = soup.find('time')
        if time_tag and time_tag.has_attr('datetime'):
            return time_tag['datetime']
        meta_date = soup.find('meta', attrs={'property': 'article:published_time'})
        if meta_date and meta_date.get('content'):
            return meta_date['content']
        return None

    def analyze_article(self, url, src_key):
        """Analyze a single news article."""
        cfg = self.sources[src_key]
        out = {
            'source': src_key,
            'url': url,
            'title': None,
            'authors': None,
            'publish_date': None,
            'summary': None,
            'sentiment_score': None,
            'sentiment_label': None,
            'language': None,
            'keywords': None,
            'error': None
        }

        try:
            art = Article(url)
            art.download()
            art.parse()
            out['title'] = art.title
            out['authors'] = ", ".join(art.authors or ["Unknown"])

            if not art.publish_date and src_key == 'gulfnews':
                soup = BeautifulSoup(art.html, 'html.parser')
                raw_date = self.extract_gulfnews_date(soup)
                out['publish_date'] = self.parse_date(raw_date, cfg['timezone'])
            else:
                out['publish_date'] = self.parse_date(str(art.publish_date), cfg['timezone'])

            art.nlp()
            out['summary'] = art.summary

            try:
                out['language'] = detect(art.text) if art.text else 'unknown'
            except LangDetectException:
                out['language'] = 'unknown'

            vs = self.analyzer.polarity_scores(art.text or "")
            c = vs['compound']
            out['sentiment_score'] = c
            out['sentiment_label'] = (
                'Positive' if c >= 0.05 else
                'Negative' if c <= -0.05 else
                'Neutral'
            )
            out['keywords'] = list(TextBlob(art.text or "").noun_phrases[:10])
            out['article_id'] = hashlib.md5(url.encode()).hexdigest()[:8]

            logging.info(f"Analyzed: {out['title']} ({out['sentiment_label']})")
        except Exception as e:
            out['error'] = str(e)
            logging.error(f"Error @ {url}: {e}")
        return out

    def analyze_articles(self, links, src_key):
        """Analyze multiple articles from a source."""
        return [self.analyze_article(url, src_key) for url in sorted(links)]

    def export_to_csv(self, results, fname):
        """Export analysis results to CSV file."""
        keys = ['source', 'url', 'title', 'authors', 'publish_date', 'language',
                'sentiment_score', 'sentiment_label', 'keywords', 'summary', 'error']
        with open(fname, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in results:
                row = {k: r[k] for k in keys}
                row['keywords'] = ';'.join(row['keywords'] or [])
                row['publish_date'] = r['publish_date'].isoformat() if r['publish_date'] else ''
                w.writerow(row)
        logging.info(f"Wrote articles → {fname}")

    def group_articles_by_topic(self, results, n_clusters=6):
        """Cluster articles by topic using enhanced ML approach."""
        valid = [r for r in results if r.get('summary') and r.get('title')]
        if not valid:
            return []

        texts = [f"{r['title']}. {r['summary']}" for r in valid]

        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=2
        )

        try:
            X = vectorizer.fit_transform(texts)
            n_clusters = min(n_clusters, max(1, len(valid) // 3))

            if n_clusters > 1:
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                model.fit(X)
                labels = model.predict(X)
            else:
                labels = [0] * len(valid)

            clusters = {}
            for idx, lab in enumerate(labels):
                clusters.setdefault(lab, []).append(idx)

            feature_names = np.array(vectorizer.get_feature_names_out())
            grouped = []

            for lab, indices in clusters.items():
                if len(indices) == 0:
                    continue

                centroid = X[indices].mean(axis=0)
                top_idxs = np.argsort(np.asarray(centroid).flatten())[::-1][:5]
                keywords = [feature_names[i] for i in top_idxs if i < len(feature_names)]

                filtered_keywords = [
                    kw for kw in keywords
                    if not kw.lower() in ['say', 'said', 'year', 'new', 'time']
                ]

                main_topic = filtered_keywords[0].replace('_', ' ').title() if filtered_keywords else "General News"
                related_keywords = list(set(filtered_keywords[1:])) if len(filtered_keywords) > 1 else []

                cluster_keywords = set()
                for i in indices:
                    if valid[i].get('keywords'):
                        cluster_keywords.update(
                            kw.lower() for kw in valid[i]['keywords']
                            if len(kw.split()) < 3
                        )

                all_keywords = list(set(filtered_keywords) | cluster_keywords)
                keyword_line = ", ".join(k.replace('_', ' ').title() for k in all_keywords[:8])

                grouped.append((
                    main_topic,
                    keyword_line,
                    len(indices),
                    [valid[i] for i in indices]
                ))

            return grouped

        except Exception as e:
            logging.error(f"Topic clustering failed: {e}")
            return []

    def export_single_article(self, f, art):
        """Write a single article to file with timestamp."""
        f.write(f"➤ {art['title'].strip()}\n")

        # Source and date
        source_line = f"📌 Source: {art['source'].upper()}"
        if art['publish_date']:
            source_line += f" | 📅 {art['publish_date'].strftime('%Y-%m-%d %H:%M')}"
        f.write(source_line + "\n")

        # Authors if available
        if art['authors'] and art['authors'] != "Unknown":
            f.write(f"✍️ Author(s): {art['authors']}\n")

        # Sentiment
        sentiment_emoji = {
            'Positive': '😊',
            'Negative': '😠',
            'Neutral': '😐'
        }.get(art.get('sentiment_label', 'Unknown'), '')
        f.write(f"{sentiment_emoji} Sentiment: {art.get('sentiment_label', 'Unknown')}\n")

        # Summary
        summary = art.get('summary', 'No summary available.').strip()
        if summary:
            if not summary.endswith(('.', '!', '?')):
                summary += '.'
            f.write(f"📝 {summary}\n")

        f.write(f"🔗 {art['url']}\n")
        f.write("-" * 80 + "\n\n")

    def export_to_txt(self, results, fname, is_full_update=False):
        """Export news to text file with different formats based on update type."""
        with open(fname, 'a' if not is_full_update else 'w', encoding='utf-8') as f:
            if is_full_update:
                grouped = self.group_articles_by_topic(results)

                if not grouped:
                    f.write("📰 Latest News Updates\n")
                    f.write("=" * 80 + "\n\n")
                    for art in sorted(results, key=lambda x: x['publish_date'] or datetime.min, reverse=True):
                        self.export_single_article(f, art)
                    return

                grouped.sort(key=lambda x: x[2], reverse=True)

                for main_topic, keywords, hype_count, articles in grouped:
                    hype_level = (
                        "🔥 Very High (Breaking news)" if hype_count >= 10 else
                        "🔴 High (Trending)" if hype_count >= 5 else
                        "🟡 Medium (Developing)" if hype_count >= 3 else
                        "🔵 Low (Coverage)"
                    )

                    f.write(f"\n📰 News on {main_topic}\n")
                    f.write(f"🏷️ Related Keywords: {keywords}\n")
                    f.write(f"📊 Coverage: {hype_level} ({hype_count} articles)\n")
                    f.write("=" * 80 + "\n\n")

                    articles_sorted = sorted(
                        articles,
                        key=lambda x: x['publish_date'] if x['publish_date'] else datetime.min,
                        reverse=True
                    )

                    for art in articles_sorted:
                        self.export_single_article(f, art)
            else:
                # Single article update
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"\n🔄 Recently updated as of {now}\n")
                f.write("=" * 80 + "\n\n")
                for art in results:
                    self.export_single_article(f, art)
                f.write("\n")  # Extra space before next update

        logging.info(f"Updated news file: {fname}")

    def run(self, sources, export_file, clusters=6):
        """Main execution loop with new article buffering."""
        seen = set()
        csv_file = export_file
        txt_file = csv_file.replace('.csv', '_organized.txt')
        article_buffer = []

        # Clear the text file at start
        open(txt_file, 'w').close()

        while True:
            new_results = []
            for src in sources:
                links = self.get_recent_news(src)
                new_links = links - seen
                seen.update(new_links)

                if not new_links:
                    logging.info(f"No new links from {src}.")
                    continue

                logging.info(f"Analyzing {len(new_links)} new articles from {src}...")
                new_results.extend(self.analyze_articles(new_links, src))

            if new_results:
                # Always update CSV
                self.export_to_csv(new_results, csv_file)

                # Add to buffer and check if we should do full update
                article_buffer.extend(new_results)

                if len(article_buffer) >= 5:
                    # Do full organized update
                    self.export_to_txt(article_buffer, txt_file, is_full_update=True)
                    article_buffer = []  # Clear buffer after full update
                else:
                    # Just append the new articles with timestamp
                    self.export_to_txt(new_results, txt_file, is_full_update=False)

            logging.info("Sleeping 60 seconds before next fetch...")
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="News Analyzer & Smart Topic Clusterer")
    parser.add_argument('--sources', nargs='+',
                        choices=['cnn', 'gulfnews', 'khaleejtimes', 'bbc', 'nytimes'],
                        default=['cnn', 'bbc', 'nytimes'], #add more news channels here
                        help='Select which sources to fetch news from')

    parser.add_argument('--export', required=True,
                        help='Base CSV filename, e.g. news.csv')
    parser.add_argument('--clusters', type=int, default=6,
                        help='Number of topic clusters to create')
    args = parser.parse_args()

    analyzer = NewsAnalyzer()
    analyzer.run(args.sources, args.export, args.clusters)


if __name__ == '__main__':
    main()
