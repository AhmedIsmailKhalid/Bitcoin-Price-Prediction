import re
import time
<<<<<<< HEAD
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session
=======
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, List, Any
from sqlalchemy.orm import Session
from urllib.parse import urljoin, urlparse
from .base_collector import BaseCollector
>>>>>>> a105ecb0eae26a545a6776e2f0e0de2712dfc8c3

from src.shared.config import settings
from src.shared.models import NewsData

<<<<<<< HEAD
from .base_collector import BaseCollector


class MultiSourceNewsCollector(BaseCollector):
    """Collector for Bitcoin news from multiple sources"""
    
    def __init__(self):
        super().__init__("MultiSourceNews", "news")
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        self.request_delay = 3  # Increased delay for full article scraping
        
        # News sources configuration
        self.sources = {
            "coindesk": {
                "type": "rss",
                "url": "https://feeds.feedburner.com/CoinDesk",
                "enabled": True,
                "full_content": True  # Scrape full articles
            },
            "cointelegraph": {
                "type": "rss", 
                "url": "https://cointelegraph.com/rss",
                "enabled": True,
                "full_content": True
            },
            "decrypt": {
                "type": "rss",
                "url": "https://decrypt.co/feed", 
                "enabled": True,
                "full_content": True
            }
        }
    
    def collect_data(self) -> List[Dict[str, Any]]:
        """Collect Bitcoin news from multiple sources"""
        
        all_articles = []
        
        for source_name, config in self.sources.items():
            if not config["enabled"]:
                continue
                
            try:
                self.logger.info(f"Collecting from {source_name}...")
                
                if config["type"] == "rss":
                    articles = self._collect_rss(source_name, config)
                else:
                    continue
                
                self.logger.info(f"Collected {len(articles)} articles from {source_name}")
                all_articles.extend(articles)
                
                # Rate limiting between sources
                time.sleep(self.request_delay)
                
            except Exception as e:
                self.logger.error(f"Failed to collect from {source_name}: {e}")
                continue
        
        # Remove duplicates by URL
        unique_articles = self._remove_duplicates(all_articles)
        
        self.logger.info(f"Total unique articles collected: {len(unique_articles)}")
        return unique_articles
    
    def _collect_rss(self, source_name: str, config: Dict) -> List[Dict[str, Any]]:
        """Collect articles from RSS feed"""
        
        articles = []
        
        try:
            feed = feedparser.parse(config["url"])
            
            if feed.bozo:
                self.logger.warning(f"RSS feed {source_name} had parsing issues")
            
            self.logger.info(f"Found {len(feed.entries)} entries in {source_name} RSS feed")
            
            # Limit articles per source
            max_articles = 5  # Reduced since we're doing full scraping
            
            for i, entry in enumerate(feed.entries[:max_articles]):
                try:
                    self.logger.info(f"Processing article {i+1}/{min(len(feed.entries), max_articles)} from {source_name}")
                    
                    article_data = self._process_rss_entry(entry, source_name, config)
                    if article_data:
                        articles.append(article_data)
                    
                    # Rate limiting between articles
                    time.sleep(self.request_delay)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process RSS entry from {source_name}: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Failed to collect RSS from {source_name}: {e}")
            return []
    
    def _process_rss_entry(self, entry, source_name: str, config: Dict) -> Dict[str, Any]:
        """Process RSS entry into article data"""
        
        title = entry.get('title', 'No title')
        url = entry.get('link', '')
        
        if not url:
            self.logger.warning(f"No URL found for article: {title}")
            return None
        
        article_data = {
            'data_source': source_name,
            'title': self._clean_text(title),
            'url': url,
        }
        
        # Get content - either from RSS or scrape full article
        if config.get("full_content", False):
            # Scrape full article content
            full_content = self._scrape_full_article(url, source_name)
            if full_content:
                article_data['content'] = full_content[:5000]  # Limit content length
                article_data['summary'] = full_content[:500]   # First 500 chars as summary
            else:
                # Fallback to RSS summary
                summary = entry.get('summary', '') or entry.get('description', '')
                article_data['content'] = self._clean_text(summary)
                article_data['summary'] = self._clean_text(summary)[:500]
        else:
            # Use RSS summary only
            summary = entry.get('summary', '') or entry.get('description', '')
            article_data['content'] = self._clean_text(summary)
            article_data['summary'] = self._clean_text(summary)[:500]
        
        # Published date
        published_parsed = entry.get('published_parsed')
        if published_parsed:
            article_data['published_at'] = datetime(*published_parsed[:6])
        else:
            article_data['published_at'] = datetime.utcnow()
        
        # Author
        article_data['author'] = entry.get('author', None)
        
        # Word count
        if article_data['content']:
            article_data['word_count'] = len(article_data['content'].split())
        
        # Validation
        if len(article_data['content']) < 50:  # Minimum content length
            self.logger.warning(f"Article content too short, skipping: {title}")
            return None
        
        return article_data
    
    def _scrape_full_article(self, url: str, source_name: str) -> str:
        """Scrape full article content from URL"""
        
        try:
            self.logger.debug(f"Scraping full content from: {url}")
            
=======

class CoinDeskNewsCollector(BaseCollector):
    """Collector for CoinDesk Bitcoin news articles"""
    
    def __init__(self):
        super().__init__("CoinDesk", "news")
        self.base_url = settings.coindesk_base_url
        
        # Request headers to appear as a normal browser
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        
        # Rate limiting
        self.request_delay = 2  # 2 seconds between requests to be respectful
    
    def collect_data(self) -> List[Dict[str, Any]]:
        """Collect Bitcoin news articles from CoinDesk"""
        
        try:
            # Start with CoinDesk Bitcoin news section
            bitcoin_section_url = f"{self.base_url}/tag/bitcoin/"
            
            self.logger.info(f"Collecting articles from: {bitcoin_section_url}")
            
            # Get the main Bitcoin news page
            response = requests.get(bitcoin_section_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links on the page
            article_links = self._extract_article_links(soup)
            
            self.logger.info(f"Found {len(article_links)} article links")
            
            # Collect articles (limit to recent articles to avoid overwhelming)
            articles = []
            max_articles = 10  # Limit for testing
            
            for i, article_url in enumerate(article_links[:max_articles]):
                try:
                    self.logger.info(f"Collecting article {i+1}/{min(len(article_links), max_articles)}: {article_url}")
                    
                    article_data = self._scrape_article(article_url)
                    if article_data:
                        articles.append(article_data)
                    
                    # Be respectful with rate limiting
                    time.sleep(self.request_delay)
                    
                except Exception as e:
                    self.logger.error(f"Failed to scrape article {article_url}: {e}")
                    continue
            
            self.logger.info(f"Successfully collected {len(articles)} articles")
            return articles
            
        except Exception as e:
            self.logger.error(f"Failed to collect news data: {e}")
            raise
    
    def _extract_article_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract article URLs from the Bitcoin news page"""
        
        article_links = []
        
        # CoinDesk typically uses these selectors for article links
        # This may need adjustment based on current CoinDesk structure
        selectors = [
            'h3 a[href]',  # Headlines in h3 tags
            'h2 a[href]',  # Headlines in h2 tags
            '.headline a[href]',  # Headlines with class
            'article a[href]',  # Links within article tags
        ]
        
        for selector in selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href:
                    # Convert relative URLs to absolute
                    full_url = urljoin(self.base_url, href)
                    
                    # Filter for actual article URLs (not ads, external links, etc.)
                    if self._is_valid_article_url(full_url):
                        article_links.append(full_url)
 
        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in article_links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
        
        return unique_links
    
    def _is_valid_article_url(self, url: str) -> bool:
        """Check if URL is a valid CoinDesk article"""
        
        parsed = urlparse(url)
        
        # Must be a CoinDesk URL
        if 'coindesk.com' not in parsed.netloc:
            return False
        
        # Skip certain URL patterns
        skip_patterns = [
            '/tag/',
            '/author/',
            '/category/',
            '/page/',
            '/wp-content/',
            '/wp-admin/',
            '#',
            'mailto:',
            'javascript:',
        ]
        
        for pattern in skip_patterns:
            if pattern in url:
                return False
        
        # Should look like an article URL
        # CoinDesk articles typically have year/month in URL
        if re.search(r'/\d{4}/\d{2}/', url):
            return True
        
        # Or have certain article indicators
        article_indicators = [
            '/news/',
            '/markets/',
            '/policy/',
            '/tech/',
        ]
        
        for indicator in article_indicators:
            if indicator in url:
                return True
        
        return False
    
    def _scrape_article(self, url: str) -> Dict[str, Any]:
        """Scrape individual article content"""
        
        try:
>>>>>>> a105ecb0eae26a545a6776e2f0e0de2712dfc8c3
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
<<<<<<< HEAD
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            content_parts = []
            
            # Extract content based on source
            if 'cointelegraph' in source_name:
                content_parts = self._extract_cointelegraph_content(soup)
            elif 'decrypt' in source_name:
                content_parts = self._extract_decrypt_content(soup)
            elif 'coindesk' in source_name:
                content_parts = self._extract_coindesk_content(soup)
            else:
                # Generic extraction
                content_parts = self._extract_generic_content(soup)
            
            if content_parts:
                full_content = ' '.join(content_parts)
                return self._clean_text(full_content)
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Failed to scrape full article from {url}: {e}")
            return ""
    
    def _extract_cointelegraph_content(self, soup: BeautifulSoup) -> List[str]:
        """Extract content specifically for Cointelegraph articles"""
        
        content_parts = []
        
        # Try multiple selectors for Cointelegraph
        selectors = [
            '.post-content p',
            '.article-content p', 
            '.entry-content p',
            'article p',
            '.post-content h2',
            '.article-content h2'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    text = element.get_text(strip=True)
                    if text and len(text) > 20:  # Skip very short paragraphs
                        content_parts.append(text)
                break  # Use first working selector
        
        return content_parts
    
    def _extract_decrypt_content(self, soup: BeautifulSoup) -> List[str]:
        """Extract content specifically for Decrypt articles"""
        
        content_parts = []
        
        # Try multiple selectors for Decrypt
        selectors = [
            '.entry-content p',
            '.article-body p',
            '.post-content p',
            'article p',
            '.entry-content h2',
            '.article-body h2'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    text = element.get_text(strip=True)
                    if text and len(text) > 20:
                        content_parts.append(text)
                break
        
        return content_parts
    
    def _extract_coindesk_content(self, soup: BeautifulSoup) -> List[str]:
        """Extract content specifically for CoinDesk articles"""
        
        content_parts = []
        
        # CoinDesk specific selectors - updated with working selector
        selectors = [
            'main p',           # This works!
            '.at-content p',    # Backup selectors
            '.entry-content p',
            '.article-content p',
            '.post-content p',
            'article p'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    text = element.get_text(strip=True)
                    if text and len(text) > 20:
                        content_parts.append(text)
                break  # Use first working selector
        
        return content_parts
    
    def _extract_generic_content(self, soup: BeautifulSoup) -> List[str]:
        """Generic content extraction for unknown sites"""
        
        content_parts = []
        
        # Generic selectors
        for tag in ['p', 'h2', 'h3']:
            elements = soup.find_all(tag)
            for element in elements:
                text = element.get_text(strip=True)
                if text and len(text) > 20:
                    content_parts.append(text)
        
        return content_parts
    
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
=======
            # Extract article data
            article_data = {
                'url': url,
                'data_source': 'coindesk',
            }
            
            # Title
            title_selectors = ['h1', '.headline', '.entry-title', 'title']
            title = self._extract_text_by_selectors(soup, title_selectors)
            article_data['title'] = self._clean_text(title) if title else "No title found"
            
            # Content
            content_selectors = [
                '.at-content',  # Common CoinDesk content class
                '.entry-content',
                '.article-content',
                '.post-content',
                'article',
                '.content',
            ]
            content = self._extract_text_by_selectors(soup, content_selectors)
            article_data['content'] = self._clean_text(content) if content else ""
            
            # Summary (first paragraph or meta description)
            summary_selectors = [
                '.excerpt',
                '.summary',
                '.lead',
                'p',  # First paragraph
            ]
            summary = self._extract_text_by_selectors(soup, summary_selectors)
            if summary:
                # Take first 500 characters as summary
                article_data['summary'] = self._clean_text(summary)[:500]
            
            # Author
            author_selectors = [
                '.author-name',
                '.byline',
                '.author',
                '[rel="author"]',
            ]
            author = self._extract_text_by_selectors(soup, author_selectors)
            article_data['author'] = self._clean_text(author) if author else None
            
            # Published date (this is tricky and may need adjustment)
            date_selectors = [
                'time[datetime]',
                '.published',
                '.date',
                '.timestamp',
            ]
            published_date = self._extract_date(soup, date_selectors)
            article_data['published_at'] = published_date
            
            # Word count
            if article_data['content']:
                article_data['word_count'] = len(article_data['content'].split())
            
            # Basic validation
            if len(article_data['content']) < 100:  # Too short to be a real article
                self.logger.warning(f"Article too short, skipping: {url}")
                return None
            
            return article_data
            
        except Exception as e:
            self.logger.error(f"Failed to scrape article {url}: {e}")
            return None
    
    def _extract_text_by_selectors(self, soup: BeautifulSoup, selectors: List[str]) -> str:
        """Try multiple CSS selectors to extract text"""
        
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                if text and len(text) > 10:  # Found meaningful text
                    return text
        
        return ""
    
    def _extract_date(self, soup: BeautifulSoup, selectors: List[str]) -> datetime:
        """Extract publication date from article"""
        
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements:
                # Try datetime attribute first
                datetime_attr = element.get('datetime')
                if datetime_attr:
                    try:
                        return datetime.fromisoformat(datetime_attr.replace('Z', '+00:00'))
                    except:
                        continue
                
                # Try text content
                text = element.get_text(strip=True)
                if text:
                    # This is basic - could be enhanced with better date parsing
                    try:
                        return datetime.strptime(text, '%Y-%m-%d')
                    except:
                        continue
        
        # Default to current time if no date found
        return datetime.utcnow()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
>>>>>>> a105ecb0eae26a545a6776e2f0e0de2712dfc8c3
        
        if not text:
            return ""
        
<<<<<<< HEAD
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
=======
>>>>>>> a105ecb0eae26a545a6776e2f0e0de2712dfc8c3
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
<<<<<<< HEAD
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)]', '', text)
        
        return text.strip()
    
    def _remove_duplicates(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles by URL"""
        
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            url = article.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        
        removed_count = len(articles) - len(unique_articles)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} duplicate articles")
        
        return unique_articles
    
    def store_data(self, data: List[Dict[str, Any]], db: Session) -> int:
        """Store news articles in database"""
=======
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', '', text)
        
        # Strip and limit length
        text = text.strip()
        
        return text
    
    def store_data(self, data: List[Dict[str, Any]], db: Session) -> int:
        """Store news articles in the database"""
>>>>>>> a105ecb0eae26a545a6776e2f0e0de2712dfc8c3
        
        stored_count = 0
        
        for article in data:
            try:
<<<<<<< HEAD
                # Check if article already exists
                existing = db.query(NewsData).filter(NewsData.url == article['url']).first()
                if existing:
                    self.logger.debug(f"Article already exists, skipping: {article['title'][:50]}...")
=======
                # Check if article already exists (by URL)
                existing = db.query(NewsData).filter(NewsData.url == article['url']).first()
                if existing:
                    self.logger.info(f"Article already exists, skipping: {article['url']}")
>>>>>>> a105ecb0eae26a545a6776e2f0e0de2712dfc8c3
                    continue
                
                news_data = NewsData(**article)
                db.add(news_data)
                stored_count += 1
                
            except Exception as e:
                self.logger.error(f"Failed to create NewsData record: {e}")
<<<<<<< HEAD
=======
                self.logger.error(f"Article data: {article}")
>>>>>>> a105ecb0eae26a545a6776e2f0e0de2712dfc8c3
                continue
        
        try:
            db.commit()
            self.logger.info(f"Stored {stored_count} news articles")
            return stored_count
            
        except Exception as e:
            db.rollback()
            self.logger.error(f"Failed to commit news data: {e}")
<<<<<<< HEAD
            raise 
=======
            raise
>>>>>>> a105ecb0eae26a545a6776e2f0e0de2712dfc8c3
