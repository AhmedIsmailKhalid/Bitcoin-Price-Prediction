import re
import time
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from src.shared.config import settings
from src.shared.models import NewsData

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
            
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
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
        
        if not text:
            return ""
        
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
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
        
        stored_count = 0
        
        for article in data:
            try:
                # Check if article already exists
                existing = db.query(NewsData).filter(NewsData.url == article['url']).first()
                if existing:
                    self.logger.debug(f"Article already exists, skipping: {article['title'][:50]}...")
                    continue
                
                news_data = NewsData(**article)
                db.add(news_data)
                stored_count += 1
                
            except Exception as e:
                self.logger.error(f"Failed to create NewsData record: {e}")
                continue
        
        try:
            db.commit()
            self.logger.info(f"Stored {stored_count} news articles")
            return stored_count
            
        except Exception as e:
            db.rollback()
            self.logger.error(f"Failed to commit news data: {e}")
            raise 