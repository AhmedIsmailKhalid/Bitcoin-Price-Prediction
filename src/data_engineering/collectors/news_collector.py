import re
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, List, Any
from sqlalchemy.orm import Session
from urllib.parse import urljoin, urlparse
from .base_collector import BaseCollector

from src.shared.config import settings
from src.shared.models import NewsData


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
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
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
        
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', '', text)
        
        # Strip and limit length
        text = text.strip()
        
        return text
    
    def store_data(self, data: List[Dict[str, Any]], db: Session) -> int:
        """Store news articles in the database"""
        
        stored_count = 0
        
        for article in data:
            try:
                # Check if article already exists (by URL)
                existing = db.query(NewsData).filter(NewsData.url == article['url']).first()
                if existing:
                    self.logger.info(f"Article already exists, skipping: {article['url']}")
                    continue
                
                news_data = NewsData(**article)
                db.add(news_data)
                stored_count += 1
                
            except Exception as e:
                self.logger.error(f"Failed to create NewsData record: {e}")
                self.logger.error(f"Article data: {article}")
                continue
        
        try:
            db.commit()
            self.logger.info(f"Stored {stored_count} news articles")
            return stored_count
            
        except Exception as e:
            db.rollback()
            self.logger.error(f"Failed to commit news data: {e}")
            raise