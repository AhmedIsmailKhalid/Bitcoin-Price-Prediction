"""Debug CoinDesk content extraction"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import requests
from bs4 import BeautifulSoup

from src.shared.logging import get_logger, setup_logging


def debug_coindesk_article():
    """Debug content extraction from a specific CoinDesk article"""
    setup_logging()
    logger = get_logger(__name__)
    
    # Use the URL from the logs that failed
    url = "https://www.coindesk.com/markets/2025/09/12/bitcoin-ether-catch-friday-afternoon-bids-rise-to-three-week-highs"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        logger.info(f"Fetching article: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        logger.info("=== Trying different content selectors ===")
        
        # Test different selectors
        selectors = [
            '.at-content p',
            '.entry-content p',
            '.article-content p',
            '.post-content p',
            'article p',
            '.content p',
            '.story-body p',
            '.article-body p',
            'div[data-module="ArticleBody"] p',
            'main p',
            'p'  # Generic fallback
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                content_parts = []
                for element in elements:
                    text = element.get_text(strip=True)
                    if text and len(text) > 20:
                        content_parts.append(text)
                
                if content_parts:
                    full_content = ' '.join(content_parts)
                    logger.info(f"Selector '{selector}': Found {len(content_parts)} paragraphs")
                    logger.info(f"Total content length: {len(full_content)} characters")
                    logger.info(f"Content preview: {full_content[:200]}...")
                    logger.info("="*50)
                    return full_content
            else:
                logger.info(f"Selector '{selector}': No elements found")
        
        # If no selectors work, show page structure
        logger.info("\n=== Page Structure Analysis ===")
        
        # Find all div classes that might contain content
        content_divs = soup.find_all('div', class_=True)
        classes = set()
        for div in content_divs:
            classes.update(div.get('class'))
        
        content_classes = [cls for cls in classes if any(word in cls.lower() for word in ['content', 'article', 'body', 'story', 'text'])]
        logger.info(f"Potential content classes: {content_classes}")
        
        # Find all paragraphs and show their parent elements
        all_paragraphs = soup.find_all('p')
        logger.info(f"Total paragraph tags found: {len(all_paragraphs)}")
        
        if all_paragraphs:
            sample_p = all_paragraphs[0]
            logger.info(f"First paragraph parent structure:")
            parent = sample_p.parent
            while parent and parent.name != 'body':
                logger.info(f"  Parent: {parent.name} | Classes: {parent.get('class', [])}")
                parent = parent.parent
        
        return ""
        
    except Exception as e:
        logger.error(f"Failed to debug CoinDesk article: {e}")
        return ""


if __name__ == "__main__":
    debug_coindesk_article()