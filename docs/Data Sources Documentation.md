# Data Sources Documentation

## Overview
This document defines all data sources that will be used for the Bitcoin Price Prediction Engine, including collection strategies, formats, and integration approaches.

---

## Selected Data Sources

### **1. Price Data Sources**

#### **Primary: CoinGecko API**
- **Access Method:** Free API (50 calls/minute limit)
- **Data Format:** OHLCV (Open, High, Low, Close, Volume)
- **Collection Frequency:** 15-minute micro-batch processing
- **Data Intervals:** 1-hour price intervals
- **Historical Coverage:** Up to 1 year of hourly data (free tier)
- **Reliability:** Very high uptime, widely adopted in industry
- **API Endpoint:** `https://api.coingecko.com/api/v3/coins/bitcoin/ohlc`

#### **Backup: CryptoCompare API**
- **Access Method:** Free tier (100,000 calls/month)
- **Data Format:** OHLCV data
- **Collection Frequency:** Same as primary (15-minute micro-batch)
- **Historical Coverage:** Good historical data availability
- **Failover Logic:** Automatic fallback if CoinGecko fails
- **API Endpoint:** `https://min-api.cryptocompare.com/data/v2/histohour`

#### **Data Requirements Justification:**
Based on our 15 selected features, we need full OHLCV data:
- **Technical Indicators:** `sma_20`, `rsi`, `atr` require high/low/volume data
- **Price Returns:** `returns_1h`, `returns_4h`, `returns_24h` need close prices
- **Volume Analysis:** `volume_ratio` requires volume data
- **Volatility Regime:** `volatility_regime` needs close prices for calculation

### **2. News Data Sources**

#### **Primary: CoinDesk (Web Scraping)**
- **Access Method:** Web scraping (no API required)
- **Content Volume:** 10-15 articles per day
- **Content Quality:** Professional journalism, consistent format
- **Collection Frequency:** 15-minute micro-batch processing
- **HTML Structure:** Predictable structure, easy to parse
- **Reliability:** High uptime, established news source
- **Target URL Pattern:** `https://www.coindesk.com/news/`

#### **Secondary: CryptoSlate (Web Scraping)**
- **Access Method:** Web scraping
- **Content Volume:** 8-12 articles per day
- **Content Quality:** Good crypto news coverage
- **Collection Frequency:** Same as primary source
- **HTML Structure:** Clean structure, extractable content
- **Purpose:** Backup source and additional news volume

#### **Tertiary: Bitcoin Magazine (Web Scraping)**
- **Access Method:** Web scraping
- **Content Volume:** 5-8 articles per day
- **Content Focus:** Bitcoin-specific news and analysis
- **Purpose:** Backup source for Bitcoin-focused content
- **Target URL Pattern:** `https://bitcoinmagazine.com/`

#### **Unified News Data Schema:**
```json
{
  "title": "string",
  "content": "string", 
  "published_at": "datetime",
  "source": "string",
  "url": "string",
  "sentiment_score": "float"
}
```

### **3. Social Media Data Sources**

#### **Primary: Reddit API**
- **Access Method:** Reddit API via PRAW (Python Reddit API Wrapper)
- **Rate Limits:** 60 calls/minute (free tier)
- **Target Subreddits:** 
  - `r/Bitcoin` (high-quality Bitcoin discussions)
  - `r/CryptoCurrency` (broader crypto community)
  - `r/BitcoinMarkets` (trading-focused discussions)
- **Collection Frequency:** 15-minute micro-batch processing
- **Data Points:** Post titles, content, upvotes, comments, timestamps
- **Content Volume:** 50-100 posts/comments per day
- **Authentication:** OAuth2 with Reddit API credentials

#### **Alternative Sources Considered:**
- **Twitter/X:** Eliminated due to API complexity and cost
- **Telegram:** Potential future addition if needed
- **YouTube:** Metadata only, limited sentiment value

#### **Reddit Data Schema:**
```json
{
  "post_id": "string",
  "title": "string",
  "content": "string",
  "subreddit": "string",
  "upvotes": "integer",
  "comments_count": "integer",
  "created_at": "datetime",
  "social_sentiment": "float"
}
```

---

## Data Collection Strategy

### **Collection Approach: Micro-Batch Processing**

#### **Selected Strategy: 15-Minute Micro-Batch**
**Rationale:**
- **Near real-time feel** without streaming infrastructure complexity
- **Reliable and testable** similar to traditional batch processing
- **Production-ready pattern** commonly used in industry
- **Feature compatibility** works well with 1-hour price return features
- **Resource efficient** balanced between real-time and batch approaches

#### **Alternative Approaches Considered:**

| **Approach** | **Pros** | **Cons** | **Decision** |
|--------------|----------|----------|--------------|
| **Real-time Streaming** | Immediate data, impressive demos | Complex infrastructure, harder debugging | ❌ Too complex for scope |
| **Hourly Batch** | Simple, reliable | Too slow for engagement | ❌ Not responsive enough |
| **15-min Micro-batch** | Near real-time, manageable complexity | Slight data lag | ✅ **Selected** |
| **Daily Batch** | Very simple | Too slow, poor user experience | ❌ Insufficient frequency |

### **Collection Schedule:**
- **Price Data:** Every 15 minutes
- **News Articles:** Every 15 minutes  
- **Social Media:** Every 15 minutes
- **Pipeline Processing:** Triggered after each collection cycle

---

## Historical Data Requirements

### **Performance Target Analysis:**
- **Target Accuracy:** >90% directional accuracy
- **Baseline:** Random prediction ≈ 50%
- **Required Improvement:** 40+ percentage points

### **Historical Data Collection Plan:**

#### **Training Data Requirements:**
- **Duration:** 6-8 months of historical data
- **Reasoning:** Sufficient to capture multiple market cycles and conditions
- **Market Coverage:** Include both bull and bear market periods for robustness

#### **Historical Data Sources:**
- **Price Data:** CoinGecko provides 1 year of hourly historical data (free)
- **News Data:** Web scraping of historical articles (6+ months backfill)
- **Social Data:** Reddit historical posts (limited by API, focus on recent data)

#### **Data Collection Timeline:**
1. **Phase 1:** Backfill 6 months of price data from CoinGecko
2. **Phase 2:** Scrape 6 months of historical news articles
3. **Phase 3:** Collect recent Reddit data (Reddit limits historical access)
4. **Phase 4:** Begin real-time collection pipeline

---

## Backup Sources & Reliability Strategy

### **Backup Source Implementation:**

#### **Simple Primary + Backup Strategy**
**Primary Sources:**
- Price: CoinGecko API
- News: CoinDesk web scraping  
- Social: Reddit API

**Backup Sources:**
- Price: CryptoCompare API
- News: CryptoSlate web scraping
- Social: None needed (Reddit reliability is high)

#### **Failover Logic:**
```python
# Example failover implementation (boilerplate)
def collect_price_data():
    try:
        return coingecko_api.get_price_data()
    except Exception as e:
        logger.warning(f"CoinGecko failed: {e}, trying backup")
        return cryptocompare_api.get_price_data()
```

#### **Benefits vs Complexity Analysis:**

| **Aspect** | **Backup Sources** | **Single Source** |
|------------|-------------------|-------------------|
| **Reliability** | Higher system uptime | Dependent on single source |
| **Complexity** | Moderate (failover logic) | Lower implementation |
| **Testing** | Need failure scenario tests | Simpler testing |
| **Monitoring** | Track multiple source health | Single source monitoring |
| **Portfolio Value** | Shows production thinking | Focus on core features |

**Decision:** Implement simple backup strategy for production readiness demonstration

---

## Free Tier Usage Strategy

### **API Rate Limits & Usage:**

#### **CoinGecko API:**
- **Limit:** 50 calls/minute = 72,000 calls/day
- **Our Usage:** ~96 calls/day (every 15 minutes)
- **Utilization:** ~0.13% of available capacity
- **Buffer:** Extremely safe margin for bursts and retries

#### **Reddit API:**
- **Limit:** 60 calls/minute = 86,400 calls/day  
- **Our Usage:** ~100 calls/day
- **Utilization:** ~0.12% of available capacity
- **Buffer:** Large margin for comment thread exploration

#### **Web Scraping:**
- **Limit:** No API limits (respectful scraping)
- **Strategy:** 1 request per 2-3 seconds to avoid overwhelming servers
- **Monitoring:** Track response times and HTTP status codes

### **Cost Analysis:**
- **Total Monthly Cost:** $0 (all free sources)
- **Scalability:** Can handle 10x current usage within free tiers
- **Risk Mitigation:** Multiple free sources reduce dependency risk

---

## Expected Data Volume & Storage

### **Daily Data Volume:**
- **Price Data:** ~24 OHLCV records per day (hourly intervals)
- **News Articles:** ~15-20 articles per day (combined sources)
- **Social Posts:** ~50-100 Reddit posts/comments per day
- **Total Daily Storage:** ~1MB compressed

### **Annual Projections:**
- **Raw Data Storage:** ~365MB per year
- **Processed Features:** ~100MB per year (after feature engineering)
- **Model Artifacts:** ~50MB per year (model versions and experiments)
- **Total Storage:** ~515MB per year

### **Storage Architecture Impact:**
- **Bronze Layer (MinIO):** Raw data storage, ~365MB/year
- **Silver Layer (PostgreSQL):** Processed data, fits well in NeonDB free tier (512MB)
- **Gold Layer (Redis + PostgreSQL):** Feature store, minimal additional storage

---

## Integration with Feature Engineering

### **Feature Mapping:**

#### **Price Features (8 features):**
- **Data Source:** CoinGecko/CryptoCompare OHLCV
- **Required Fields:** Open, High, Low, Close, Volume
- **Processing:** TA-Lib technical indicators + custom calculations

#### **Sentiment Features (4 features):**
- **Data Source:** CoinDesk/CryptoSlate articles
- **Processing:** CryptoBERT sentiment analysis
- **Aggregation:** Hourly sentiment mean, count, change, extremes

#### **Social Features (2 features):**
- **Data Source:** Reddit posts and comments
- **Processing:** Volume counting, engagement analysis
- **Metrics:** Social volume, social sentiment

#### **Time Features (1 feature):**
- **Data Source:** Timestamp from all sources
- **Processing:** Hour of day extraction

---

## MLOps Integration Points

### **Data Drift Detection:**
- **News Content:** Compare new article content distributions vs historical
- **Social Sentiment:** Monitor Reddit sentiment distribution changes
- **Price Patterns:** Detect shifts in price volatility and trading patterns

### **Retraining Triggers:**
- **Volume-based:** Trigger retraining when accumulated new articles reach threshold
- **Performance-based:** Retrain when prediction accuracy degrades
- **Schedule-based:** Weekly retraining with fresh data

### **Data Quality Monitoring:**
- **API Health:** Monitor response times and failure rates
- **Content Quality:** Validate article length, sentiment score ranges
- **Data Freshness:** Alert on stale data or collection failures

---

## Implementation Impact

### **Data Engineering Pipeline:**
- **Ingestion:** 15-minute micro-batch from multiple sources
- **Transformation:** Unified schema conversion and validation
- **Storage:** Multi-layer architecture (Bronze → Silver → Gold)

### **Development Priorities:**
1. **Price data collection** (foundational for all features)
2. **News article scraping** (core sentiment features)  
3. **Reddit integration** (social features)
4. **Historical data backfill** (training dataset)

### **Infrastructure Requirements:**
- **Compute:** Minimal (web scraping + API calls)
- **Storage:** ~1GB total for full system
- **Network:** Standard HTTP requests, no special requirements
- **Monitoring:** API health checks and data quality validation

---

**This data source plan provides the foundation for all subsequent implementation planning and ensures we have reliable, free, and comprehensive data for our ML system.**