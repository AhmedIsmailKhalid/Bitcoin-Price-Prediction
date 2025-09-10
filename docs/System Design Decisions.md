# System Design Decisions

## Overview
This document captures the core architectural decisions for the Bitcoin Price Prediction Engine, balancing technical impressiveness with development simplicity while prioritizing MLOps and ML Engineering capabilities.

## Design Philosophy

### Core Principles
- **KISS (Keep It Simple, Stupid):** Prioritize simplicity and maintainability
- **Technical Impressiveness with Purpose:** Showcase modern practices without over-engineering
- **Extensibility First:** Design for easy addition of complex features when needed
- **Unit-First Development:** Test each component immediately before integration
- **Laptop-Friendly Development:** Smooth local development experience for rapid iteration

### Hero Priority Ranking
1. **MLOps** (Primary Focus)
2. **AI/ML Engineering** 
3. **Data Science**
4. **Data Engineering**

---

## 1. Overall Architecture Pattern: **HYBRID ARCHITECTURE**

### Selected Approach: Monolith + Strategic Microservices

```
┌──────────────────────────────────────────────────────────┐
│                   Core Application                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│  │   Data      │ │   ML        │ │    MLOps            │ │
│  │  Pipeline   │ │  Pipeline   │ │    Engine           │ │
│  └─────────────┘ └─────────────┘ └─────────────────────┘ │
└──────────────────────────────────────────────────────────┘
                                │
                    ┌────────────────────┐
                    │  Prediction API    │ (Separate Service)
                    │  (ML Engineering)  │ 
                    └────────────────────┘
                                │
                    ┌────────────────────┐
                    │  Frontend/Demo     │ (Separate Service)
                    │  (User Interface)  │
                    └────────────────────┘
```

### Components Breakdown

#### **Core Application (Monolith)**
- **Data Pipeline:** Ingestion, validation, transformation
- **ML Pipeline:** Training, evaluation, model management
- **MLOps Engine:** Experiment tracking, model registry, monitoring

#### **Prediction API (Microservice)**
- **Purpose:** Demonstrate ML Engineering serving patterns
- **Technology:** FastAPI with async capabilities
- **Features:** Real-time predictions, health checks, metrics

#### **Frontend/Demo (Microservice)**
- **Purpose:** User interface and demo capabilities
- **Technology:** Streamlit for rapid development
- **Features:** Real-time dashboard, model performance visualization

### Rationale
- **Development Simplicity:** Core logic in single application for easier debugging
- **Technical Showcase:** Separate services demonstrate microservices understanding
- **MLOps Focus:** Core application emphasizes MLOps and ML Engineering
- **Extensibility:** Easy to break out additional services later if needed

---

## 2. Data Flow and Processing Patterns: **HYBRID PROCESSING**

### Selected Approach: Micro-Batch with Real-Time Feel

```
News Sources → Near Real-time Ingestion → Micro-batches (15min) → Fast Predictions
     │
     └─→ Historical Data → Batch Training (Daily) → Model Updates
```

### Processing Components

#### **Real-Time-ish Ingestion**
- **Frequency:** Every 15 minutes
- **Technology:** Scheduled jobs (Airflow) + caching (Redis)
- **Data Sources:** News APIs, social media, price feeds

#### **Micro-Batch Processing**
- **Batch Size:** 15-minute windows
- **Processing:** Feature engineering, sentiment analysis
- **Output:** Fresh features for prediction

#### **Batch Training**
- **Frequency:** Daily (configurable)
- **Trigger:** Performance degradation or data drift detection
- **Process:** Full model retraining with hyperparameter optimization

### Rationale
- **Balance:** Real-time feel without streaming complexity
- **Practicality:** 15-minute latency acceptable for crypto markets
- **Development:** Easier to test and debug than true streaming
- **Scalability:** Can upgrade to streaming later without architectural changes

---

## 3. Storage Strategy: **DATA LAKE ARCHITECTURE**

### Selected Approach: Medallion Architecture

```
Raw Data (Bronze) → Processed Data (Silver) → Analytics Data (Gold)
     │                      │                        │
   Object Store         Relational DB           Feature Store
```

### Storage Layers

#### **Bronze Layer (Raw Data)**
- **Purpose:** Immutable raw data storage
- **Data Types:** News articles, tweets, price data, API responses
- **Format:** Parquet files with partitioning
- **Retention:** Permanent (compressed after 30 days)

#### **Silver Layer (Processed Data)**
- **Purpose:** Cleaned, validated, structured data
- **Data Types:** Parsed articles, sentiment scores, technical indicators
- **Format:** Database tables with proper schemas
- **Quality:** Validated with Great Expectations

#### **Gold Layer (Analytics Data)**
- **Purpose:** Feature store with engineered features
- **Data Types:** ML-ready features, aggregations, derived metrics
- **Format:** Optimized for fast serving (both online/offline)
- **Versioning:** Feature versioning and lineage tracking

### Technology Selection (Refined in Tech Stack)
- **Tools will be decided in Technology Stack discussion**
- **Requirements:** Laptop-friendly, open-source, production-grade patterns

### Rationale
- **Industry Standard:** Medallion architecture widely adopted
- **Data Quality:** Progressive quality improvement through layers
- **MLOps Focus:** Supports model versioning and feature management
- **Flexibility:** Supports both batch and real-time access patterns

---

## 4. Communication Patterns

### Internal Communication

#### **Within Core Application**
- **Pattern:** Direct function calls and dependency injection
- **Benefits:** Simplicity, performance, easy debugging
- **Testing:** Unit tests for each component

#### **Between Services**
- **Pattern:** HTTP REST APIs with OpenAPI documentation
- **Benefits:** Standard, well-understood, easy to test
- **Implementation:** FastAPI automatic documentation

#### **Async Processing**
- **Pattern:** Message queues for non-blocking operations
- **Technology:** Redis for simplicity (vs. Kafka complexity)
- **Use Cases:** Model training triggers, batch job notifications

### External Integrations

#### **Data Sources**
- **News APIs:** HTTP REST with rate limiting and retry logic
- **Price Feeds:** WebSocket for real-time data (fallback to REST)
- **Social Media:** REST APIs with pagination handling

#### **Model Serving**
- **Pattern:** REST API with JSON payloads
- **Features:** Health checks, metrics endpoints, graceful degradation
- **Scaling:** Horizontal scaling capability (though not implemented initially)

### Rationale
- **Simplicity:** Standard HTTP patterns everyone understands
- **Testing:** Easy to mock and test each integration
- **Monitoring:** Standard HTTP metrics and logging
- **Documentation:** Self-documenting with OpenAPI

---

## 5. Development and Testing Strategy

### Unit-First Development Approach

#### **Development Flow**
1. **Component Design:** Define interface and contracts
2. **Unit Implementation:** Build component with comprehensive tests
3. **Integration Testing:** Test component integration immediately
4. **System Testing:** Full system validation only after units work

#### **Testing Pyramid**
```
        E2E Tests (Few)
    ─────────────────────
   Integration Tests (Some)
 ─────────────────────────────
Unit Tests (Many) + Property Tests
```

#### **Immediate Testing Requirements**
- **Every function:** Unit test before moving on
- **Every API endpoint:** Integration test with mocked dependencies
- **Every data transformation:** Property-based testing for edge cases
- **Every ML component:** Model validation and performance tests

### Rationale
- **Avoid Integration Hell:** Catch issues early at unit level
- **Rapid Development:** Immediate feedback on component functionality
- **Confidence:** Know each piece works before combining
- **Maintainability:** Easier to refactor with comprehensive test coverage

---

## 6. Technology Constraints and Decisions

### Development Environment
- **Primary:** Local development on Alienware m18 (RTX 4090, i9-13980HX, 64GB RAM)
- **Containerization:** Docker for consistent environments
- **Resource Management:** Efficient use of laptop resources

### Deployment Target
- **Primary:** HuggingFace Spaces (CPU-based)
- **Fallback:** Local deployment with Docker Compose
- **Scalability:** Design for GPU when available, CPU when not

### Technology Philosophy
- **Open Source Only:** No proprietary tools or services
- **Production Patterns:** Use industry-standard tools and practices
- **Laptop Friendly:** Must run smoothly on single machine
- **Cloud Ready:** Architecture supports cloud deployment later

---

## 7. Success Criteria

### Technical Success
- **Local Development:** Sub-30-second startup time for development environment
- **Testing:** >90% test coverage with fast test execution (<2 minutes)
- **Performance:** Predictions served in <100ms
- **Reliability:** System recovery from failures without manual intervention

### Portfolio Success
- **MLOps Showcase:** Clear demonstration of modern MLOps practices
- **Engineering Excellence:** Production-quality code and architecture
- **Business Value:** Measurable improvement over baseline approaches
- **Storytelling:** Clear narrative of engineering decisions and trade-offs

### Development Success
- **Rapid Iteration:** Easy to add new features and models
- **Debugging:** Clear logging and error handling for quick issue resolution
- **Documentation:** Self-documenting code and architecture
- **Extensibility:** Easy migration to more complex architectures when needed

---

## Next Steps

1. **Technology Stack Selection:** Choose specific tools for each component
2. **Data Architecture Design:** Detail the data schemas and pipelines  
3. **ML Pipeline Architecture:** Design training and serving workflows
4. **Infrastructure Planning:** Docker, CI/CD, and deployment strategy

---

## Decision Log

| **Decision** | **Options Considered** | **Selected** | **Rationale** |
|--------------|----------------------|-------------|---------------|
| Architecture Pattern | Monolith, Microservices, Hybrid | Hybrid | Balance complexity and showcase |
| Processing Pattern | Batch, Streaming, Hybrid | Hybrid (Micro-batch) | Real-time feel without complexity |
| Storage Strategy | Files, Database, Data Lake | Data Lake (Medallion) | Industry standard, scalable |
| Communication | REST, GraphQL, Message Queues | REST + Redis | Simple, well-understood |
| Testing Strategy | Integration-first, Unit-first | Unit-first | Avoid integration hell |

---

**Key Takeaways:**
- ✅ Hybrid architecture balancing simplicity with technical impressiveness
- ✅ MLOps-focused design with clear service boundaries
- ✅ Laptop-friendly development with production patterns
- ✅ Unit-first testing to avoid integration hell
- ✅ Extensible design for future complexity when needed

---

*This document serves as the foundational architecture blueprint for all subsequent implementation decisions.*

---