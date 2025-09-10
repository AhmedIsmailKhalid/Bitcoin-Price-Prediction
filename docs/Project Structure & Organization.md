# Project Structure & Organization (Revised)

## Core Philosophy
- **Critical files only** - every file serves a specific, essential purpose
- **Clean organization** - logical grouping without unnecessary nesting
- **Technology stack alignment** - structure directly supports our chosen technologies
- **Multi-platform deployment ready** - organized for Vercel + Render deployment
- **Development workflow** - optimized for efficient development and testing

---

## Complete Project Structure

```
bitcoin-prediction-engine/
├── README.md                           # Project overview and setup
├── .gitignore                          # Git ignore patterns
├── .env.example                        # Environment variables template
├── docker-compose.yml                  # Local development orchestration
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Poetry configuration
├── Dockerfile                          # Backend development container
│
├── src/                                # All Python/ML source code
│   ├── __init__.py
│   ├── data_engineering/               # Data pipelines and processing
│   │   ├── __init__.py
│   │   ├── collectors/                 # Data collection modules
│   │   │   ├── __init__.py
│   │   │   ├── news_collector.py       # CoinDesk/CryptoSlate scraping
│   │   │   ├── price_collector.py      # CoinGecko/CryptoCompare APIs
│   │   │   └── social_collector.py     # Reddit API integration
│   │   ├── processors/                 # Data processing modules
│   │   │   ├── __init__.py
│   │   │   ├── news_processor.py       # News content processing
│   │   │   ├── price_processor.py      # OHLCV data processing
│   │   │   └── social_processor.py     # Reddit data processing
│   │   ├── validators/                 # Data quality validation
│   │   │   ├── __init__.py
│   │   │   ├── schema_validator.py     # Pandera schema validation
│   │   │   └── quality_checker.py      # Great Expectations integration
│   │   └── pipeline.py                 # Main data pipeline orchestration
│   │
│   ├── features/                       # Feature engineering
│   │   ├── __init__.py
│   │   ├── price_features.py           # Technical indicators (TA-Lib)
│   │   ├── sentiment_features.py       # CryptoBERT sentiment processing
│   │   ├── social_features.py          # Reddit engagement features
│   │   ├── time_features.py            # Time-based features
│   │   └── feature_store.py            # Feature storage and serving
│   │
│   ├── models/                         # ML models and training
│   │   ├── __init__.py
│   │   ├── sentiment_model.py          # CryptoBERT implementation
│   │   ├── price_model.py              # XGBoost price prediction
│   │   ├── trainer.py                  # Model training orchestration
│   │   ├── evaluator.py                # Model evaluation and validation
│   │   └── registry.py                 # MLflow model registry integration
│   │
│   ├── api/                            # FastAPI backend
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI application entry point
│   │   ├── routes/                     # API route definitions
│   │   │   ├── __init__.py
│   │   │   ├── predictions.py          # Prediction endpoints
│   │   │   ├── health.py               # Health check endpoints
│   │   │   └── monitoring.py           # Monitoring endpoints
│   │   ├── models/                     # Pydantic request/response models
│   │   │   ├── __init__.py
│   │   │   ├── prediction_models.py    # Prediction API schemas
│   │   │   └── monitoring_models.py    # Monitoring API schemas
│   │   ├── services/                   # Business logic services
│   │   │   ├── __init__.py
│   │   │   ├── prediction_service.py   # Prediction logic
│   │   │   └── monitoring_service.py   # Metrics collection
│   │   └── middleware/                 # Custom middleware
│   │       ├── __init__.py
│   │       ├── logging_middleware.py   # Request logging
│   │       └── metrics_middleware.py   # Prometheus metrics
│   │
│   ├── mlops/                          # MLOps automation
│   │   ├── __init__.py
│   │   ├── drift_detector.py           # Data drift detection
│   │   ├── performance_monitor.py      # Model performance monitoring
│   │   ├── retraining_scheduler.py     # Automated retraining logic
│   │   └── alerting.py                 # Slack/email notifications
│   │
│   └── shared/                         # Shared utilities
│       ├── __init__.py
│       ├── config.py                   # Configuration management
│       ├── database.py                 # Database connections
│       ├── storage.py                  # Storage connections (Redis/S3)
│       ├── logging.py                  # Logging configuration
│       └── utils.py                    # General utilities
│
├── frontend/                           # React TypeScript application
│   ├── package.json                    # Node.js dependencies
│   ├── tsconfig.json                   # TypeScript configuration
│   ├── tailwind.config.js              # Tailwind CSS configuration
│   ├── vite.config.ts                  # Vite build configuration
│   ├── public/                         # Static assets
│   │   ├── index.html                  # HTML template
│   │   └── favicon.ico                 # Application icon
│   └── src/                            # Frontend source code
│       ├── main.tsx                    # Application entry point
│       ├── App.tsx                     # Root component
│       ├── components/                 # Reusable UI components
│       │   ├── common/                 # Generic components
│       │   │   ├── Button.tsx
│       │   │   ├── Card.tsx
│       │   │   └── LoadingSpinner.tsx
│       │   ├── charts/                 # Chart components
│       │   │   ├── PerformanceChart.tsx
│       │   │   ├── PredictionChart.tsx
│       │   │   └── DriftChart.tsx
│       │   └── dashboard/              # Dashboard components
│       │       ├── PredictionPanel.tsx
│       │       ├── MetricsPanel.tsx
│       │       └── SystemStatus.tsx
│       ├── hooks/                      # Custom React hooks
│       │   ├── useWebSocket.ts         # WebSocket integration
│       │   ├── useApi.ts               # API calls
│       │   └── usePredictions.ts       # Prediction logic
│       ├── services/                   # Frontend services
│       │   ├── api.ts                  # API service layer
│       │   ├── websocket.ts            # WebSocket service
│       │   └── storage.ts              # Local storage utilities
│       ├── types/                      # TypeScript type definitions
│       │   ├── api.ts                  # API types
│       │   ├── models.ts               # Data model types
│       │   └── ui.ts                   # UI component types
│       ├── pages/                      # Page components
│       │   ├── Dashboard.tsx           # Main dashboard
│       │   ├── PredictionsPage.tsx     # Live predictions
│       │   └── MonitoringPage.tsx      # System monitoring
│       └── utils/                      # Frontend utilities
│           ├── formatters.ts           # Data formatting
│           ├── constants.ts            # Application constants
│           └── helpers.ts              # General helpers
│
├── platform/                          # Platform-specific deployment configurations
│   ├── vercel/                         # Vercel (Frontend) configurations
│   │   ├── vercel.json                 # Vercel deployment configuration
│   │   ├── .env.production             # Frontend production environment variables
│   │   └── build-config.js             # Custom build configuration
│   ├── render/                         # Render (Backend/Services) configurations
│   │   ├── render.yaml                 # Multi-service deployment configuration
│   │   ├── api/                        # API service configuration
│   │   │   ├── Dockerfile              # FastAPI service container
│   │   │   └── startup.sh              # Service startup script
│   │   ├── mlflow/                     # MLflow service configuration
│   │   │   ├── Dockerfile              # MLflow service container
│   │   │   └── startup.sh              # MLflow startup script
│   │   ├── prometheus/                 # Prometheus service configuration
│   │   │   ├── Dockerfile              # Prometheus container
│   │   │   ├── prometheus.yml          # Prometheus configuration
│   │   │   └── startup.sh              # Prometheus startup script
│   │   └── grafana/                    # Grafana service configuration
│   │       ├── Dockerfile              # Grafana container
│   │       ├── datasources.yaml        # Grafana data source configuration
│   │       ├── dashboards/             # Dashboard definitions
│   │       │   ├── model_performance.json
│   │       │   ├── system_health.json
│   │       │   └── data_quality.json
│   │       └── startup.sh              # Grafana startup script
│   └── external/                       # External service configurations
│       ├── neondb/                     # NeonDB PostgreSQL configuration
│       │   ├── schema.sql              # Database schema definition
│       │   ├── migrations/             # Database migration scripts
│       │   │   ├── 001_initial_schema.sql
│       │   │   ├── 002_add_features_table.sql
│       │   │   └── 003_add_monitoring_tables.sql
│       │   └── seed_data.sql           # Development seed data
│       ├── upstash/                    # Upstash Redis configuration
│       │   ├── redis_config.yaml       # Redis configuration settings
│       │   └── key_patterns.yaml       # Redis key naming patterns
│       └── cloudinary/                 # Cloudinary file storage configuration
│           ├── upload_presets.json     # Upload configuration presets
│           └── transformation_rules.json # Image transformation rules
│
├── tests/                              # All test files
│   ├── __init__.py
│   ├── conftest.py                     # Pytest configuration
│   ├── unit/                           # Unit tests
│   │   ├── test_data_engineering/
│   │   │   ├── test_collectors.py
│   │   │   ├── test_processors.py
│   │   │   └── test_validators.py
│   │   ├── test_features/
│   │   │   ├── test_price_features.py
│   │   │   ├── test_sentiment_features.py
│   │   │   └── test_social_features.py
│   │   ├── test_models/
│   │   │   ├── test_sentiment_model.py
│   │   │   ├── test_price_model.py
│   │   │   └── test_trainer.py
│   │   └── test_api/
│   │       ├── test_routes.py
│   │       └── test_services.py
│   ├── integration/                    # Integration tests
│   │   ├── test_data_pipeline.py
│   │   ├── test_ml_pipeline.py
│   │   ├── test_api_integration.py
│   │   └── test_cross_platform.py      # Multi-platform integration tests
│   ├── e2e/                           # End-to-end tests
│   │   ├── test_full_workflow.py
│   │   └── test_deployment_pipeline.py # Deployment workflow tests
│   └── fixtures/                       # Test data and factories
│       ├── data_fixtures.py
│       ├── model_fixtures.py
│       ├── api_fixtures.py
│       └── platform_fixtures.py        # Platform-specific test fixtures
│
├── config/                             # Configuration files
│   ├── settings/                       # Environment-specific settings
│   │   ├── base.yaml                   # Base application settings
│   │   ├── development.yaml            # Development overrides
│   │   ├── staging.yaml                # Staging environment settings
│   │   └── production.yaml             # Production environment settings
│   ├── logging/                        # Logging configurations
│   │   ├── development.yaml            # Development logging
│   │   ├── production.yaml             # Production logging
│   │   └── monitoring.yaml             # Monitoring-specific logging
│   └── models/                         # ML model configurations
│       ├── sentiment_model.yaml        # CryptoBERT configuration
│       ├── price_model.yaml            # XGBoost configuration
│       └── training.yaml               # Training pipeline configuration
│
├── infrastructure/                     # DevOps and local development
│   ├── docker/                         # Local development Docker configurations
│   │   ├── postgres/
│   │   │   └── init.sql                # Local database initialization
│   │   ├── redis/
│   │   │   └── redis.conf              # Local Redis configuration
│   │   ├── minio/
│   │   │   └── setup.sh                # Local MinIO setup script
│   │   └── nginx/
│   │       └── nginx.conf              # Local reverse proxy configuration
│   ├── monitoring/                     # Local monitoring configurations
│   │   ├── prometheus.yml              # Local Prometheus configuration
│   │   └── grafana/
│   │       └── dashboards/             # Local Grafana dashboards
│   │           ├── development.json
│   │           └── testing.json
│   └── scripts/                        # Infrastructure scripts
│       ├── setup_local_env.sh          # Local environment setup
│       ├── deploy_to_staging.sh        # Staging deployment
│       ├── deploy_to_production.sh     # Production deployment
│       └── health_check.sh             # Cross-platform health check
│
├── dags/                               # Airflow DAGs
│   ├── data_collection_dag.py          # Data collection pipeline
│   ├── feature_engineering_dag.py      # Feature processing pipeline
│   ├── model_training_dag.py           # Model training pipeline
│   └── deployment_dag.py               # Automated deployment pipeline
│
├── data/                               # Local data storage (gitignored)
│   ├── raw/                            # Raw collected data
│   │   ├── price/                      # Price data files
│   │   ├── news/                       # News article files
│   │   └── social/                     # Social media data files
│   ├── processed/                      # Processed data
│   │   ├── features/                   # Feature engineering output
│   │   └── training/                   # Training datasets
│   └── exports/                        # Data export files
│       ├── backups/                    # Data backups
│       └── reports/                    # Analysis reports
│
├── models/                             # Model artifacts (gitignored)
│   ├── sentiment/                      # CryptoBERT model files
│   │   ├── checkpoints/                # Model checkpoints
│   │   └── exports/                    # Exported model files
│   ├── price/                          # XGBoost model files
│   │   ├── versions/                   # Model versions
│   │   └── experiments/                # Experiment artifacts
│   └── metadata/                       # Model metadata and configs
│       ├── performance/                # Performance metrics
│       └── lineage/                    # Model lineage tracking
│
├── scripts/                            # Utility scripts
│   ├── development/                    # Development utilities
│   │   ├── setup_dev.py                # Development environment setup
│   │   ├── reset_dev_data.py           # Reset development data
│   │   └── run_local_tests.py          # Local testing script
│   ├── data/                           # Data management scripts
│   │   ├── collect_historical_data.py  # Historical data collection
│   │   ├── migrate_data.py             # Data migration utilities
│   │   └── validate_data.py            # Data validation script
│   ├── models/                         # Model management scripts
│   │   ├── train_model.py              # Manual model training
│   │   ├── evaluate_model.py           # Model evaluation script
│   │   └── deploy_model.py             # Model deployment script
│   └── deployment/                     # Deployment utilities
│       ├── deploy_frontend.py          # Frontend deployment script
│       ├── deploy_backend.py           # Backend deployment script
│       ├── setup_external_services.py  # External service setup
│       └── validate_deployment.py      # Deployment validation
│
├── docs/                               # Documentation
│   ├── api/                            # API documentation
│   │   ├── openapi.yaml                # OpenAPI specification
│   │   └── endpoints/                  # Endpoint documentation
│   │       ├── predictions.md
│   │       ├── monitoring.md
│   │       └── health.md
│   ├── architecture/                   # System architecture docs
│   │   ├── system_design.md
│   │   ├── data_flow.md
│   │   ├── deployment.md
│   │   └── multi_platform.md           # Multi-platform architecture
│   ├── deployment/                     # Deployment documentation
│   │   ├── vercel_setup.md             # Vercel deployment guide
│   │   ├── render_setup.md             # Render deployment guide
│   │   ├── external_services.md        # External service configuration
│   │   └── ci_cd_pipeline.md           # CI/CD pipeline documentation
│   ├── user_guide/                     # User documentation
│   │   ├── getting_started.md
│   │   ├── api_usage.md
│   │   ├── dashboard_guide.md
│   │   └── monitoring_guide.md
│   └── development/                    # Development documentation
│       ├── setup.md
│       ├── testing.md
│       ├── contributing.md
│       ├── platform_development.md     # Multi-platform development guide
│       └── troubleshooting.md
│
└── .github/                            # GitHub-specific files
    ├── workflows/                      # CI/CD workflows
    │   ├── ci.yml                      # Continuous integration
    │   ├── deploy_frontend.yml         # Vercel deployment workflow
    │   ├── deploy_backend.yml          # Render deployment workflow
    │   ├── deploy_services.yml         # Services deployment workflow
    │   ├── model_training.yml          # Automated model training
    │   └── cross_platform_tests.yml    # Multi-platform integration tests
    ├── ISSUE_TEMPLATE/                 # Issue templates
    │   ├── bug_report.md
    │   ├── feature_request.md
    │   └── deployment_issue.md
    └── PULL_REQUEST_TEMPLATE.md        # Pull request template
```

---

## File Purpose Justification

### **New Platform-Specific Organization**

#### **platform/ Directory**
**Purpose:** Centralized platform-specific deployment configurations
- **vercel/:** Frontend deployment settings, build configurations, environment variables
- **render/:** Backend service definitions, container configurations, startup scripts
- **external/:** External service configurations and setup templates

#### **Platform Configuration Benefits**
- ✅ **Deployment clarity:** All platform configs in one location
- ✅ **Environment separation:** Clear separation between local dev and production
- ✅ **Service organization:** Each service has dedicated configuration space
- ✅ **Scalability:** Easy to add new platforms or services

### **Enhanced Testing Structure**

#### **Cross-Platform Testing**
- **test_cross_platform.py:** Multi-platform integration testing
- **test_deployment_pipeline.py:** End-to-end deployment workflow testing
- **platform_fixtures.py:** Platform-specific test data and configurations

### **Expanded Configuration Management**

#### **Environment-Specific Settings**
- **Base configuration:** Common settings across all environments
- **Environment overrides:** Specific settings for development/staging/production
- **Logging configurations:** Environment-appropriate logging levels and outputs
- **Model configurations:** ML model settings and hyperparameters

### **Enhanced Documentation Structure**

#### **Multi-Platform Documentation**
- **multi_platform.md:** Architecture documentation for distributed deployment
- **Platform setup guides:** Specific instructions for each deployment platform
- **platform_development.md:** Development workflow for multi-platform architecture

---

## File Naming Conventions

### **Platform Configurations**
- **Service-specific:** `api/Dockerfile`, `mlflow/Dockerfile`
- **Configuration files:** `render.yaml`, `vercel.json`
- **Startup scripts:** `startup.sh` for service initialization
- **Environment files:** `.env.production`, `.env.staging`

### **Multi-Platform Components**
- **Cross-platform tests:** `test_cross_platform.py`
- **Deployment workflows:** `deploy_frontend.yml`, `deploy_backend.yml`
- **Platform documentation:** `vercel_setup.md`, `render_setup.md`

### **Configuration Hierarchy**
- **Base settings:** `base.yaml` with common configurations
- **Environment overrides:** `development.yaml`, `production.yaml`
- **Service-specific:** `sentiment_model.yaml`, `price_model.yaml`

---

## Multi-Platform Integration Points

### **Service Communication**
- **API endpoints:** Environment-specific service URLs in configuration
- **Authentication:** Service-to-service authentication tokens
- **Health checks:** Cross-service health validation endpoints

### **Data Flow**
- **Frontend → Backend:** Vercel frontend communicating with Render API
- **Backend → Services:** API communicating with MLflow, Prometheus on Render
- **External Services:** All services connecting to NeonDB, Upstash, Cloudinary

### **Deployment Coordination**
- **CI/CD orchestration:** GitHub Actions coordinating multi-platform deployments
- **Service dependencies:** Proper startup order and health checks
- **Configuration management:** Environment-specific settings per platform

---

## Development Workflow Integration

### **Local Development**
- **Infrastructure/:** Docker configurations for local service emulation
- **Scripts/development/:** Utilities for local development environment
- **Config/:** Local development configurations

### **Platform Development**
- **Platform/:** Platform-specific configurations for testing deployments
- **Tests/integration/:** Cross-platform integration testing
- **Docs/deployment/:** Platform-specific deployment guides

### **Production Deployment**
- **Platform configurations:** Ready-to-deploy service definitions
- **CI/CD workflows:** Automated deployment to multiple platforms
- **External service configs:** Production-ready external service setup

---

## Critical Dependencies & Integrations

### **Technology Stack Alignment**
- **Data Engineering:** Supports Airflow, collectors, processors, validators
- **ML Engineering:** MLflow integration across platforms, model training and serving
- **Data Science:** Feature engineering, model artifacts, experiment tracking
- **MLOps:** Monitoring, drift detection, automated retraining across platforms
- **Frontend:** React TypeScript with platform-optimized deployment

### **Platform-Specific Optimizations**
- **Vercel optimizations:** Frontend build settings, CDN configuration, edge functions
- **Render optimizations:** Service resource allocation, networking, persistent storage
- **External service integration:** Optimal connection patterns for each platform

### **Development-to-Production Pipeline**
- **Local development:** Full-featured local environment with Docker
- **Platform testing:** Staging deployments on production platforms
- **Production deployment:** Automated deployment with health checks and rollback

---

## Environment-Specific Considerations

### **Development Environment**
- **Local services:** Docker-based local development
- **Configuration:** Development-specific settings and debug mode
- **Testing:** Local testing with service mocks and real integrations

### **Staging Environment**
- **Platform deployment:** Production platforms with staging configurations
- **Data:** Production-like data for realistic testing
- **Validation:** Pre-production testing and performance validation

### **Production Environment**
- **Multi-platform deployment:** Optimized deployment across Vercel and Render
- **Monitoring:** Full monitoring stack with alerting and dashboards
- **Security:** Production security configurations and secret management

---

## Critical Success Factors

### **Platform Integration**
- ✅ **Service coordination:** Seamless communication between platforms
- ✅ **Deployment automation:** Coordinated deployment across platforms
- ✅ **Configuration management:** Consistent settings across environments
- ✅ **Monitoring integration:** Unified monitoring across distributed services

### **Development Efficiency**
- ✅ **Local development:** Fast iteration with Docker services
- ✅ **Platform testing:** Easy staging deployment for validation
- ✅ **Production deployment:** Automated, reliable production deployment
- ✅ **Debugging support:** Clear logging and monitoring across platforms

### **Operational Excellence**
- ✅ **Service reliability:** Health checks and automatic recovery
- ✅ **Performance monitoring:** Comprehensive metrics across all services
- ✅ **Security management:** Secure configuration and secret management
- ✅ **Scalability planning:** Architecture supports independent service scaling

---

**This revised structure provides complete support for our multi-platform deployment strategy while maintaining clean organization and development efficiency. The platform-specific configurations enable seamless deployment to Vercel and Render while preserving all desired functionality.**