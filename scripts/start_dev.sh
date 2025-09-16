#!/bin/bash
# Start complete development environment

echo "🚀 Starting Bitcoin Prediction Engine development environment"

# Start Docker services
echo "📦 Starting Docker services..."
docker-compose -f docker-compose.dev.yml up -d

# Wait for services
echo "⏳ Waiting for services to be ready..."
sleep 30

# Start backend (in background)
echo "🐍 Starting Python backend..."
cd "$(dirname "$0")/.."
poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start frontend (in background)
echo "⚛️ Starting React frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo "✅ Development environment started!"
echo ""
echo "🌐 Available services:"
echo "  - Frontend: http://localhost:3000"
echo "  - Backend API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - MLflow: http://localhost:5000"
echo "  - Airflow: http://localhost:8080"
echo "  - Grafana: http://localhost:3001"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "kill $BACKEND_PID $FRONTEND_PID; docker-compose -f docker-compose.dev.yml down; exit" INT
wait