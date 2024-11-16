# Othor API

## Overview

Its a FastAPI-based backend application designed to manage Wayne Enterprise data, handle user authentication, and generate narrative reports. This project leverages advanced data analysis and machine learning techniques to provide insightful metrics and forecasts for business intelligence.

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Technology Stack](#technology-stack)
4. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Configuration](#configuration)
5. [API Documentation](#api-documentation)
6. [Data Models](#data-models)
7. [Authentication](#authentication)
8. [Caching](#caching)
9. [Testing](#testing)
10. [Deployment](#deployment)
11. [Contributing](#contributing)
12. [License](#license)
13. [Contact](#contact)

## Features

- **User Management**: Secure signup and login functionality with JWT-based authentication.
- **Wayne Enterprise Metrics**: 
  - Leading Indicators: Real-time analysis of key performance indicators.
  - Lagging Indicators: Historical performance metrics.
  - Forecasting: Advanced predictive analytics using ensemble methods.
- **Narrative Generation**: AI-powered generation of business intelligence reports.
- **Data Caching**: Optimized performance with Redis-based caching.
- **Scalable Architecture**: Designed for high performance and easy scalability.

## Project Structure

```
OTHOR-BACKEND/
├── routers/
│   ├── narrative.py
│   ├── users.py
│   └── wayne_enterprise.py
├── services/
│   └── email_service.py
├── tests/
│   └── test_run.py
├── .env
├── auth.py
├── config.py
├── database.py
├── main.py
├── models.py
├── schemas.py
├── requirements.txt
└── wayne_enterprise_data.csv
```

## Technology Stack

- **Framework**: FastAPI
- **ORM**: SQLAlchemy
- **Data Validation**: Pydantic
- **Caching**: Redis
- **Machine Learning**: 
  - Prophet
  - SARIMA
  - Exponential Smoothing
- **Natural Language Processing**: OpenAI API
- **Authentication**: JWT
- **Testing**: pytest

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- Redis server
- PostgreSQL

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/othorai/backend-apis.git
   cd backend-apis
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Configuration

1. Create a `.env` file in the root directory with the following variables:
   ```
   DATABASE_URL=postgresql://user:password@localhost/dbname
   REDIS_URL=redis://localhost:6379
   SECRET_KEY=your_secret_key
   ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   OPENAI_API_KEY=your_openai_api_key
   ```

2. Set up the database:

## API Documentation

Once the server is running, you can access the interactive API documentation at `http://localhost:8000/docs`.

Key endpoints include:

- `POST /authorization/signup`: Create a new user account
- `POST /authorization/login`: Authenticate and receive a JWT token
- `GET /metrics/leading_indicators`: Retrieve leading indicators
- `GET /metrics/lagging_indicators`: Retrieve lagging indicators
- `GET /metrics/forecast`: Get forecast data
- `GET /narrative/feed`: Generate a narrative news feed

## Data Models

Key data models include:

- `User`: Represents user accounts
- `WayneEnterprise`: Stores Wayne Enterprise metrics and KPIs

Refer to `models.py` for detailed schema information.

## Authentication

The API uses JWT (JSON Web Tokens) for authentication. Include the JWT token in the Authorization header for protected routes:

```
Authorization: Bearer <your_token_here>
```

## Caching

Redis is used for caching frequently accessed data to improve performance. The caching logic is implemented in the respective route handlers.

