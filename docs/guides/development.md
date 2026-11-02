# Development Guide

This guide provides detailed instructions for setting up and working with the UnMask development environment.

## Development Environment Setup

### Prerequisites

- Python 3.7+
- Git
- PostgreSQL 12+
- Redis 6+
- CUDA-compatible GPU (recommended)

### Initial Setup

1. Clone the repository:

```bash
git clone https://github.com/BSM21/UNMASK.git
cd deffyp
```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

4. Set up pre-commit hooks:

```bash
pre-commit install
```

5. Configure environment variables:

```bash
cp .env.example .env
# Edit .env with your development settings
```

6. Initialize database:

```bash
alembic upgrade head
```

## Development Workflow

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run all checks:

```bash
pre-commit run --all-files
```

### Testing

1. Run tests:

```bash
pytest
```

2. Generate coverage report:

```bash
pytest --cov=UnMask tests/
```

3. Run specific test file:

```bash
pytest tests/test_file.py
```

### Documentation

1. Build documentation:

```bash
cd docs
make html
```

2. View documentation:

```bash
open _build/html/index.html
```

## Project Structure

```
UnMask/
├── ai_model/          # AI model implementation
│   ├── config/        # Model configurations
│   ├── models/        # Model definitions
│   └── utils/         # Model utilities
├── api/               # API endpoints
│   ├── routes/        # API routes
│   └── schemas/       # Pydantic schemas
├── core/              # Core functionality
│   ├── config/        # Application config
│   ├── database/      # Database models
│   └── security/      # Authentication
├── docs/              # Documentation
├── front_end/         # Web interface
│   ├── static/        # Static files
│   └── templates/     # HTML templates
├── tests/             # Test suite
└── utils/             # Utility functions
```

## API Development

### Adding New Endpoints

1. Create route file in `api/routes/`
2. Define Pydantic schemas in `api/schemas/`
3. Add route to `api/main.py`
4. Write tests in `tests/api/`

### Testing API Endpoints

```bash
# Run API tests
pytest tests/api/

# Test specific endpoint
pytest tests/api/test_endpoint.py
```

## Frontend Development

### Setup

1. Install dependencies:

```bash
cd front_end
npm install
```

2. Start development server:

```bash
npm run dev
```

### Adding New Components

1. Create component in `front_end/components/`
2. Add styles in `front_end/styles/`
3. Update routes in `front_end/routes/`

## Database Management

### Migrations

1. Create new migration:

```bash
alembic revision --autogenerate -m "description"
```

2. Apply migrations:

```bash
alembic upgrade head
```

3. Rollback migration:

```bash
alembic downgrade -1
```

## Deployment

### Local Deployment

1. Build frontend:

```bash
cd front_end
npm run build
```

2. Start application:

```bash
uvicorn main:app --reload
```

### Production Deployment

See [Deployment Guide](deployment.md) for detailed instructions.

## Troubleshooting

### Common Issues

1. **Database Connection**

   - Check PostgreSQL is running
   - Verify connection string in `.env`

2. **GPU Issues**

   - Verify CUDA installation
   - Check GPU memory usage

3. **API Errors**
   - Check error logs
   - Verify endpoint documentation

### Getting Help

- Email: bilal.saeed.malik123@gmail.com or zainulabidin970946@gmail.com
