# Contributing to UnMask

Thank you for your interest in contributing to UnMask! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### 1. Fork the Repository

1. Fork the repository on GitHub
2. Clone your fork locally
3. Configure the upstream remote

```bash
git remote add upstream https://github.com/BSM21/UNMASK.git
```

### 2. Set Up Development Environment

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

3. Install pre-commit hooks:

```bash
pre-commit install
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Make Your Changes

- Follow the project's coding style
- Write tests for new features
- Update documentation as needed
- Keep commits focused and atomic

### 5. Run Tests and Linters

```bash
# Run tests
pytest

# Run linters
flake8
black .
isort .
mypy .
```

### 6. Commit Your Changes

```bash
git add .
git commit -m "Description of your changes"
```

### 7. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 8. Create a Pull Request

1. Go to the original repository
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill in the PR template
5. Submit the PR

## Development Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions and classes
- Keep functions small and focused

### Testing

- Write unit tests for new features
- Ensure test coverage is maintained
- Run all tests before submitting PR

### Documentation

- Update relevant documentation
- Add docstrings to new functions/classes
- Update README if necessary

### Git Commit Messages

- Use present tense
- Be descriptive but concise
- Reference issues when applicable

## Project Structure

```
UnMask/
├── ai_model/          # AI model implementation
├── api/               # API endpoints
├── core/              # Core functionality
├── docs/              # Documentation
├── front_end/         # Web interface
├── tests/             # Test suite
└── utils/             # Utility functions
```

## Review Process

1. PR is assigned to reviewers
2. Reviewers provide feedback
3. Address feedback and make changes
4. PR is merged when approved

## Getting Help

- Open an issue on GitHub

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.
