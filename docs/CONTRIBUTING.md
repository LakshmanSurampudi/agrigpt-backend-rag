# Contributing to AgriGPT Backend RAG

Thank you for your interest in contributing to AgriGPT Backend RAG! This document provides guidelines and instructions for contributing to this open-source project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Style Guide](#style-guide)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community](#community)

---

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Examples of behavior that contributes to a positive environment:**

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Examples of unacceptable behavior:**

- The use of sexualized language or imagery and unwelcome sexual attention or advances
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team. All complaints will be reviewed and investigated promptly and fairly.

---

## 🎯 Project Overview

AgriGPT Backend RAG is a modern Retrieval-Augmented Generation (RAG) chatbot backend built with FastAPI. The project enables intelligent document processing and question-answering capabilities using:

- **FastAPI** - High-performance Python web framework
- **LangChain** - Framework for LLM applications
- **Google Gemini** - Advanced language model for generation
- **Pinecone** - Vector database for semantic search
- **CLIP** - Multimodal embeddings for text and images
- **Cloudflare R2** - Object storage for media files

### Key Features

- 📄 PDF document upload and processing
- 💬 AI-powered question answering with source citations
- 🖼️ Multimodal support (text and images)
- 🔍 Semantic search using vector embeddings
- 🗑️ Knowledge base management
- 📊 LangSmith observability integration

---

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11.x** (required - not 3.12 or 3.13)
- **Git** for version control
- **pip** for package management
- **Virtual environment** tool (venv or virtualenv)

### Required API Keys

You'll need to obtain the following API keys:

1. **Google AI API Key** - [Get it here](https://ai.google.dev/)
2. **Pinecone API Key** - [Get it here](https://www.pinecone.io/)
3. **LangSmith API Key** (optional) - [Get it here](https://smith.langchain.com/)
4. **Cloudflare R2 Credentials** (optional for storage features)

---

## 💻 Development Setup

### 1. Fork and Clone the Repository

```bash
# Fork the repository on GitHub first, then clone your fork
git clone https://github.com/YOUR_USERNAME/agrigpt-backend-rag.git
cd agrigpt-backend-rag

# Add upstream remote
git remote add upstream https://github.com/alumnx-ai-labs/agrigpt-backend-rag.git
```

### 2. Create a Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Install development dependencies (if available)
pip install pytest pytest-asyncio black flake8 mypy
```

### 4. Configure Environment Variables

```bash
# Copy the template
cp .env.template .env

# Edit .env with your API keys
nano .env  # or use your preferred editor
```

Required environment variables:

```env
# Google AI (Required)
GOOGLE_API_KEY=your_google_api_key_here

# Pinecone (Required)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=agrigpt-backend-rag-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# LangSmith (Optional - for observability)
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=agrigpt-backend-rag

# Cloudflare R2 (Optional - for storage)
R2_ACCOUNT_ID=your_r2_account_id
R2_ACCESS_KEY_ID=your_r2_access_key
R2_SECRET_ACCESS_KEY=your_r2_secret_key
R2_BUCKET_NAME=your_bucket_name
R2_PUBLIC_URL=your_public_url
```

### 5. Create Pinecone Index

Before running the application, create a Pinecone index:

1. Log in to [Pinecone Console](https://app.pinecone.io/)
2. Create a new index with the following settings:
   - **Name**: `agrigpt-backend-rag-index` (or your custom name)
   - **Dimensions**: `768` (for CLIP embeddings)
   - **Metric**: `cosine`
   - **Cloud**: `aws`
   - **Region**: `us-east-1`

### 6. Run the Development Server

```bash
# Start the FastAPI server with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 7. Verify Installation

```bash
# Test the health endpoint
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "service": "RAG Chatbot API",
#   "services_ready": true,
#   "clip_available": true,
#   "clip_import_error": null
# }
```

---

## 📁 Project Structure

```
agrigpt-backend-rag/
├── .env                      # Environment variables (gitignored)
├── .env.template             # Environment template
├── .gitignore                # Git ignore rules
├── .python-version           # Python version specification
├── main.py                   # FastAPI application entry point
├── requirements.txt          # Python dependencies
├── render.yaml               # Render deployment configuration
├── README.md                 # Project documentation
├── USAGE_GUIDE.md            # Usage instructions
│
├── docs/                     # Documentation
│   └── CONTRIBUTING.md       # This file
│
├── routes/                   # API route handlers
│   ├── rag_routes.py         # RAG endpoints (upload, chat, clear)
│   ├── clip_routes.py        # CLIP query endpoints
│   └── clip_ingest_routes.py # CLIP ingestion endpoints
│
├── services/                 # Business logic layer
│   ├── rag_service.py        # RAG service (PDF processing, chat)
│   ├── clip_service.py       # CLIP query service
│   ├── clip_ingest_service.py # CLIP ingestion service
│   ├── r2_storage_service.py # Cloudflare R2 storage
│   └── local_storage_service.py # Local file storage
│
├── tests/                    # Test files
│   ├── test_clip_isolated.py
│   └── ...
│
├── data/                     # Data directory (gitignored)
│   └── uploaded_files/       # Temporary file storage
│
└── static/                   # Static files (images, etc.)
```

### Directory Responsibilities

- **`routes/`**: Define API endpoints and request/response handling
- **`services/`**: Implement core business logic and external integrations
- **`tests/`**: Unit and integration tests
- **`docs/`**: Project documentation
- **`data/`**: Runtime data storage (not committed to git)
- **`static/`**: Static assets served by the API

---

## 🎨 Coding Standards

### Python Version

- **Use Python 3.11.x** exclusively
- Do not use features from Python 3.12+ that are incompatible with 3.11
- Check compatibility before adding new dependencies

### Code Quality

All code contributions must:

1. **Follow PEP 8** - Python's official style guide
2. **Pass linting** - Use `flake8` for linting
3. **Include type hints** - Use Python type annotations where applicable
4. **Be well-documented** - Include docstrings for all functions and classes
5. **Handle errors gracefully** - Use proper exception handling
6. **Be tested** - Include unit tests for new features

### Code Organization

- **Separation of Concerns**: Keep routes, services, and utilities separate
- **Single Responsibility**: Each function/class should have one clear purpose
- **DRY Principle**: Don't Repeat Yourself - extract common logic
- **KISS Principle**: Keep It Simple, Stupid - avoid over-engineering

### Naming Conventions

```python
# Variables and functions: snake_case
user_name = "John"
def calculate_total_price():
    pass

# Classes: PascalCase
class RAGService:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_FILE_SIZE = 10_000_000
API_VERSION = "1.0.0"

# Private methods/variables: prefix with underscore
def _internal_helper():
    pass
```

### Import Organization

Organize imports in the following order:

```python
# 1. Standard library imports
import os
import asyncio
from typing import List, Dict, Optional

# 2. Third-party imports
from fastapi import FastAPI, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# 3. Local application imports
from services.rag_service import RAGService
from routes.clip_routes import router
```

---

## 📝 Style Guide

### Docstrings

Use Google-style docstrings for all functions and classes:

```python
def process_document(file_path: str, chunk_size: int = 1000) -> List[str]:
    """
    Process a document and split it into chunks.

    Args:
        file_path: Path to the document file
        chunk_size: Maximum size of each chunk in characters

    Returns:
        List of text chunks extracted from the document

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If chunk_size is less than 100

    Example:
        >>> chunks = process_document("document.pdf", chunk_size=500)
        >>> len(chunks)
        42
    """
    pass
```

### Type Hints

Always use type hints for function parameters and return values:

```python
from typing import List, Dict, Optional, Union

async def fetch_user_data(
    user_id: str,
    include_metadata: bool = False
) -> Optional[Dict[str, Union[str, int]]]:
    """Fetch user data from the database."""
    pass
```

### Error Handling

Use specific exception types and provide helpful error messages:

```python
from fastapi import HTTPException

async def upload_file(file: UploadFile):
    """Upload and process a file."""
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )

        # Process file
        content = await file.read()

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log unexpected errors
        print(f"Error processing file: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process file: {str(e)}"
        )
```

### Async/Await

Use async functions for I/O operations:

```python
# Good: Async for I/O operations
async def fetch_embeddings(text: str) -> List[float]:
    """Fetch embeddings from the API."""
    response = await client.get_embeddings(text)
    return response.embeddings

# Bad: Blocking I/O in async function
async def fetch_embeddings_bad(text: str) -> List[float]:
    response = requests.get(f"api/embeddings?text={text}")  # Blocking!
    return response.json()
```

### Logging

Use proper logging instead of print statements:

```python
import logging

logger = logging.getLogger(__name__)

async def process_request(data: dict):
    """Process incoming request."""
    logger.info(f"Processing request with {len(data)} items")
    try:
        result = await perform_operation(data)
        logger.debug(f"Operation completed: {result}")
        return result
    except Exception as e:
        logger.error(f"Operation failed: {e}", exc_info=True)
        raise
```

---

## 📋 Commit Guidelines

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, missing semicolons, etc.)
- **refactor**: Code refactoring without changing functionality
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **chore**: Maintenance tasks (dependencies, build config, etc.)
- **ci**: CI/CD configuration changes

### Scopes

Common scopes in this project:

- **rag**: RAG service and routes
- **clip**: CLIP service and routes
- **storage**: Storage services (R2, local)
- **api**: API routes and endpoints
- **deps**: Dependencies
- **config**: Configuration files
- **docs**: Documentation

### Examples

```bash
# Feature addition
feat(clip): add multimodal image search capability

# Bug fix
fix(rag): resolve PDF parsing error for scanned documents

# Documentation
docs(readme): update setup instructions for Python 3.11

# Refactoring
refactor(storage): extract common storage interface

# Performance improvement
perf(clip): optimize embedding generation with batch processing

# Dependency update
chore(deps): upgrade langchain to 0.2.16

# Breaking change
feat(api)!: change chat endpoint response format

BREAKING CHANGE: The chat endpoint now returns a structured response
with separate fields for answer and sources instead of a single text field.
```

### Commit Message Rules

1. **Use imperative mood**: "add feature" not "added feature"
2. **Keep subject line under 72 characters**
3. **Capitalize the subject line**
4. **Don't end subject line with a period**
5. **Separate subject from body with a blank line**
6. **Wrap body at 72 characters**
7. **Use body to explain what and why, not how**

### Example with Body

```
feat(rag): implement document chunking with overlap

Add support for overlapping chunks to improve context preservation
across chunk boundaries. This helps maintain semantic coherence when
retrieving relevant passages.

- Add overlap_size parameter to chunking function
- Update tests to verify overlap behavior
- Document new parameter in API docs

Closes #123
```

---

## 🔄 Pull Request Process

### Before Submitting a PR

1. **Sync with upstream**:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests**:

   ```bash
   pytest tests/
   ```

3. **Check code quality**:

   ```bash
   # Format code
   black .

   # Check linting
   flake8 .

   # Type checking
   mypy .
   ```

4. **Update documentation** if needed

### Workflow: Addressing Issues

#### External Contributors (Forks)

If you are contributing from a fork (standard for open source contributors):

1. **Claim the Issue**: Comment on the issue you want to work on. Wait for a maintainer to assign it to you or give the go-ahead to avoid duplicate work.
2. **Fork & Clone**: Fork the repository and clone it locally (see [Development Setup](#development-setup)).
3. **Create a Branch**: Create a descriptive branch in your fork (e.g., `fix/42-crash-on-startup`).
4. **Implement**: Make your changes and commit them.
5. **Sync**: Ensure your branch is up to date with `upstream/main` before pushing.
6. **Push**: Push the branch to your fork `git push origin fix/42-crash-on-startup`.
7. **Open PR**: Create a Pull Request from your fork against the `upstream` repository's `main` branch.
8. **Link Issue**: In the **PR description**, **you must link the issue** using a supported keyword (e.g., `Fixes #42`, `Closes #42`, `Resolves #42`). This ensures the issue is automatically closed when the PR is merged.

- **`Fixes #<issue_number>`**: Indicates that the changes in the PR address and resolve the specified issue.
- **`Closes #<issue_number>`**: Similar to `Fixes`, this keyword also signifies that the PR's merge will close the linked issue.
- **`Resolves #<issue_number>`**: Another keyword with the same effect, communicating that the PR provides a solution for the referenced issue.

#### Internal Contributors

1. **Assign Issue**: Assign the relevant issue to yourself on GitHub.
2. **Branch**: Create a branch directly in the main repository.
3. **Open PR**: Create a PR and link the issue as described above (e.g., `Fixes #42`).

### PR Title Format

Use the same format as commit messages:

```
feat(clip): add image similarity search
fix(rag): resolve memory leak in document processing
```

### PR Description Template

```markdown
## Description

Brief description of what this PR does.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Related Issues

Fixes #(issue number)
Relates to #(issue number)

## Changes Made

- Change 1
- Change 2
- Change 3

## Testing

Describe the tests you ran and how to reproduce them:

- [ ] Test A
- [ ] Test B

## Screenshots (if applicable)

Add screenshots to help explain your changes.

## Checklist

- [ ] My code follows the project's coding standards
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings or errors
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Additional Notes

Any additional information that reviewers should know.
```

### Review Process

1. **Automated Checks**: All PRs must pass automated tests and linting
2. **Code Review**: At least one maintainer must review and approve
3. **Testing**: Verify that changes work as expected
4. **Documentation**: Ensure documentation is updated if needed
5. **Merge**: Maintainers will merge approved PRs

### PR Guidelines

- **Keep PRs focused**: One feature/fix per PR
- **Keep PRs small**: Easier to review and merge
- **Write clear descriptions**: Explain what and why
- **Respond to feedback**: Address review comments promptly
- **Be patient**: Reviews may take time
- **Be respectful**: Maintain a positive and constructive tone

---

## 🐛 Issue Guidelines

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Check documentation** for answers
3. **Verify it's reproducible** in the latest version
4. **Gather relevant information** (logs, screenshots, etc.)

### Issue Templates

#### Bug Report Template

```markdown
## Bug Description

A clear and concise description of the bug.

## Steps to Reproduce

1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened.

## Environment

- OS: [e.g., Ubuntu 22.04]
- Python Version: [e.g., 3.11.9]
- Package Versions: [paste relevant versions from requirements.txt]

## Logs/Error Messages
```

Paste error messages or logs here

```

## Screenshots
If applicable, add screenshots.

## Additional Context
Any other context about the problem.
```

#### Feature Request Template

```markdown
## Feature Description

A clear and concise description of the feature you'd like.

## Problem Statement

What problem does this feature solve?

## Proposed Solution

How would you like this feature to work?

## Alternatives Considered

What alternative solutions have you considered?

## Additional Context

Any other context, mockups, or examples.

## Willingness to Contribute

- [ ] I'm willing to implement this feature
- [ ] I'm willing to help test this feature
- [ ] I'm just suggesting the idea
```

#### Question Template

```markdown
## Question

What would you like to know?

## Context

Provide context for your question.

## What I've Tried

What have you already tried or researched?

## Environment (if relevant)

- OS:
- Python Version:
- Package Versions:
```

### Issue Labels

- **bug**: Something isn't working
- **feature**: New feature request
- **documentation**: Documentation improvements
- **question**: Questions about usage
- **help wanted**: Extra attention needed
- **good first issue**: Good for newcomers
- **enhancement**: Improvement to existing feature
- **performance**: Performance-related issues
- **security**: Security-related issues
- **wontfix**: This will not be worked on
- **duplicate**: Duplicate of another issue

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_clip_isolated.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run with verbose output
pytest -v
```

### Writing Tests

Create test files in the `tests/` directory:

```python
# tests/test_rag_service.py
import pytest
from services.rag_service import RAGService

@pytest.fixture
async def rag_service():
    """Create a RAG service instance for testing."""
    service = RAGService()
    await service.initialize()
    yield service
    # Cleanup if needed

@pytest.mark.asyncio
async def test_document_upload(rag_service):
    """Test document upload functionality."""
    # Arrange
    test_file = "test_document.pdf"

    # Act
    result = await rag_service.upload_document(test_file)

    # Assert
    assert result["status"] == "success"
    assert "chunks" in result
    assert len(result["chunks"]) > 0

@pytest.mark.asyncio
async def test_chat_query(rag_service):
    """Test chat query functionality."""
    # Arrange
    query = "What is the main topic?"

    # Act
    response = await rag_service.chat(query)

    # Assert
    assert "answer" in response
    assert "sources" in response
    assert len(response["answer"]) > 0
```

### Test Coverage

- Aim for **80%+ code coverage**
- Focus on critical paths and edge cases
- Test both success and failure scenarios
- Include integration tests for API endpoints

---

## 📚 Documentation

### Documentation Standards

1. **Keep README.md updated** with new features
2. **Document API changes** in code and docs
3. **Add inline comments** for complex logic
4. **Update USAGE_GUIDE.md** for user-facing changes
5. **Create examples** for new features

### API Documentation

FastAPI automatically generates API documentation at `/docs`. Ensure your endpoints have:

```python
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

router = APIRouter()

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str
    max_tokens: int = 1000

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str
    sources: list[str]

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with documents",
    description="Send a query and get an AI-generated answer based on uploaded documents.",
    tags=["RAG"]
)
async def chat(request: ChatRequest):
    """
    Chat endpoint for question-answering.

    This endpoint uses RAG to answer questions based on the uploaded documents.
    It retrieves relevant chunks and generates a response using Google Gemini.

    Args:
        request: Chat request containing the query

    Returns:
        ChatResponse with answer and source citations

    Raises:
        HTTPException: If no documents are uploaded or query fails
    """
    pass
```

---

## 🤝 Community

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Email**: Contact maintainers directly for sensitive issues

### Contributing Beyond Code

You can contribute in many ways:

- **Report bugs** and suggest features
- **Improve documentation** and fix typos
- **Answer questions** from other users
- **Share your use cases** and success stories
- **Write tutorials** and blog posts
- **Spread the word** on social media

### Recognition

Contributors will be recognized in:

- **README.md** contributors section
- **Release notes** for significant contributions
- **GitHub contributors** page

---

## 📄 License

By contributing to AgriGPT Backend RAG, you agree that your contributions will be licensed under the same license as the project (specify your license here, e.g., MIT License).

---

## 🙏 Thank You!

Thank you for taking the time to contribute to AgriGPT Backend RAG! Your contributions help make this project better for everyone.

If you have any questions about contributing, feel free to reach out by opening an issue or discussion.

Happy coding! 🚀

---

**Credits**: Based on the template by [Hemanth](https://github.com/hemanth090/RagChatbot-01).
