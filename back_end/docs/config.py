from typing import Dict, Any

API_TITLE = "UnMask API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
UnMask API for deepfake detection and analysis.

## Features
- User authentication and authorization
- File upload and processing
- Deepfake detection
- Result analysis and reporting
- Processing status tracking

## Authentication
Most endpoints require authentication using JWT tokens.
To authenticate, use the `/auth/login` endpoint with your credentials.
Include the received token in the `Authorization` header as `Bearer <token>`.

## Error Handling
The API uses standard HTTP status codes:
- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 500: Internal Server Error

Error responses include a JSON body with a `detail` field describing the error.
"""

TAGS_METADATA = [
    {
        "name": "auth",
        "description": "Authentication and user management operations",
    },
    {
        "name": "users",
        "description": "User profile and management operations",
    },
    {
        "name": "files",
        "description": "File upload and management operations",
    },
    {
        "name": "processing",
        "description": "File processing and result management",
    },
    {
        "name": "models",
        "description": "Model management and configuration",
    },
    {
        "name": "predictions",
        "description": "Prediction and analysis operations",
    },
]

SWAGGER_UI_PARAMETERS = {
    "displayRequestDuration": True,
    "filter": True,
    "docExpansion": "list",
    "persistAuthorization": True,
    "syntaxHighlight.theme": "monokai",
} 