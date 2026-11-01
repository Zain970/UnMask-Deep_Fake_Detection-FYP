# UnMask API Documentation

## Overview

The UnMask API provides a comprehensive set of endpoints for deepfake detection, user management, and media processing. This document outlines the available endpoints, request/response formats, and authentication requirements.

## Base URL

```
https://api.unmask.com/v1
```

## Authentication

All API requests require authentication using JWT tokens. Include the token in the Authorization header:

```
Authorization: Bearer <your_token>
```

## Endpoints

### Authentication

#### Login
```http
POST /auth/login
```
Request:
```json
{
    "email": "user@example.com",
    "password": "password123",
    "remember_me": true
}
```
Response:
```json
{
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "user": {
        "id": 1,
        "email": "user@example.com",
        "name": "John Doe"
    }
}
```

#### Register
```http
POST /auth/register
```
Request:
```json
{
    "name": "John Doe",
    "email": "user@example.com",
    "password": "password123"
}
```
Response:
```json
{
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "user": {
        "id": 1,
        "email": "user@example.com",
        "name": "John Doe"
    }
}
```

### Media Processing

#### Upload Media
```http
POST /upload
Content-Type: multipart/form-data
```
Request:
- `file`: Media file (image/video)
- `type`: Media type (image/video)

Response:
```json
{
    "id": "abc123",
    "status": "processing",
    "message": "File uploaded successfully"
}
```

#### Check Processing Status
```http
GET /status/{id}
```
Response:
```json
{
    "id": "abc123",
    "status": "completed",
    "progress": 100,
    "result": {
        "is_deepfake": true,
        "confidence": 0.95,
        "details": {
            "model_used": "Xception",
            "processing_time": "2.5s"
        }
    }
}
```

#### Get Results
```http
GET /results/{id}
```
Response:
```json
{
    "id": "abc123",
    "status": "completed",
    "result": {
        "is_deepfake": true,
        "confidence": 0.95,
        "details": {
            "model_used": "Xception",
            "processing_time": "2.5s",
            "analysis": {
                "face_landmarks": [...],
                "texture_analysis": {...},
                "temporal_analysis": {...}
            }
        }
    }
}
```

## Error Handling

All errors follow this format:
```json
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Human readable error message",
        "details": {}
    }
}
```

Common error codes:
- `AUTH_REQUIRED`: Authentication required
- `INVALID_TOKEN`: Invalid or expired token
- `INVALID_CREDENTIALS`: Invalid email/password
- `FILE_TOO_LARGE`: Uploaded file exceeds size limit
- `INVALID_FILE_TYPE`: Unsupported file type
- `PROCESSING_ERROR`: Error during media processing

## Rate Limiting

- 100 requests per minute per IP
- 1000 requests per hour per user
- Rate limit headers included in responses

## WebSocket API

For real-time updates on processing status:

```javascript
const ws = new WebSocket('wss://api.unmask.com/v1/ws');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    // Handle status updates
};
```

## SDKs

Official SDKs available for:
- Python
- JavaScript/Node.js
- Java
- Go

## Support

For API support:
- Email: support@unmask.com
- Documentation: https://docs.unmask.com
- Status Page: https://status.unmask.com 