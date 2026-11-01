# UnMask API Usage Guide

## Table of Contents
1. [Authentication](#authentication)
2. [File Upload](#file-upload)
3. [Processing](#processing)
4. [Results](#results)
5. [Error Handling](#error-handling)

## Authentication

### Login
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=user@example.com&password=password123"
```

Response:
```json
{
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "expires_in": 1800
}
```

### Register
```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
     -H "Content-Type: application/json" \
     -d '{
         "email": "user@example.com",
         "password": "password123",
         "full_name": "John Doe"
     }'
```

## File Upload

### Upload Image
```bash
curl -X POST "http://localhost:8000/api/v1/files/upload/image" \
     -H "Authorization: Bearer <token>" \
     -F "file=@/path/to/image.jpg"
```

### Upload Video
```bash
curl -X POST "http://localhost:8000/api/v1/files/upload/video" \
     -H "Authorization: Bearer <token>" \
     -F "file=@/path/to/video.mp4"
```

## Processing

### Start Processing
```bash
curl -X POST "http://localhost:8000/api/v1/processing/process" \
     -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     -d '{
         "file_path": "/uploads/images/example.jpg",
         "options": {
             "model": "ensemble",
             "confidence_threshold": 0.8
         }
     }'
```

### Check Status
```bash
curl -X GET "http://localhost:8000/api/v1/processing/status/{processing_id}" \
     -H "Authorization: Bearer <token>"
```

### Get Results
```bash
curl -X GET "http://localhost:8000/api/v1/processing/result/{processing_id}" \
     -H "Authorization: Bearer <token>"
```

## Results

### Processing Status
```json
{
    "status": "processing",
    "started_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:01:00Z",
    "file_path": "/uploads/images/example.jpg",
    "user_id": 1,
    "progress": 50
}
```

### Processing Result
```json
{
    "processing_id": "1_1704067200",
    "status": "completed",
    "result": {
        "file_path": "/uploads/images/example.jpg",
        "processed_at": "2024-01-01T00:02:00Z",
        "confidence": 0.95,
        "is_deepfake": true
    },
    "started_at": "2024-01-01T00:00:00Z",
    "completed_at": "2024-01-01T00:02:00Z"
}
```

## Error Handling

### Error Response Format
```json
{
    "detail": "Error message description"
}
```

### Common Error Codes
- 400: Bad Request - Invalid input data
- 401: Unauthorized - Missing or invalid authentication
- 403: Forbidden - Insufficient permissions
- 404: Not Found - Resource not found
- 500: Internal Server Error - Server-side error

## Best Practices

1. Always include the Authorization header with your JWT token
2. Handle rate limiting (429 responses) with exponential backoff
3. Check processing status before requesting results
4. Implement proper error handling for all API calls
5. Use appropriate content types for file uploads
6. Monitor API response times and implement timeouts
7. Cache results when appropriate
8. Implement retry logic for failed requests 