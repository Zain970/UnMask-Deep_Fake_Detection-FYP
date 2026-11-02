# Deployment Guide

## Overview

This guide provides detailed instructions for deploying the UnMask application in various environments, from development to production.

## Prerequisites

- Python 3.7+
- Node.js 14.x+
- Docker and Docker Compose
- PostgreSQL 12+
- Redis 6+
- Nginx
- SSL certificates

## Environment Setup

1. Clone the repository:

```bash
git clone https://github.com/BSM21/UNMASK.git
cd UnMask
```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
cd front_end
npm install
```

## Configuration

1. Copy and configure environment files:

```bash
cp .env.example .env
cp deploy/config.yaml.example deploy/config.yaml
```

2. Update configuration files with your settings:

- `.env`: Environment variables
- `deploy/config.yaml`: Application settings

## Database Setup

1. Initialize the database:

```bash
python back_end/scripts/init_db.py
```

2. Run migrations:

```bash
alembic upgrade head
```

## Development Deployment

1. Start the backend:

```bash
cd back_end
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

2. Start the frontend:

```bash
cd front_end
npm run dev
```

## Production Deployment

### Docker Deployment

1. Build the Docker images:

```bash
docker-compose build
```

2. Start the services:

```bash
docker-compose up -d
```

### Manual Deployment

1. Set up Nginx:

```nginx
server {
    listen 80;
    server_name api.unmask.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

server {
    listen 80;
    server_name app.unmask.com;

    location / {
        root /path/to/front_end/dist;
        try_files $uri $uri/ /index.html;
    }
}
```

2. Configure SSL:

```bash
certbot --nginx -d api.unmask.com -d app.unmask.com
```

3. Start the application:

```bash
# Backend
gunicorn back_end.app:app -w 4 -k uvicorn.workers.UvicornWorker

# Frontend
cd front_end
npm run build
```

## Monitoring and Logging

1. Set up Prometheus:

```yaml
scrape_configs:
  - job_name: "unmask"
    static_configs:
      - targets: ["localhost:9090"]
```

2. Configure logging:

```bash
mkdir -p /var/log/unmask
chown -R unmask:unmask /var/log/unmask
```

## Backup and Recovery

1. Set up automated backups:

```bash
# Add to crontab
0 0 * * * /path/to/backup_script.sh
```

2. Test recovery:

```bash
python back_end/scripts/restore_backup.py latest
```

## Scaling

### Horizontal Scaling

1. Set up load balancer:

```nginx
upstream backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}
```

2. Configure Redis for session storage:

```python
SESSION_TYPE = 'redis'
SESSION_REDIS = redis.from_url('redis://localhost:6379/0')
```

### Vertical Scaling

1. Optimize database:

```sql
ALTER SYSTEM SET max_connections = '200';
ALTER SYSTEM SET shared_buffers = '4GB';
```

2. Configure worker processes:

```bash
gunicorn back_end.app:app -w 8 -k uvicorn.workers.UvicornWorker
```

## Security Considerations

1. Update firewall rules:

```bash
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable
```

2. Configure security headers:

```nginx
add_header X-Frame-Options "SAMEORIGIN";
add_header X-XSS-Protection "1; mode=block";
add_header X-Content-Type-Options "nosniff";
```

## Troubleshooting

Common issues and solutions:

1. Database connection errors:

   - Check database credentials
   - Verify network connectivity
   - Ensure database is running

2. Performance issues:

   - Monitor system resources
   - Check database indexes
   - Review query performance

3. SSL/TLS issues:
   - Verify certificate validity
   - Check private key permissions
   - Test SSL configuration

## Maintenance

Regular maintenance tasks:

1. Update dependencies:

```bash
pip install -r requirements.txt --upgrade
cd front_end && npm update
```

2. Database maintenance:

```sql
VACUUM ANALYZE;
REINDEX DATABASE unmask;
```

3. Log rotation:

```bash
logrotate /etc/logrotate.d/unmask
```

## Support

For deployment support:

- Email: bilal.saeed.malik123@gmail.com or zainulabidin970946@gmail.com
- GitHub Issues: https://github.com/Zain970/UnMask-Deep_Fake_Detection-FYP.git/issues
