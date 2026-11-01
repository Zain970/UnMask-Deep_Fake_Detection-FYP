#!/bin/bash

# Exit on error
set -e

# Wait for PostgreSQL
echo "Waiting for PostgreSQL..."
while ! nc -z db 5432; do
  sleep 0.1
done
echo "PostgreSQL started"

# Wait for Redis
echo "Waiting for Redis..."
while ! nc -z redis 6379; do
  sleep 0.1
done
echo "Redis started"

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Start supervisor
exec "$@" 