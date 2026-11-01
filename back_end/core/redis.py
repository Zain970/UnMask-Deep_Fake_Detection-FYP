import redis
from ..config import settings

redis_client = None

async def init_redis():
    global redis_client
    redis_client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        decode_responses=True
    )
    
    # Test connection
    try:
        redis_client.ping()
    except redis.ConnectionError:
        raise Exception("Failed to connect to Redis server")

def get_redis():
    if redis_client is None:
        raise Exception("Redis client not initialized")
    return redis_client 