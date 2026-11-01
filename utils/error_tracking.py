import traceback
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from .logging_config import logger

class ErrorTracker:
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.last_error_time: Dict[str, datetime] = {}

    def track_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Track and log an error with context."""
        error_type = type(error).__name__
        error_message = str(error)
        error_key = f"{error_type}:{error_message}"

        # Update error counts
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        self.last_error_time[error_key] = datetime.now()

        # Log error details
        error_info = {
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
            "count": self.error_counts[error_key],
            "last_occurrence": self.last_error_time[error_key].isoformat(),
            "traceback": traceback.format_exc()
        }

        logger.error(f"Error occurred: {error_info}")

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "unique_errors": len(self.error_counts),
            "error_counts": self.error_counts,
            "last_error_times": {
                k: v.isoformat() for k, v in self.last_error_time.items()
            }
        }

    def clear_errors(self) -> None:
        """Clear all tracked errors."""
        self.error_counts.clear()
        self.last_error_time.clear()

# Initialize error tracker
error_tracker = ErrorTracker()

def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Global error handling function."""
    error_tracker.track_error(error, context)
    
    # Additional error handling logic can be added here
    # For example, sending error reports to external services 