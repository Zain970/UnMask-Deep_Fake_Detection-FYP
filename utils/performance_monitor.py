import time
import psutil
import threading
from typing import Dict, Any, Optional
from datetime import datetime
from .logging_config import perf_logger

class PerformanceMonitor:
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self._lock = threading.Lock()

    def start_monitoring(self, operation_name: str) -> None:
        """Start monitoring a specific operation."""
        with self._lock:
            self.start_time = time.time()
            self.metrics[operation_name] = {
                "start_time": datetime.now().isoformat(),
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.Process().memory_info().rss,
                "disk_io": psutil.disk_io_counters(),
                "network_io": psutil.net_io_counters()
            }

    def stop_monitoring(self, operation_name: str) -> Dict[str, Any]:
        """Stop monitoring and return performance metrics."""
        with self._lock:
            if operation_name not in self.metrics:
                return {}

            end_time = time.time()
            duration = end_time - self.start_time

            metrics = self.metrics[operation_name]
            metrics.update({
                "end_time": datetime.now().isoformat(),
                "duration": duration,
                "final_cpu_usage": psutil.cpu_percent(),
                "final_memory_usage": psutil.Process().memory_info().rss,
                "final_disk_io": psutil.disk_io_counters(),
                "final_network_io": psutil.net_io_counters()
            })

            # Calculate deltas
            metrics["cpu_usage_delta"] = metrics["final_cpu_usage"] - metrics["cpu_usage"]
            metrics["memory_usage_delta"] = metrics["final_memory_usage"] - metrics["memory_usage"]
            metrics["disk_io_delta"] = {
                k: getattr(metrics["final_disk_io"], k) - getattr(metrics["disk_io"], k)
                for k in metrics["disk_io"]._fields
            }
            metrics["network_io_delta"] = {
                k: getattr(metrics["final_network_io"], k) - getattr(metrics["network_io"], k)
                for k in metrics["network_io"]._fields
            }

            # Log performance metrics
            perf_logger.info(f"Performance metrics for {operation_name}: {metrics}")

            return metrics

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "network_io": psutil.net_io_counters()._asdict(),
            "timestamp": datetime.now().isoformat()
        }

# Initialize performance monitor
performance_monitor = PerformanceMonitor()

def monitor_performance(operation_name: str):
    """Decorator for monitoring function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            performance_monitor.start_monitoring(operation_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                performance_monitor.stop_monitoring(operation_name)
        return wrapper
    return decorator 