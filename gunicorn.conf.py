import os

# Binding
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"

# Concurrency Model:
# Single worker process with multiple threads guarantees that thread-safe
# in-memory log streams (_PIPELINE_LOGS) and session states remain synchronized
# without WSGI process-boundary isolation issues.
workers = 1
threads = 4

# Extended timeout (3 minutes) to accommodate slower model queues / free endpoints
timeout = 180

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Preload application to optimize memory usage across threads
preload_app = True
