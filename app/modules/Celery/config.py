from celery import Celery

from app.config import ENV_PROJECT
from app.modules.Celery import signals

celery = Celery("tasks", backend=ENV_PROJECT.BACKEND, broker=ENV_PROJECT.BROKER)

# Import the tasks module to register the tasks
from app.modules.Celery import tasks
