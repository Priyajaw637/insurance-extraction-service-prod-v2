from fastapi import APIRouter

from app.routes.insurance_commercial import insurance_commercial
from app.routes.test import test_router

routers = APIRouter()

routers.include_router(insurance_commercial, prefix="/commercial", tags=["Commercial"])
routers.include_router(test_router, prefix="/test", tags=["Test"])
