import uvicorn
from fastapi import FastAPI

from apps.platform.src.config import ENVIRONMENT, HOST, PORT
from apps.platform.src.modules.route import platform_route

app = FastAPI(
    title='Application: Get your celebrity twin.',
    version='0.0.2',
    docs_url='/docs',
    redoc_url='/redoc'
)

app.include_router(platform_route)

if __name__ == '__main__':
    uvicorn.run(
        'apps.platform.src.app:app',
        host=HOST,
        port=PORT,
        reload=ENVIRONMENT,
        use_colors=True
    )
