# api/fastapi_server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os

class FastAPIServer:
    def __init__(self, config):
        self.config = config
        self.app = FastAPI(
            title="VoicePrint ID API",
            description="Multi-Speaker Recognition System by mwasifanwar",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        self.setup_middleware()
        self.setup_routes()
    
    def setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        from .endpoints import router as api_router
        self.app.include_router(api_router, prefix="/api/v1")
        
        @self.app.get("/")
        async def root():
            return {
                "message": "VoicePrint ID API", 
                "version": "1.0.0",
                "author": "mwasifanwar",
                "endpoints": {
                    "speaker": "/api/v1/speaker",
                    "emotion": "/api/v1/emotion", 
                    "language": "/api/v1/language",
                    "spoof": "/api/v1/spoof",
                    "enhance": "/api/v1/enhance"
                }
            }
        
        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "components": {
                    "speaker_recognizer": "active",
                    "emotion_detector": "active",
                    "language_detector": "active",
                    "anti_spoofing": "active",
                    "voice_enhancer": "active"
                }
            }
    
    def run(self):
        uvicorn.run(
            self.app,
            host=self.config.get('api.host', '0.0.0.0'),
            port=self.config.get('api.port', 8000),
            debug=self.config.get('api.debug', False)
        )