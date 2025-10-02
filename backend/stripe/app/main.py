# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import payments , cotizar , cotizar_colombia
import os

app = FastAPI(title="Stripe Payments API")

# CORS (ajusta tus orígenes del front)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas
app.include_router(payments.router)
app.include_router(cotizar.router)
app.include_router(cotizar_colombia.router)


# @app.get("/")
# def root(data):
#     print(data)
#     return {"ok": True}


@app.post("/webhook")
def root(data:dict):
    print(data)
    return {"ok": True}