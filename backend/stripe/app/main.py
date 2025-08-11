from fastapi import FastAPI
from routes import payments
app=FastAPI()
app.include_router(payments.router)

@app.get('/')
def hi():
    return {'msg':'stripe'}
