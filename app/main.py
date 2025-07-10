from fastapi import FastAPI
from app.controllers.predict import router

app = FastAPI()

#Controller predict
app.include_router(router)