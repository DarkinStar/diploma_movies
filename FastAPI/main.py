from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from model import search_similar_movies
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (optional if you have CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Route to serve the HTML page
@app.get("/", response_class=HTMLResponse)
def get_index():
    return Path("static/index.html").read_text(encoding="utf-8")

# Your POST endpoint
class Query(BaseModel):
    text: str

@app.post("/search")
async def search_movies(query: Query):
    results = search_similar_movies(query.text)
    return {"results": results}