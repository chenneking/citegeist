from fastapi import FastAPI, Response
from pydantic import BaseModel
from generator import generate_related_work


class Item(BaseModel):
    abstract: str
    breadth: int
    depth: int
    diversity: int


app = FastAPI()


@app.get("/")
def frontpage():
    with open("./frontend/index.html") as f:
        data = f.read()
    return Response(content=data, media_type="text/html")


@app.post("/generate")
def generate(item: Item):
    return generate_related_work(
        item.abstract,
        item.breadth,
        item.depth,
        item.diversity
    )
