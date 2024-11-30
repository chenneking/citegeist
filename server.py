from fastapi import (
    FastAPI,
    Response
)
from pydantic import BaseModel

class Item(BaseModel):
    abstract: str
    breadth: int
    depth: int
    diversity: int
app = FastAPI()

@app.get("/")
def frontpage():
    with open('./frontend/index.html') as f:
        data = f.read()
    return Response(content=data, media_type="text/html")


@app.post("/generate")
def generate(item: Item):
    # TODO: link this to the logic
    print(item)
    data = {
        'related_works': 'Testing123',
        'citations': [
            'Abraham Lincoln (1910)',
            'Sample Author'
        ]
    }
    return data