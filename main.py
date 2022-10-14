
import uvicorn
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel

from prediction import predict_txt

app = FastAPI()

class Item(BaseModel):
    text: str

@app.get('/')
async def home():
    return {'API' : 'SMS Classification API'}

@app.post('/predict')
async def predict(data: Item):
    text = data.text
    output = predict_txt(text)
    return {'status' : 'success', 'prediction' : output}

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="SMS Classification API",
        version="1.0.0",
        description="This is a simple API for SMS Classification",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

def main():
    uvicorn.run(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    main()
