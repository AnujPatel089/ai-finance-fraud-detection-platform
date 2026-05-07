from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():
    return {"message": "working"}


@app.get("/test-shap")
def test_shap():
    return {"message": "TEST SHAP ROUTE WORKING"}