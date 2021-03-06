from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from counter import Counter

counter = Counter()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
    "application_name": "Person counter (Fast API)",
    "author": "Maxime MOREILLON",
    }


@app.post("/predict")
async def predict(image: UploadFile = File (...)):
    result = await counter.predict(image)
    return result
