from fastapi import FastAPI, CORSMiddleware
from pydantic import BaseModel
import torch
from app.model import LogisticRegression

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

input_dim = 4
output_dim = 3
model = LogisticRegression(input_dim, output_dim)
model.load_state_dict(torch.load("app/model.pt"))
model.eval()


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
def predict(input: IrisInput):
    features = [[
        input.sepal_length,
        input.sepal_width,
        input.petal_length,
        input.petal_width
    ]]
    tensor = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        logits = model(tensor)
        predicted = torch.argmax(logits, dim=1).item()
    return {"predicted_class": predicted}
