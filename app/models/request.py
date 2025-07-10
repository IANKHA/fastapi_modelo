from pydantic import BaseModel
from typing import List

class Predict(BaseModel):
    image:List[List[float]]
    model:str