import pickle
import numpy as np
from pydantic import BaseModel

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import numpy
import onnxruntime as rt

import uvicorn
from fastapi import FastAPI


app = FastAPI()
feature = ['Frequency','Recency','Lifetime', 'MonetaryValue']
print("features:", feature)
session = rt.InferenceSession("km_user.onnx")
first_input_name = session.get_inputs()[0].name
print("input:", first_input_name)
first_output_name = session.get_outputs()[0].name
print("output:", first_output_name)



class Data(BaseModel):
    Frequency : float
    Recency: float
    Lifetime: float
    MonetaryValue: float



@app.post("/predict")
def predict(data: Data):
    try:
        # Extract data in correct order
        data_dict = data.dict()
        print("dictionary:", data_dict)
        to_predict = [data_dict[feat] for feat in feature]
        to_predict = np.array(to_predict).reshape(1, -1)
        print("array:", to_predict)
        pred_onx = session.run([], {first_input_name: to_predict.astype(numpy.float32)})[0]

        print("prediction", pred_onx)

        return {"prediction": float(pred_onx[0])}

    except:
        return {"prediction": "error"}

#uvicorn main:app
