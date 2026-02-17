import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Union
import joblib
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware

MODEL_FILE = 'best_consumption_model.pkl'
DATA_FILE = 'processed_consumption_data.csv'

class DataStore:
    def __init__(self):
        self.df_data = self._load_data()
        self.ml_model = self._load_model()
        
    def _load_data(self) -> pd.DataFrame:
        if os.path.exists(DATA_FILE):
            try:
                df = pd.read_csv(DATA_FILE)
                if 'RecordID' in df.columns:
                    df.set_index('RecordID', inplace=True)
                    return df
            except Exception as e:
                print(f"ERROR: Failed to load data file: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    def _load_model(self):
        if os.path.exists(MODEL_FILE):
            return joblib.load(MODEL_FILE)
        else:
            print(f"ERROR: {MODEL_FILE} file not found. Prediction endpoint will not work.")
            return None

    def save_data(self):
        if not self.df_data.empty:
            self.df_data.reset_index().to_csv(DATA_FILE, index=False)
            
    def get_next_record_id(self) -> int:
        if self.df_data.empty or self.df_data.index.empty:
            return 1
        
        if self.df_data.index.dtype == np.int64:
            return self.df_data.index.max() + 1
        else:
            return self.df_data.reset_index()['RecordID'].max() + 1

app_state = DataStore()


class ConsumptionRecord(BaseModel):
    Hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    DayOfWeek: int = Field(..., ge=0, le=6, description="Day of the week (0=Mon, 6=Sun)")
    DayOfMonth: int = Field(..., ge=1, le=31, description="Day of the month (1-31)")
    Consumption: float = Field(..., ge=0, description="Measured consumption value")

class ConsumptionRecordOut(ConsumptionRecord):
    RecordID: int

class PredictionInput(BaseModel):
    Hour: int = Field(..., ge=0, le=23)
    DayOfWeek: int = Field(..., ge=0, le=6)
    DayOfMonth: int = Field(..., ge=1, le=31)

app = FastAPI(
    title="Smart Utility Consumption Forecasting API",
    description="A service for consumption data management (CRUD) and forecasting using a Random Forest Regressor."
)

origins = [
    "*", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

@app.get("/records", response_model=List[ConsumptionRecordOut], summary="Retrieve all records (READ ALL)")
async def get_all_records():
    df = app_state.df_data
    if df.empty:
        return []
        
    records = []
    for index, row in df.iterrows():
        records.append(ConsumptionRecordOut(RecordID=index, **row.to_dict()))
    return records

@app.get("/records/{record_id}", response_model=ConsumptionRecordOut, summary="Retrieve a specific record (READ ONE)")
async def get_record(record_id: int):
    df = app_state.df_data
    if record_id not in df.index:
        raise HTTPException(status_code=404, detail=f"RecordID {record_id} not found.")
    
    record = df.loc[record_id].to_dict()
    return ConsumptionRecordOut(RecordID=record_id, **record)

@app.post("/records", response_model=ConsumptionRecordOut, status_code=201, summary="Add a new record (CREATE)")
async def add_record(record: ConsumptionRecord):
    
    df = app_state.df_data
    new_id = app_state.get_next_record_id()
    new_record_data = record.model_dump()
    
    new_record_series = pd.Series(new_record_data, name=new_id)
    
    if df.empty:
        app_state.df_data = pd.DataFrame([new_record_data], index=[new_id])
        app_state.df_data.index.name = 'RecordID'
    else:
        app_state.df_data = pd.concat([df, new_record_series.to_frame().T])
        app_state.df_data.index.name = 'RecordID'

    app_state.save_data() 
    
    return ConsumptionRecordOut(RecordID=new_id, **new_record_data)

@app.put("/records/{record_id}", response_model=ConsumptionRecordOut, summary="Update an existing record (UPDATE)")
async def update_record(record_id: int, record: ConsumptionRecord):
    df = app_state.df_data
    if record_id not in df.index:
        raise HTTPException(status_code=404, detail=f"RecordID {record_id} not found.")

    update_data = record.model_dump()
    
    app_state.df_data.loc[record_id] = pd.Series(update_data)

    app_state.save_data() 

    return ConsumptionRecordOut(RecordID=record_id, **app_state.df_data.loc[record_id].to_dict())

@app.delete("/records/{record_id}", status_code=204, summary="Remove a record (DELETE)")
async def delete_record(record_id: int):
    df = app_state.df_data
    if record_id not in df.index:
        raise HTTPException(status_code=404, detail=f"RecordID {record_id} not found.")

    app_state.df_data.drop(index=record_id, inplace=True)

    app_state.save_data() 
    
    return 

@app.post("/predict_consumption", summary="Predict consumption using the best ML model (PREDICT)")
async def predict_consumption(features: PredictionInput):
    
    model = app_state.ml_model
    if model is None:
        raise HTTPException(status_code=500, detail="ML Model not loaded. Check training stage and server logs.")

    input_data = np.array([
        features.Hour,
        features.DayOfWeek,
        features.DayOfMonth
    ]).reshape(1, -1)

    prediction = model.predict(input_data)[0]

    return {
        "Hour": features.Hour,
        "DayOfWeek": features.DayOfWeek,
        "DayOfMonth": features.DayOfMonth,
        "Predicted_Consumption": round(float(prediction), 4),
        "Model_Used": "Random Forest Regressor"
    }