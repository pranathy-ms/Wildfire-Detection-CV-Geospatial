"""
Web Interface - FastAPI backend for wildfire prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import sqlite3
import config
from prediction import FirePredictor
from datetime import datetime, timedelta

app = FastAPI(title="Wildfire Spread Prediction")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize predictor
predictor = None

class PredictionRequest(BaseModel):
    date: str

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main page"""
    with open("static/index.html", "r") as f:
        return f.read()

@app.get("/api/health")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/available_dates")
async def get_available_dates():
    """Get dates that have fire data"""
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT date FROM fires ORDER BY date")
        dates = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return {
            "dates": dates,
            "count": len(dates)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/fires/{date}")
async def get_fires(date: str):
    """Get fire locations for a specific date"""
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT lat, lon, frp, brightness, confidence
            FROM fires
            WHERE date = ?
        """, (date,))
        
        fires = []
        for row in cursor.fetchall():
            fires.append({
                "lat": row[0],
                "lon": row[1],
                "frp": row[2],
                "brightness": row[3],
                "confidence": row[4]
            })
        
        conn.close()
        
        return {
            "date": date,
            "fires": fires,
            "count": len(fires)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict_spread(request: PredictionRequest):
    """Make fire spread prediction"""
    global predictor
    
    try:
        # Initialize predictor if needed
        if predictor is None:
            predictor = FirePredictor()
        
        # Make prediction
        predictions = predictor.predict(request.date)
        
        if not predictions:
            raise HTTPException(status_code=404, detail=f"No fires found for {request.date}")
        
        # Format response
        high_risk = [p for p in predictions if p['spread_probability'] > 0.7]
        medium_risk = [p for p in predictions if 0.5 < p['spread_probability'] <= 0.7]
        low_risk = [p for p in predictions if 0.3 < p['spread_probability'] <= 0.5]
        
        return {
            "date": request.date,
            "summary": {
                "total_predictions": len(predictions),
                "high_risk": len(high_risk),
                "medium_risk": len(medium_risk),
                "low_risk": len(low_risk),
                "overall_risk": float(sum(p['spread_probability'] for p in predictions) / len(predictions))
            },
            "predictions": {
                "high": high_risk,
                "medium": medium_risk,
                "low": low_risk
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model_info")
async def get_model_info():
    """Get model information"""
    try:
        import torch
        model_path = config.DATA_DIR / "models" / "trained_model.pth"
        
        if not model_path.exists():
            return {"error": "Model not found"}
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        return {
            "training_f1": float(checkpoint['history']['val_f1'][-1]),
            "training_precision": float(checkpoint['history']['val_precision'][-1]),
            "training_recall": float(checkpoint['history']['val_recall'][-1]),
            "model_config": checkpoint['config']
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print(" WILDFIRE SPREAD PREDICTION - WEB INTERFACE")
    print("=" * 60)
    print("\nStarting server...")
    print("Open your browser to: http://localhost:8000")
    print("\nPress CTRL+C to stop the server")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)