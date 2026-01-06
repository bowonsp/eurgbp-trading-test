"""
MOCK API SERVER - Quick Start Version
This uses simple technical analysis instead of AI for testing
Deploy this FIRST to test your system, then upgrade to real AI later

Deploy to Render: This will work immediately without model files!
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
import numpy as np

app = FastAPI(title="EURGBP Mock API - Quick Start", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MarketData(BaseModel):
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float]
    timestamp: Optional[str] = None

class PredictionResponse(BaseModel):
    signal: str
    confidence: float
    probabilities: dict
    risk_mode: str
    timestamp: str
    features_used: int
    method: str

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    prices = np.array(prices)
    deltas = np.diff(prices)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:]) if len(gains) >= period else np.mean(gains)
    avg_loss = np.mean(losses[-period:]) if len(losses) >= period else np.mean(losses)
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices):
    """Calculate MACD"""
    prices = np.array(prices)
    
    # EMA 12 and 26
    ema_12 = prices[-12:].mean() if len(prices) >= 12 else prices.mean()
    ema_26 = prices[-26:].mean() if len(prices) >= 26 else prices.mean()
    
    macd = ema_12 - ema_26
    return macd

def calculate_moving_averages(prices):
    """Calculate SMA"""
    prices = np.array(prices)
    
    sma_20 = prices[-20:].mean() if len(prices) >= 20 else prices.mean()
    sma_50 = prices[-50:].mean() if len(prices) >= 50 else prices.mean()
    
    return sma_20, sma_50

def simple_prediction(data: MarketData, risk_mode: str):
    """
    Simple technical analysis based prediction
    This mimics AI behavior for testing
    """
    
    closes = np.array(data.close)
    
    # Calculate indicators
    rsi = calculate_rsi(closes)
    macd = calculate_macd(closes)
    sma_20, sma_50 = calculate_moving_averages(closes)
    current_price = closes[-1]
    
    # Trend
    trend_up = sma_20 > sma_50
    trend_down = sma_20 < sma_50
    
    # Signals
    signal = "HOLD"
    confidence = 0.5
    
    # Conservative logic (stricter)
    if risk_mode == "conservative":
        if rsi < 30 and trend_up and macd > 0:
            signal = "BUY"
            confidence = 0.70 + (30 - rsi) / 100  # Higher confidence if more oversold
        elif rsi > 70 and trend_down and macd < 0:
            signal = "SELL"
            confidence = 0.70 + (rsi - 70) / 100
        elif 30 <= rsi <= 50 and trend_up and current_price > sma_20:
            signal = "BUY"
            confidence = 0.65
        elif 50 <= rsi <= 70 and trend_down and current_price < sma_20:
            signal = "SELL"
            confidence = 0.65
    
    # Moderate logic (more relaxed)
    else:  # moderate
        if rsi < 35 and macd > 0:
            signal = "BUY"
            confidence = 0.65 + (35 - rsi) / 100
        elif rsi > 65 and macd < 0:
            signal = "SELL"
            confidence = 0.65 + (rsi - 65) / 100
        elif 35 <= rsi <= 55 and trend_up:
            signal = "BUY"
            confidence = 0.60
        elif 45 <= rsi <= 65 and trend_down:
            signal = "SELL"
            confidence = 0.60
    
    # Cap confidence
    confidence = min(confidence, 0.95)
    
    # Calculate probabilities
    if signal == "BUY":
        prob_buy = confidence
        prob_sell = (1 - confidence) * 0.3
        prob_hold = 1 - prob_buy - prob_sell
    elif signal == "SELL":
        prob_sell = confidence
        prob_buy = (1 - confidence) * 0.3
        prob_hold = 1 - prob_sell - prob_buy
    else:  # HOLD
        prob_hold = confidence
        prob_buy = (1 - confidence) * 0.5
        prob_sell = (1 - confidence) * 0.5
    
    return {
        "signal": signal,
        "confidence": confidence,
        "probabilities": {
            "HOLD": float(prob_hold),
            "BUY": float(prob_buy),
            "SELL": float(prob_sell)
        },
        "debug": {
            "rsi": float(rsi),
            "macd": float(macd),
            "sma_20": float(sma_20),
            "sma_50": float(sma_50),
            "trend": "UP" if trend_up else "DOWN"
        }
    }

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "mode": "MOCK API - Technical Analysis",
        "message": "This is a test version. Upgrade to AI version later.",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health():
    """Health endpoint"""
    return {
        "status": "healthy",
        "models_loaded": ["conservative_mock", "moderate_mock"],
        "mode": "technical_analysis",
        "uptime": "running",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict/{risk_mode}", response_model=PredictionResponse)
async def predict(risk_mode: str, data: MarketData):
    """
    Get prediction based on technical analysis
    """
    
    if risk_mode not in ["conservative", "moderate"]:
        raise HTTPException(
            status_code=400, 
            detail="risk_mode must be 'conservative' or 'moderate'"
        )
    
    if len(data.close) < 50:
        raise HTTPException(
            status_code=400,
            detail="Need at least 50 candles for analysis"
        )
    
    try:
        # Get prediction
        result = simple_prediction(data, risk_mode)
        
        return PredictionResponse(
            signal=result["signal"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            risk_mode=risk_mode,
            timestamp=datetime.utcnow().isoformat(),
            features_used=len(data.close),
            method="technical_analysis"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/model/info/{risk_mode}")
async def model_info(risk_mode: str):
    """Get model info"""
    return {
        "symbol": "EURGBP",
        "timeframe": "H1",
        "risk_mode": risk_mode,
        "method": "technical_analysis",
        "indicators": ["RSI", "MACD", "SMA"],
        "note": "This is MOCK version for testing. Upgrade to AI later."
    }

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
