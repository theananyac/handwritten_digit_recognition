import React, { useRef, useState, useEffect } from "react";
import "./App.css";

function App() {
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);

  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [strokeWidth, setStrokeWidth] = useState(15);

  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = 300;
    canvas.height = 300;

    const ctx = canvas.getContext("2d");
    ctx.lineCap = "round";
    ctx.strokeStyle = "white";
    ctx.lineWidth = strokeWidth;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctxRef.current = ctx;
  }, [strokeWidth]);

  const startDrawing = (e) => {
    ctxRef.current.beginPath();
    ctxRef.current.moveTo(
      e.nativeEvent.offsetX,
      e.nativeEvent.offsetY
    );
    setIsDrawing(true);
  };

  const draw = (e) => {
    if (!isDrawing) return;

    ctxRef.current.lineTo(
      e.nativeEvent.offsetX,
      e.nativeEvent.offsetY
    );
    ctxRef.current.stroke();
  };

  const stopDrawing = () => {
    ctxRef.current.closePath();
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    ctxRef.current.fillStyle = "black";
    ctxRef.current.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
    setConfidence(null);
  };

  const predict = async () => {
    const canvas = canvasRef.current;
    const image = canvas.toDataURL("image/png");

    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    });

    const data = await response.json();
    setPrediction(data.digit);
    setConfidence(data.confidence);
  };

  return (
    <div className="container">
      <h1>Handwritten Digit Recognition</h1>
      <p>Draw a digit and click Predict</p>

      <div className="controls">
        <label>Brush Size: {strokeWidth}</label>
        <input
          type="range"
          min="5"
          max="40"
          value={strokeWidth}
          onChange={(e) => setStrokeWidth(e.target.value)}
        />
      </div>

      <canvas
        ref={canvasRef}
        className="drawing-canvas"
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
      ></canvas>

      <div className="buttons">
        <button onClick={predict} className="predict-btn">Predict</button>
        <button onClick={clearCanvas} className="clear-btn">Clear</button>
      </div>

      {prediction !== null && (
        <div className="result">
          <h2>Predicted: {prediction}</h2>
          <p>Confidence: {(confidence * 100).toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
}
export default App;