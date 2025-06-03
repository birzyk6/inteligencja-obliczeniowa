import React, { useRef, useState, useEffect } from "react";
import styled from "styled-components";

const CanvasContainer = styled.div`
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
`;

const CanvasTitle = styled.h3`
    color: #f8fafc;
    margin-bottom: 15px;
    text-align: center;
    font-weight: 600;
`;

const Canvas = styled.canvas`
    border: 2px solid #475569;
    border-radius: 8px;
    cursor: crosshair;
    display: block;
    margin: 0 auto 20px;
    background: #ffffff;
    transition: border-color 0.3s ease;

    &:hover {
        border-color: #60a5fa;
    }
`;

const ButtonContainer = styled.div`
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap;
`;

const Button = styled.button`
    padding: 12px 24px;
    border: none;
    border-radius: 8px;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;

    &:disabled {
        opacity: 0.6;
        cursor: not-allowed;
    }
`;

const PredictButton = styled(Button)`
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    color: white;

    &:hover:not(:disabled) {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }
`;

const ClearButton = styled(Button)`
    background: #374151;
    color: #f3f4f6;
    border: 2px solid #4b5563;

    &:hover:not(:disabled) {
        background: #4b5563;
        border-color: #6b7280;
    }
`;

const Instructions = styled.p`
    text-align: center;
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 15px;
    font-style: italic;
`;

const DrawingCanvas = ({ onPredict, onClear, isLoading }) => {
    const canvasRef = useRef(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [hasDrawing, setHasDrawing] = useState(false);

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        // Set canvas size
        canvas.width = 280;
        canvas.height = 280;

        // Set drawing properties
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 8;

        // Clear canvas
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }, []);

    const startDrawing = (e) => {
        setIsDrawing(true);
        draw(e);
    };

    const draw = (e) => {
        if (!isDrawing) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        const rect = canvas.getBoundingClientRect();

        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);

        setHasDrawing(true);
    };

    const stopDrawing = () => {
        if (!isDrawing) return;

        setIsDrawing(false);
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        ctx.beginPath();
    };

    const clearCanvas = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        setHasDrawing(false);
        onClear();
    };

    const predictDrawing = () => {
        const canvas = canvasRef.current;
        const imageData = canvas.toDataURL("image/png");
        onPredict(imageData);
    };

    // Touch events for mobile
    const handleTouchStart = (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent("mousedown", {
            clientX: touch.clientX,
            clientY: touch.clientY,
        });
        canvas.dispatchEvent(mouseEvent);
    };

    const handleTouchMove = (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent("mousemove", {
            clientX: touch.clientX,
            clientY: touch.clientY,
        });
        canvas.dispatchEvent(mouseEvent);
    };

    const handleTouchEnd = (e) => {
        e.preventDefault();
        const mouseEvent = new MouseEvent("mouseup", {});
        canvas.dispatchEvent(mouseEvent);
    };

    const canvas = canvasRef.current;
    if (canvas) {
        canvas.addEventListener("touchstart", handleTouchStart);
        canvas.addEventListener("touchmove", handleTouchMove);
        canvas.addEventListener("touchend", handleTouchEnd);
    }

    return (
        <CanvasContainer>
            <CanvasTitle>✏️ Draw a Digit (0-9)</CanvasTitle>
            <Instructions>Use your mouse or finger to draw a digit in the box below</Instructions>
            <Canvas
                ref={canvasRef}
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={stopDrawing}
                onMouseOut={stopDrawing}
            />
            <ButtonContainer>
                <PredictButton onClick={predictDrawing} disabled={!hasDrawing || isLoading}>
                    {isLoading ? "🧠 Analyzing..." : "🔍 Predict"}
                </PredictButton>
                <ClearButton onClick={clearCanvas} disabled={isLoading}>
                    🗑️ Clear
                </ClearButton>
            </ButtonContainer>
        </CanvasContainer>
    );
};

export default DrawingCanvas;
