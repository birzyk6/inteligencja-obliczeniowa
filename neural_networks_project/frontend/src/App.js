import React, { useState } from "react";
import styled from "styled-components";
import DrawingCanvas from "./components/DrawingCanvas";
import NetworkVisualization from "./components/NetworkVisualization";
import PredictionDisplay from "./components/PredictionDisplay";
import ModelInfo from "./components/ModelInfo";
import Header from "./components/Header";
import { toast } from "react-hot-toast";

const AppContainer = styled.div`
    min-height: 100vh;
    background: #0a0a0a;
    background-image: radial-gradient(at 20% 50%, hsla(214, 100%, 15%, 0.5) 0px, transparent 50%),
        radial-gradient(at 80% 20%, hsla(251, 100%, 15%, 0.5) 0px, transparent 50%),
        radial-gradient(at 40% 40%, hsla(214, 100%, 15%, 0.3) 0px, transparent 50%);
    padding: 20px;
    color: #ffffff;
`;

const MainContent = styled.div`
    max-width: 1400px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 30px;
    margin-top: 20px;

    @media (max-width: 1024px) {
        grid-template-columns: 1fr;
        gap: 20px;
    }
`;

const LeftPanel = styled.div`
    display: flex;
    flex-direction: column;
    gap: 20px;
`;

const RightPanel = styled.div`
    display: flex;
    flex-direction: column;
    gap: 20px;
`;

function App() {
    const [prediction, setPrediction] = useState(null);
    const [visualization, setVisualization] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const handlePrediction = async (imageData) => {
        setIsLoading(true);
        try {
            const API_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

            // Get prediction
            const predictionResponse = await fetch(`${API_URL}/api/predict`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ image: imageData }),
            });

            if (!predictionResponse.ok) {
                throw new Error("Failed to get prediction");
            }

            const predictionData = await predictionResponse.json();
            setPrediction(predictionData);

            // Get visualization
            const vizResponse = await fetch(`${API_URL}/api/visualize`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ image: imageData }),
            });

            if (!vizResponse.ok) {
                throw new Error("Failed to get visualization");
            }

            const vizData = await vizResponse.json();
            setVisualization(vizData);

            toast.success(`Predicted digit: ${predictionData.prediction}`);
        } catch (error) {
            console.error("Error:", error);
            toast.error("Failed to process image. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };

    const handleClear = () => {
        setPrediction(null);
        setVisualization(null);
    };

    return (
        <AppContainer>
            <Header />
            <MainContent>
                <LeftPanel>
                    <DrawingCanvas onPredict={handlePrediction} onClear={handleClear} isLoading={isLoading} />
                    <PredictionDisplay prediction={prediction} />
                    <ModelInfo />
                </LeftPanel>
                <RightPanel>
                    <NetworkVisualization visualization={visualization} isLoading={isLoading} />
                </RightPanel>
            </MainContent>
        </AppContainer>
    );
}

export default App;
