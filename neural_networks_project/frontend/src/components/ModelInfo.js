import React, { useState, useEffect } from "react";
import styled from "styled-components";
import Skeleton from "react-loading-skeleton";
import "react-loading-skeleton/dist/skeleton.css";

const ModelContainer = styled.div`
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
`;

const ModelTitle = styled.h3`
    color: #f8fafc;
    margin-bottom: 15px;
    text-align: center;
    font-weight: 600;
`;

const InfoGrid = styled.div`
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
    margin-bottom: 20px;

    @media (max-width: 480px) {
        grid-template-columns: 1fr;
    }
`;

const InfoCard = styled.div`
    background: #374151;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    border: 1px solid #4b5563;
`;

const InfoLabel = styled.div`
    color: #94a3b8;
    font-size: 0.9rem;
    margin-bottom: 5px;
`;

const InfoValue = styled.div`
    color: #f3f4f6;
    font-size: 1.1rem;
    font-weight: 600;
`;

const LayersContainer = styled.div`
    max-height: 200px;
    overflow-y: auto;
    border: 1px solid #4b5563;
    border-radius: 8px;
    padding: 10px;
    background: #374151;
`;

const LayerItem = styled.div`
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    margin-bottom: 5px;
    background: #4b5563;
    border-radius: 6px;
    font-size: 0.9rem;
`;

const LayerName = styled.span`
    font-weight: 600;
    color: #f3f4f6;
`;

const LayerInfo = styled.span`
    color: #cbd5e1;
    font-size: 0.8rem;
`;

const ErrorMessage = styled.div`
    text-align: center;
    color: #f87171;
    font-style: italic;
    padding: 20px;
`;

const ModelInfo = () => {
    const [modelInfo, setModelInfo] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchModelInfo = async () => {
            try {
                const API_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";
                const response = await fetch(`${API_URL}/api/model-info`);

                if (!response.ok) {
                    throw new Error("Failed to fetch model info");
                }

                const data = await response.json();
                setModelInfo(data);
            } catch (err) {
                setError(err.message);
                console.error("Error fetching model info:", err);
            } finally {
                setLoading(false);
            }
        };

        fetchModelInfo();
    }, []);

    const formatNumber = (num) => {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + "M";
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + "K";
        }
        return num.toString();
    };

    const formatShape = (shape) => {
        if (Array.isArray(shape)) {
            return shape.slice(1).join("×"); // Remove batch dimension
        }
        return "N/A";
    };

    if (loading) {
        return (
            <ModelContainer>
                <ModelTitle>🏗️ Model Architecture</ModelTitle>
                <InfoGrid>
                    <InfoCard>
                        <InfoLabel>Total Parameters</InfoLabel>
                        <InfoValue>
                            <Skeleton width={60} />
                        </InfoValue>
                    </InfoCard>
                    <InfoCard>
                        <InfoLabel>Layers</InfoLabel>
                        <InfoValue>
                            <Skeleton width={40} />
                        </InfoValue>
                    </InfoCard>
                </InfoGrid>
                <Skeleton height={150} />
            </ModelContainer>
        );
    }
    if (error) {
        return (
            <ModelContainer>
                <ModelTitle>Model Architecture</ModelTitle>
                <ErrorMessage>Unable to load model information: {error}</ErrorMessage>
            </ModelContainer>
        );
    }

    return (
        <ModelContainer>
            <ModelTitle>Model Architecture</ModelTitle>

            <InfoGrid>
                <InfoCard>
                    <InfoLabel>Total Parameters</InfoLabel>
                    <InfoValue>{formatNumber(modelInfo.total_params)}</InfoValue>
                </InfoCard>
                <InfoCard>
                    <InfoLabel>Layers</InfoLabel>
                    <InfoValue>{modelInfo.layers.length}</InfoValue>
                </InfoCard>
                <InfoCard>
                    <InfoLabel>Input Shape</InfoLabel>
                    <InfoValue>{formatShape(modelInfo.input_shape)}</InfoValue>
                </InfoCard>
                <InfoCard>
                    <InfoLabel>Output Classes</InfoLabel>
                    <InfoValue>10 (0-9)</InfoValue>
                </InfoCard>
            </InfoGrid>

            <LayersContainer>
                {modelInfo.layers.map((layer, index) => (
                    <LayerItem key={index}>
                        <div>
                            <LayerName>{layer.name}</LayerName>
                            {layer.type && <LayerInfo> ({layer.type})</LayerInfo>}
                        </div>
                        <LayerInfo>
                            {formatShape(layer.output_shape)}
                            {layer.trainable_params > 0 && (
                                <span> • {formatNumber(layer.trainable_params)} params</span>
                            )}
                        </LayerInfo>
                    </LayerItem>
                ))}
            </LayersContainer>
        </ModelContainer>
    );
};

export default ModelInfo;
