import React, { useState } from "react";
import styled from "styled-components";
import Skeleton from "react-loading-skeleton";

const VisualizationContainer = styled.div`
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    min-height: 500px;
`;

const VisualizationTitle = styled.h3`
    color: #f8fafc;
    margin-bottom: 20px;
    text-align: center;
    font-weight: 600;
`;

const LayerTabs = styled.div`
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-bottom: 20px;
    border-bottom: 2px solid #374151;
    padding-bottom: 10px;
`;

const LayerTab = styled.button`
    padding: 8px 16px;
    border: none;
    background: ${(props) => (props.active ? "linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)" : "#374151")};
    color: ${(props) => (props.active ? "white" : "#cbd5e1")};
    border-radius: 20px;
    cursor: pointer;
    font-size: 0.9rem;
    font-weight: 500;
    transition: all 0.3s ease;

    &:hover {
        background: ${(props) => (props.active ? "linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)" : "#4b5563")};
        transform: translateY(-1px);
    }
`;

const LayerContent = styled.div`
    display: flex;
    flex-direction: column;
    gap: 20px;
`;

const StatsGrid = styled.div`
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 15px;
    margin-bottom: 20px;
`;

const StatCard = styled.div`
    background: #374151;
    padding: 12px;
    border-radius: 8px;
    text-align: center;
    border: 1px solid #4b5563;
`;

const StatLabel = styled.div`
    color: #94a3b8;
    font-size: 0.8rem;
    margin-bottom: 5px;
`;

const StatValue = styled.div`
    color: #f3f4f6;
    font-size: 1rem;
    font-weight: 600;
`;

const VisualizationImage = styled.img`
    width: 100%;
    max-height: 400px;
    object-fit: contain;
    border-radius: 10px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
`;

const NoDataMessage = styled.div`
    text-align: center;
    color: #94a3b8;
    font-style: italic;
    padding: 60px 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 15px;
`;

const InstructionText = styled.p`
    color: #cbd5e1;
    margin: 0;
    line-height: 1.5;
`;

const LoadingContainer = styled.div`
    display: flex;
    flex-direction: column;
    gap: 20px;
    padding: 20px 0;
`;

const NetworkVisualization = ({ visualization, isLoading }) => {
    const [activeLayer, setActiveLayer] = useState(0);
    if (isLoading) {
        return (
            <VisualizationContainer>
                <VisualizationTitle>Neural Network Visualization</VisualizationTitle>
                <LoadingContainer>
                    <Skeleton height={40} />
                    <Skeleton height={200} />
                    <Skeleton height={300} />
                </LoadingContainer>
            </VisualizationContainer>
        );
    }

    if (!visualization) {
        return (
            <VisualizationContainer>
                <VisualizationTitle>Neural Network Visualization</VisualizationTitle>
                <NoDataMessage>
                    <InstructionText>
                        <strong>How it works:</strong>
                        <br />
                        1. Draw a digit (0-9) in the canvas on the left
                        <br />
                        2. Click "Predict" to see the magic happen
                        <br />
                        3. Watch how each layer processes your drawing
                        <br />
                        4. Explore different layers using the tabs that will appear here
                    </InstructionText>
                </NoDataMessage>
            </VisualizationContainer>
        );
    }

    const currentLayer = visualization.layers[activeLayer];

    const formatNumber = (num) => {
        if (typeof num !== "number") return "N/A";
        if (num < 0.001 && num > 0) return num.toExponential(2);
        if (num >= 1000) return num.toFixed(0);
        return num.toFixed(3);
    };
    return (
        <VisualizationContainer>
            <VisualizationTitle>Neural Network Visualization</VisualizationTitle>

            {visualization.prediction && (
                <div
                    style={{
                        background: "#374151",
                        padding: "15px",
                        borderRadius: "10px",
                        marginBottom: "20px",
                        textAlign: "center",
                        border: "2px solid #60a5fa",
                    }}
                >
                    <strong style={{ color: "#f3f4f6" }}>Final Prediction: </strong>
                    <span
                        style={{
                            fontSize: "1.5rem",
                            color: "#60a5fa",
                            fontWeight: "bold",
                        }}
                    >
                        {visualization.prediction.digit}
                    </span>
                    <span style={{ color: "#cbd5e1", marginLeft: "10px" }}>
                        ({(visualization.prediction.confidence * 100).toFixed(1)}% confidence)
                    </span>
                </div>
            )}

            <LayerTabs>
                {visualization.layers.map((layer, index) => (
                    <LayerTab key={index} active={activeLayer === index} onClick={() => setActiveLayer(index)}>
                        {layer.name} ({layer.type})
                    </LayerTab>
                ))}
            </LayerTabs>

            {currentLayer && (
                <LayerContent>
                    <StatsGrid>
                        <StatCard>
                            <StatLabel>Layer Type</StatLabel>
                            <StatValue>{currentLayer.type}</StatValue>
                        </StatCard>
                        <StatCard>
                            <StatLabel>Output Shape</StatLabel>
                            <StatValue>{currentLayer.shape.slice(1).join("×")}</StatValue>
                        </StatCard>
                        <StatCard>
                            <StatLabel>Mean Activation</StatLabel>
                            <StatValue>{formatNumber(currentLayer.activation_stats.mean)}</StatValue>
                        </StatCard>
                        <StatCard>
                            <StatLabel>Max Activation</StatLabel>
                            <StatValue>{formatNumber(currentLayer.activation_stats.max)}</StatValue>
                        </StatCard>
                        <StatCard>
                            <StatLabel>Active Neurons</StatLabel>
                            <StatValue>
                                {currentLayer.activation_stats.total - currentLayer.activation_stats.zeros}
                            </StatValue>
                        </StatCard>
                        <StatCard>
                            <StatLabel>Sparsity</StatLabel>
                            <StatValue>
                                {(
                                    (currentLayer.activation_stats.zeros / currentLayer.activation_stats.total) *
                                    100
                                ).toFixed(1)}
                                %
                            </StatValue>
                        </StatCard>
                    </StatsGrid>{" "}
                    {currentLayer.visualization && currentLayer.visualization.image && (
                        <div>
                            <h4 style={{ color: "#f8fafc", marginBottom: "10px" }}>Layer Visualization:</h4>
                            <VisualizationImage
                                src={currentLayer.visualization.image}
                                alt={`${currentLayer.name} visualization`}
                            />

                            {currentLayer.visualization.stats && (
                                <div
                                    style={{
                                        marginTop: "15px",
                                        padding: "10px",
                                        background: "#374151",
                                        borderRadius: "8px",
                                        fontSize: "0.9rem",
                                        color: "#cbd5e1",
                                        border: "1px solid #4b5563",
                                    }}
                                >
                                    <strong style={{ color: "#f3f4f6" }}>Visualization Stats:</strong>
                                    <ul style={{ margin: "5px 0", paddingLeft: "20px" }}>
                                        {Object.entries(currentLayer.visualization.stats).map(([key, value]) => (
                                            <li key={key}>
                                                {key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}:{" "}
                                                {value}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </div>
                    )}
                </LayerContent>
            )}
        </VisualizationContainer>
    );
};

export default NetworkVisualization;
