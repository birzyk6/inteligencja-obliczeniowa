import React from "react";
import styled from "styled-components";
import { Bar } from "react-chartjs-2";
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const PredictionContainer = styled.div`
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
`;

const PredictionTitle = styled.h3`
    color: #f8fafc;
    margin-bottom: 15px;
    text-align: center;
    font-weight: 600;
`;

const ResultDisplay = styled.div`
    text-align: center;
    margin-bottom: 20px;
`;

const PredictedDigit = styled.div`
    font-size: 4rem;
    font-weight: bold;
    color: #60a5fa;
    margin-bottom: 10px;
`;

const Confidence = styled.div`
    font-size: 1.2rem;
    color: #cbd5e1;
    margin-bottom: 5px;
`;

const ConfidenceBar = styled.div`
    width: 100%;
    height: 20px;
    background: #374151;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 20px;
`;

const ConfidenceLevel = styled.div`
    height: 100%;
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    border-radius: 10px;
    transition: width 0.3s ease;
    width: ${(props) => props.confidence * 100}%;
`;

const ChartContainer = styled.div`
    height: 200px;
    margin-top: 20px;
`;

const NoResultsMessage = styled.div`
    text-align: center;
    color: #94a3b8;
    font-style: italic;
    padding: 40px 20px;
`;

const PredictionDisplay = ({ prediction }) => {
    if (!prediction) {
        return (
            <PredictionContainer>
                <PredictionTitle>Prediction Results</PredictionTitle>
                <NoResultsMessage>Draw a digit above and click "Predict" to see the results!</NoResultsMessage>
            </PredictionContainer>
        );
    }

    const chartData = {
        labels: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        datasets: [
            {
                label: "Probability",
                data: prediction.probabilities,
                backgroundColor: prediction.probabilities.map((prob, index) =>
                    index === prediction.prediction ? "rgba(59, 130, 246, 0.8)" : "rgba(59, 130, 246, 0.3)"
                ),
                borderColor: prediction.probabilities.map((prob, index) =>
                    index === prediction.prediction ? "rgba(59, 130, 246, 1)" : "rgba(59, 130, 246, 0.5)"
                ),
                borderWidth: 2,
            },
        ],
    };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false,
            },
            title: {
                display: true,
                text: "Probability Distribution",
                color: "#f8fafc",
            },
            tooltip: {
                backgroundColor: "rgba(30, 41, 59, 0.9)",
                titleColor: "#f8fafc",
                bodyColor: "#e2e8f0",
                callbacks: {
                    label: function (context) {
                        return `Digit ${context.label}: ${(context.parsed.y * 100).toFixed(2)}%`;
                    },
                },
            },
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 1,
                ticks: {
                    color: "#94a3b8",
                    callback: function (value) {
                        return (value * 100).toFixed(0) + "%";
                    },
                },
                grid: {
                    color: "rgba(148, 163, 184, 0.2)",
                },
            },
            x: {
                ticks: {
                    color: "#94a3b8",
                },
                grid: {
                    display: false,
                },
            },
        },
    };
    return (
        <PredictionContainer>
            <PredictionTitle>Prediction Results</PredictionTitle>

            <ResultDisplay>
                <PredictedDigit>{prediction.prediction}</PredictedDigit>
                <Confidence>Confidence: {(prediction.confidence * 100).toFixed(2)}%</Confidence>
                <ConfidenceBar>
                    <ConfidenceLevel confidence={prediction.confidence} />
                </ConfidenceBar>
            </ResultDisplay>

            <ChartContainer>
                <Bar data={chartData} options={chartOptions} />
            </ChartContainer>
        </PredictionContainer>
    );
};

export default PredictionDisplay;
