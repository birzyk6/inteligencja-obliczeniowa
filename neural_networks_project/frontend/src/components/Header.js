import React from "react";
import styled from "styled-components";

const HeaderContainer = styled.header`
    text-align: center;
    color: #ffffff;
    margin-bottom: 20px;
`;

const Title = styled.h1`
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 10px;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;

    @media (max-width: 768px) {
        font-size: 2.5rem;
    }
`;

const Subtitle = styled.p`
    font-size: 1.2rem;
    opacity: 0.8;
    margin-bottom: 10px;
    color: #e2e8f0;
`;

const Description = styled.p`
    font-size: 1rem;
    opacity: 0.7;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.5;
    color: #cbd5e1;
`;

const Header = () => {
    return (
        <HeaderContainer>
            {" "}
            <Title>Neural Network Visualizer</Title>
            <Subtitle>AI-Powered Digit Recognition</Subtitle>
            <Description>
                Draw a digit (0-9) below and watch how a neural network processes your drawing through each layer to
                make a prediction.
            </Description>
        </HeaderContainer>
    );
};

export default Header;
