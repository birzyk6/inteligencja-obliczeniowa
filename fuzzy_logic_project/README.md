# Fuzzy Logic Investment Risk Assessment System

This project implements a fuzzy logic-based system for evaluating investment opportunities based on multiple economic factors. The system uses fuzzy logic to model the inherent uncertainty in investment decisions and provides a risk assessment score based on defined rules.

## Overview

The system evaluates investment opportunities using three key input factors:

1. **Market Volatility** (low to high)
2. **Company Financial Health** (poor to excellent)
3. **Industry Growth Potential** (declining to booming)

Based on these inputs, the system produces an **Investment Risk Assessment** score on a scale from very safe to very risky.

## Features

-   Implementation of fuzzy membership functions for all input and output variables
-   Comprehensive rule base with 18 rules covering various investment scenarios
-   Visualization of membership functions for all variables
-   Contour plots showing risk assessment for different combinations of inputs
-   3D surface plot for visualizing the relationship between inputs and output
-   Example case analysis for demonstration

## Requirements

-   Python 3.6+
-   NumPy
-   Matplotlib
-   scikit-fuzzy

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python src/fuzzy_investment.py
```

The script will:

1. Create the fuzzy logic system
2. Generate visualizations of membership functions
3. Create contour plots for different scenarios
4. Generate 3D surface plots
5. Run example investment scenarios and output the results

## Membership Functions

### Input Variables

-   **Market Volatility**:

    -   Low (0-50)
    -   Medium (20-80)
    -   High (50-100)

-   **Financial Health**:

    -   Poor (0-40)
    -   Average (20-80)
    -   Excellent (60-100)

-   **Industry Growth**:
    -   Declining (0-40)
    -   Stable (20-80)
    -   Booming (60-100)

### Output Variable

-   **Investment Risk**:
    -   Very Safe (0-25)
    -   Safe (0-50)
    -   Moderate (25-75)
    -   Risky (50-100)
    -   Very Risky (75-100)

## Defuzzification Method

The system uses the centroid method (center of gravity) for defuzzification. This method calculates the center of the area under the curve of the output membership function.

## Output

The project generates:

-   Membership function visualizations (`membership_functions.png`)
-   Contour plots for risk assessment (`risk_contour_plots.png`)
-   3D surface plot (`risk_3d_plot.png`)
-   Text file with example results (`example_results.txt`)

## License

MIT
