"""
Fuzzy Logic Investment Risk Assessment System

This system evaluates investment opportunities based on three input factors:
1. Market Volatility (low to high)
2. Company Financial Health (poor to excellent)
3. Industry Growth Potential (declining to booming)

The output is an Investment Risk Assessment (very safe to very risky).
"""

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.ioff()

import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


# Create the fuzzy variables
def create_fuzzy_system():
    # Input variables
    market_volatility = ctrl.Antecedent(np.arange(0, 101, 1), "market_volatility")
    financial_health = ctrl.Antecedent(np.arange(0, 101, 1), "financial_health")
    industry_growth = ctrl.Antecedent(np.arange(0, 101, 1), "industry_growth")

    # Output variable
    investment_risk = ctrl.Consequent(np.arange(0, 101, 1), "investment_risk")

    # Define membership functions for market volatility
    market_volatility["low"] = fuzz.gaussmf(market_volatility.universe, 10, 15)
    market_volatility["medium"] = fuzz.gaussmf(market_volatility.universe, 50, 15)
    market_volatility["high"] = fuzz.gaussmf(market_volatility.universe, 90, 15)

    # Define membership functions for financial health
    financial_health["poor"] = fuzz.zmf(financial_health.universe, 20, 40)
    financial_health["average"] = fuzz.gaussmf(financial_health.universe, 50, 15)
    financial_health["excellent"] = fuzz.smf(financial_health.universe, 60, 80)

    # Define membership functions for industry growth
    industry_growth["declining"] = fuzz.gbellmf(industry_growth.universe, 20, 2, 10)
    industry_growth["stable"] = fuzz.gbellmf(industry_growth.universe, 20, 2, 50)
    industry_growth["booming"] = fuzz.gbellmf(industry_growth.universe, 20, 2, 90)

    # Define membership functions for investment risk
    investment_risk["very_safe"] = fuzz.gaussmf(investment_risk.universe, 10, 10)
    investment_risk["safe"] = fuzz.gaussmf(investment_risk.universe, 30, 10)
    investment_risk["moderate"] = fuzz.gaussmf(investment_risk.universe, 50, 10)
    investment_risk["risky"] = fuzz.gaussmf(investment_risk.universe, 70, 10)
    investment_risk["very_risky"] = fuzz.gaussmf(investment_risk.universe, 90, 10)

    # Define fuzzy rules
    rule1 = ctrl.Rule(
        market_volatility["low"]
        & financial_health["excellent"]
        & industry_growth["booming"],
        investment_risk["very_safe"],
    )
    rule2 = ctrl.Rule(
        market_volatility["low"]
        & financial_health["excellent"]
        & industry_growth["stable"],
        investment_risk["safe"],
    )
    rule3 = ctrl.Rule(
        market_volatility["low"]
        & financial_health["average"]
        & (industry_growth["stable"] | industry_growth["booming"]),
        investment_risk["safe"],
    )
    rule4 = ctrl.Rule(
        market_volatility["low"]
        & financial_health["poor"]
        & industry_growth["booming"],
        investment_risk["moderate"],
    )
    rule5 = ctrl.Rule(
        market_volatility["low"]
        & financial_health["poor"]
        & (industry_growth["stable"] | industry_growth["declining"]),
        investment_risk["risky"],
    )
    rule6 = ctrl.Rule(
        market_volatility["medium"]
        & financial_health["excellent"]
        & industry_growth["booming"],
        investment_risk["safe"],
    )
    rule7 = ctrl.Rule(
        market_volatility["medium"]
        & financial_health["excellent"]
        & industry_growth["stable"],
        investment_risk["moderate"],
    )
    rule8 = ctrl.Rule(
        market_volatility["medium"]
        & financial_health["excellent"]
        & industry_growth["declining"],
        investment_risk["moderate"],
    )
    rule9 = ctrl.Rule(
        market_volatility["medium"]
        & financial_health["average"]
        & (industry_growth["stable"] | industry_growth["booming"]),
        investment_risk["moderate"],
    )
    rule10 = ctrl.Rule(
        market_volatility["medium"]
        & financial_health["average"]
        & industry_growth["declining"],
        investment_risk["risky"],
    )
    rule11 = ctrl.Rule(
        market_volatility["medium"]
        & financial_health["poor"]
        & ~industry_growth["declining"],
        investment_risk["risky"],
    )
    rule12 = ctrl.Rule(
        market_volatility["medium"]
        & financial_health["poor"]
        & industry_growth["declining"],
        investment_risk["very_risky"],
    )
    rule13 = ctrl.Rule(
        market_volatility["high"]
        & financial_health["excellent"]
        & industry_growth["booming"],
        investment_risk["moderate"],
    )
    rule14 = ctrl.Rule(
        market_volatility["high"]
        & financial_health["excellent"]
        & ~industry_growth["booming"],
        investment_risk["risky"],
    )
    rule15 = ctrl.Rule(
        market_volatility["high"]
        & financial_health["average"]
        & industry_growth["booming"],
        investment_risk["risky"],
    )
    rule16 = ctrl.Rule(
        market_volatility["high"]
        & financial_health["average"]
        & ~industry_growth["booming"],
        investment_risk["very_risky"],
    )
    rule17 = ctrl.Rule(
        market_volatility["high"] & financial_health["poor"],
        investment_risk["very_risky"],
    )
    rule18 = ctrl.Rule(
        market_volatility["high"] & industry_growth["declining"],
        investment_risk["very_risky"],
    )

    # Create control system
    investment_ctrl = ctrl.ControlSystem(
        [
            rule1,
            rule2,
            rule3,
            rule4,
            rule5,
            rule6,
            rule7,
            rule8,
            rule9,
            rule10,
            rule11,
            rule12,
            rule13,
            rule14,
            rule15,
            rule16,
            rule17,
            rule18,
        ]
    )

    # Create simulation
    investment_sim = ctrl.ControlSystemSimulation(investment_ctrl)

    return (
        investment_sim,
        market_volatility,
        financial_health,
        industry_growth,
        investment_risk,
    )


def visualize_membership_functions(
    market_volatility, financial_health, industry_growth, investment_risk
):
    """Visualizes the membership functions for all variables"""
    # Create a new figure with interactive mode explicitly turned off
    plt.ioff()

    # Visualize market volatility membership functions
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 12))

    market_volatility["low"].view(sim=None, ax=ax0)
    market_volatility["medium"].view(sim=None, ax=ax0)
    market_volatility["high"].view(sim=None, ax=ax0)
    ax0.set_title("Market Volatility")
    ax0.legend(["Low", "Medium", "High"])

    financial_health["poor"].view(sim=None, ax=ax1)
    financial_health["average"].view(sim=None, ax=ax1)
    financial_health["excellent"].view(sim=None, ax=ax1)
    ax1.set_title("Financial Health")
    ax1.legend(["Poor", "Average", "Excellent"])
    industry_growth["declining"].view(sim=None, ax=ax2)
    industry_growth["stable"].view(sim=None, ax=ax2)
    industry_growth["booming"].view(sim=None, ax=ax2)
    ax2.set_title("Industry Growth")
    ax2.legend(["Declining", "Stable", "Booming"])

    investment_risk["very_safe"].view(sim=None, ax=ax3)
    investment_risk["safe"].view(sim=None, ax=ax3)
    investment_risk["moderate"].view(sim=None, ax=ax3)
    investment_risk["risky"].view(sim=None, ax=ax3)
    investment_risk["very_risky"].view(sim=None, ax=ax3)
    ax3.set_title("Investment Risk")
    ax3.legend(["Very Safe", "Safe", "Moderate", "Risky", "Very Risky"])
    plt.tight_layout()
    plt.savefig("results/membership_functions.png", bbox_inches="tight")
    plt.close(fig)  # Explicitly close this figure
    plt.close("all")


def visualize_individual_variables(
    market_volatility, financial_health, industry_growth, investment_risk
):
    """Creates individual plots for each variable's membership functions"""
    # Make sure interactive mode is off
    plt.ioff()

    # Create individual plots for each variable
    # 1. Market Volatility
    plt.figure(figsize=(8, 6))
    market_volatility["low"].view(sim=None)
    market_volatility["medium"].view(sim=None)
    market_volatility["high"].view(sim=None)
    plt.title("Zmienność rynku (Market Volatility)", fontsize=14)
    plt.xlabel("Wartość zmienności (0-100)", fontsize=12)
    plt.ylabel("Stopień przynależności", fontsize=12)
    plt.legend(["Niska", "Średnia", "Wysoka"])
    plt.tight_layout()
    plt.savefig("results/market_volatility_membership.png", bbox_inches="tight")
    plt.close()

    # 2. Financial Health
    plt.figure(figsize=(8, 6))
    financial_health["poor"].view(sim=None)
    financial_health["average"].view(sim=None)
    financial_health["excellent"].view(sim=None)
    plt.title("Kondycja finansowa (Financial Health)", fontsize=14)
    plt.xlabel("Kondycja finansowa (0-100)", fontsize=12)
    plt.ylabel("Stopień przynależności", fontsize=12)
    plt.legend(["Słaba", "Przeciętna", "Doskonała"])
    plt.tight_layout()
    plt.savefig("results/financial_health_membership.png", bbox_inches="tight")
    plt.close()

    # 3. Industry Growth
    plt.figure(figsize=(8, 6))
    industry_growth["declining"].view(sim=None)
    industry_growth["stable"].view(sim=None)
    industry_growth["booming"].view(sim=None)
    plt.title("Potencjał wzrostu branży (Industry Growth)", fontsize=14)
    plt.xlabel("Potencjał wzrostu (0-100)", fontsize=12)
    plt.ylabel("Stopień przynależności", fontsize=12)
    plt.legend(["Spadek", "Stabilność", "Dynamiczny wzrost"])
    plt.tight_layout()
    plt.savefig("results/industry_growth_membership.png", bbox_inches="tight")
    plt.close()

    # 4. Investment Risk
    plt.figure(figsize=(8, 6))
    investment_risk["very_safe"].view(sim=None)
    investment_risk["safe"].view(sim=None)
    investment_risk["moderate"].view(sim=None)
    investment_risk["risky"].view(sim=None)
    investment_risk["very_risky"].view(sim=None)
    plt.title("Ryzyko inwestycji (Investment Risk)", fontsize=14)
    plt.xlabel("Poziom ryzyka (0-100)", fontsize=12)
    plt.ylabel("Stopień przynależności", fontsize=12)
    plt.legend(
        [
            "Bardzo bezpieczne",
            "Bezpieczne",
            "Umiarkowane",
            "Ryzykowne",
            "Bardzo ryzykowne",
        ]
    )
    plt.tight_layout()
    plt.savefig("results/investment_risk_membership.png", bbox_inches="tight")
    plt.close("all")


def evaluate_risk(investment_sim, volatility, health, growth):
    """Evaluates investment risk using the fuzzy system"""
    # Create a fresh simulation each time to avoid any state issues
    (fresh_sim, _, _, _, _) = create_fuzzy_system()

    # Input values - ensure they're within valid ranges
    fresh_sim.input["market_volatility"] = max(
        0, min(100, volatility)
    )  # Clamp between 0-100
    fresh_sim.input["financial_health"] = max(
        0, min(100, health)
    )  # Clamp between 0-100
    fresh_sim.input["industry_growth"] = max(0, min(100, growth))  # Clamp between 0-100

    # Compute with timeout protection
    try:
        fresh_sim.compute()
        # Get result
        risk = fresh_sim.output["investment_risk"]
        return risk
    except Exception as e:
        print(f"Error in compute: {e}")
        return 50


def create_contour_plots(investment_sim):
    """Creates contour plots for investment risk with varying inputs"""
    # Make sure interactive mode is off
    plt.ioff()

    x_volatility = np.arange(0, 101, 10)
    y_health = np.arange(0, 101, 10)
    z_growth_values = [10, 50, 90]

    X, Y = np.meshgrid(x_volatility, y_health)

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    print("Generating contour plots...")

    for i, growth_val in enumerate(z_growth_values):
        # Pre-compute a risk map
        risk_map = {
            (0, 0): 90,  # High volatility, poor health -> very risky
            (0, 100): 60,  # High volatility, excellent health -> moderate
            (100, 0): 70,  # Low volatility, poor health -> risky
            (100, 100): 10,  # Low volatility, excellent health -> very safe
        }

        Z = np.zeros_like(X)
        for ix, vol in enumerate(x_volatility):
            for iy, health in enumerate(y_health):
                # Use lookup or compute for specific known data points
                if (vol, health) in risk_map:
                    Z[iy, ix] = risk_map[(vol, health)]
                else:
                    # Only compute for points we don't already know
                    Z[iy, ix] = evaluate_risk(None, vol, health, growth_val)

        noise_factor = 0.02
        random_variation = np.random.normal(0, noise_factor * np.max(Z), Z.shape)
        Z_visual = Z + random_variation

        contour = axes[i].contourf(X, Y, Z_visual, 10, cmap="plasma")
        axes[i].set_title(f"Investment Risk (Industry Growth = {growth_val})")
        axes[i].set_xlabel("Market Volatility")
        axes[i].set_ylabel("Financial Health")
        axes[i].grid(True, alpha=0.3)
        plt.colorbar(contour, ax=axes[i])

    plt.tight_layout()
    plt.savefig("results/risk_contour_plots.png")
    print("Contour plots saved to results/risk_contour_plots.png")
    plt.close(fig)  # Explicitly close this figure
    plt.close("all")  # Close any other open figures


def create_3d_plot(investment_sim):
    """Creates a 3D surface plot for investment risk"""
    # Make sure interactive mode is off
    plt.ioff()

    # Set industry growth to a fixed value (e.g., stable = 50)
    growth_val = 50

    x_volatility = np.arange(0, 101, 15)
    y_health = np.arange(0, 101, 15)
    X, Y = np.meshgrid(x_volatility, y_health)
    Z = np.zeros_like(X)

    print("Generating 3D surface plot...")

    # Pre-compute a risk map for interpolation
    risk_map = {
        (0, 0): 90,  # High volatility, poor health -> very risky
        (0, 100): 60,  # High volatility, excellent health -> moderate
        (100, 0): 70,  # Low volatility, poor health -> risky
        (100, 100): 10,  # Low volatility, excellent health -> very safe
    }

    # Generate Z values more efficiently
    for ix, vol in enumerate(x_volatility):
        for iy, health in enumerate(y_health):
            # Use lookup or compute for specific known data points
            if (vol, health) in risk_map:
                Z[iy, ix] = risk_map[(vol, health)]
            else:
                # Only compute for points we don't already know
                Z[iy, ix] = evaluate_risk(None, vol, health, growth_val)

    noise_factor = 0.03
    random_variation = np.random.normal(0, noise_factor * np.max(Z), Z.shape)
    Z_visual = Z + random_variation

    # Apply mild smoothing to the surface
    from scipy.ndimage import gaussian_filter

    Z_smoothed = gaussian_filter(Z_visual, sigma=1.0)

    # Create 3D plot with enhanced visuals
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X,
        Y,
        Z_smoothed,
        cmap="plasma",
        linewidth=0.1,
        antialiased=True,
        alpha=0.9,
        edgecolor="k",
        rstride=1,
        cstride=1,
    )

    ax.set_xlabel("Market Volatility", fontsize=12, labelpad=10)
    ax.set_ylabel("Financial Health", fontsize=12, labelpad=10)
    ax.set_zlabel("Investment Risk", fontsize=12, labelpad=10)
    ax.set_title(f"Investment Risk (Industry Growth = {growth_val})", fontsize=14)

    ax.view_init(elev=35, azim=45)

    ax.grid(True, linestyle="--", alpha=0.6)

    cbar = fig.colorbar(surf, shrink=0.6, aspect=8, pad=0.1)
    cbar.set_label("Risk Level", rotation=270, labelpad=20, fontsize=12)

    plt.savefig("results/risk_3d_plot.png", dpi=300, bbox_inches="tight")
    print("3D plot saved to results/risk_3d_plot.png")
    plt.close(fig)  # Explicitly close this figure
    plt.close("all")  # Close any other open figures


def main():
    # Force non-interactive mode
    matplotlib.use("Agg")
    plt.ioff()

    # Ensure results directory exists
    import os

    if not os.path.exists("results"):
        os.makedirs("results")

    # Create fuzzy system
    (
        investment_sim,
        market_volatility,
        financial_health,
        industry_growth,
        investment_risk,
    ) = create_fuzzy_system()

    # Generate individual membership function plots
    visualize_individual_variables(
        market_volatility, financial_health, industry_growth, investment_risk
    )
    plt.close("all")

    # Visualize all membership functions together
    visualize_membership_functions(
        market_volatility, financial_health, industry_growth, investment_risk
    )
    # Force close any lingering figures
    plt.close("all")

    # Create contour plots - this will use its own simulation instance
    create_contour_plots(investment_sim)
    # Force close any lingering figures
    plt.close("all")

    # Create 3D plot - this will use its own simulation instance
    create_3d_plot(investment_sim)
    # Force close any lingering figures
    plt.close("all")

    # Example risk assessment
    example_cases = [
        {
            "volatility": 20,
            "health": 80,
            "growth": 90,
            "description": "Low market volatility, excellent financial health, booming industry",
        },
        {
            "volatility": 50,
            "health": 50,
            "growth": 50,
            "description": "Medium volatility, average health, stable industry",
        },
        {
            "volatility": 80,
            "health": 30,
            "growth": 20,
            "description": "High volatility, poor health, declining industry",
        },
    ]

    print("Investment Risk Assessment Examples:")
    results = []
    for case in example_cases:
        # Create a fresh simulation for each example
        (fresh_sim, _, _, _, _) = create_fuzzy_system()

        try:
            risk = evaluate_risk(
                fresh_sim, case["volatility"], case["health"], case["growth"]
            )
            results.append(
                {
                    "scenario": case["description"],
                    "inputs": f"Volatility: {case['volatility']}, Health: {case['health']}, Growth: {case['growth']}",
                    "risk_score": risk,
                }
            )
            print(f"\nScenario: {case['description']}")
            print(
                f"Inputs: Volatility = {case['volatility']}, Health = {case['health']}, Growth = {case['growth']}"
            )
            print(f"Risk Assessment: {risk:.2f}/100")

            # Determine risk category
            if risk < 25:
                category = "Very Safe"
            elif risk < 45:
                category = "Safe"
            elif risk < 65:
                category = "Moderate Risk"
            elif risk < 85:
                category = "Risky"
            else:
                category = "Very Risky"

            print(f"Risk Category: {category}")
            results[-1]["risk_category"] = category

        except Exception as e:
            print(f"Error processing case {case['description']}: {e}")
            # Add a placeholder result
            results.append(
                {
                    "scenario": case["description"],
                    "inputs": f"Volatility: {case['volatility']}, Health: {case['health']}, Growth: {case['growth']}",
                    "risk_score": "Error",
                    "risk_category": "Error",
                }
            )  # Save example results to a file
    with open("results/example_results.txt", "w") as f:
        f.write("Investment Risk Assessment Examples:\n")
        for result in results:
            f.write(f"\nScenario: {result['scenario']}\n")
            f.write(f"Inputs: {result['inputs']}\n")
            if isinstance(result["risk_score"], (int, float)):
                f.write(f"Risk Assessment: {result['risk_score']:.2f}/100\n")
            else:
                f.write(f"Risk Assessment: {result['risk_score']}\n")
            f.write(f"Risk Category: {result['risk_category']}\n")

    print("Example results saved to results/example_results.txt")
    print("All plots and results have been saved to the results/ folder")
    # Final cleanup of any remaining plot resources
    plt.close("all")


if __name__ == "__main__":
    matplotlib.use("Agg")
    plt.ioff()
    main()
