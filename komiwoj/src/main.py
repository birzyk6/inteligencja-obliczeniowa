"""
Główny plik wykonawczy dla problemu komiwojażera z wykorzystaniem algorytmu genetycznego.
"""

import numpy as np
import os
import time
import argparse
import sys
import pandas as pd
from typing import Dict, Any, List, Optional
import matplotlib
import pandas as pd

# Set the backend to a non-interactive one to ensure plots can be saved without display
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import the modules
from src.ga import GeneticAlgorithm

from src.optimizer import ParameterOptimizer
from src.visualization import (
    plot_route,
    plot_fitness_history,
    plot_distance_history,
    plot_time_history,
    create_evolution_animation,
    compare_methods,
    plot_optimization_results,
    plot_parameter_comparison,
    plot_parameter_boxplot,
    plot_mutation_rate_history,
)

# Define _GPU_OPTIMIZED_AVAILABLE as False since we're removing GPU implementations
_GPU_OPTIMIZED_AVAILABLE = False


def generate_cities(
    num_cities: int = 50,
    seed: Optional[int] = None,
    min_coord: int = 0,
    max_coord: int = 100,
) -> np.ndarray:
    """
    Generowanie losowych współrzędnych miast.

    Args:
        num_cities: Liczba miast
        seed: Ziarno dla generatora liczb losowych
        min_coord: Minimalna wartość współrzędnej
        max_coord: Maksymalna wartość współrzędnej

    Returns:
        Tablica współrzędnych miast (Nx2)
    """
    if seed is not None:
        np.random.seed(seed)

    cities = np.random.uniform(min_coord, max_coord, size=(num_cities, 2))

    return cities


def run_without_optimization(
    cities: np.ndarray,
    generations: int = 500,
    save_path: str = "/results",
    save_animation: bool = True,
    animation_interval: int = 10,
) -> Dict[str, Any]:
    """
    Uruchomienie algorytmu genetycznego bez optymalizacji parametrów.

    Args:
        cities: Tablica współrzędnych miast
        generations: Liczba pokoleń
        save_path: Ścieżka do zapisu wyników
        save_animation: Czy zapisać animację ewolucji
        animation_interval: Co ile generacji zapisywać klatkę animacji

    Returns:
        Słownik z wynikami
    """
    # Domyślne parametry
    params = {
        "population_size": 100,
        "elite_size": 20,
        "mutation_rate": 0.01,
        "tournament_size": 5,
        "crossover_type": "OX",
        "mutation_type": "swap",
        "selection_type": "tournament",
    }

    print("Uruchamianie algorytmu bez optymalizacji parametrów...")

    # Utworzenie instancji algorytmu genetycznego
    start_time = time.time()

    print("Używanie implementacji CPU")
    ga = GeneticAlgorithm(cities=cities, generations=generations, **params)

    # Uruchomienie algorytmu
    results = ga.evolve(verbose=True, save_interval=animation_interval)

    elapsed_time = time.time() - start_time
    results["time"] = elapsed_time

    print(f"Czas wykonania: {elapsed_time:.2f} s")
    print(f"Najlepsza trasa: {results['best_distance']:.2f}")

    # Zapisanie wizualizacji
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    print(
        f"Saving visualizations to: {os.path.abspath(save_path)}"
    )  # Save path for each plot
    route_path = os.path.join(save_path, "best_route_without_opt.png")
    fitness_path = os.path.join(save_path, "fitness_history_without_opt.png")
    distance_path = os.path.join(save_path, "distance_history_without_opt.png")
    time_path = os.path.join(save_path, "time_history_without_opt.png")
    mutation_rate_path = os.path.join(
        save_path, "mutation_rate_history_without_opt.png"
    )

    # Save plots with explicit save paths
    try:
        # Wizualizacja najlepszej trasy
        plot_route(
            cities=cities,
            route=results["best_route"],
            title=f"Najlepsza trasa (bez optymalizacji): {results['best_distance']:.2f}",
            save_path=route_path,
            show=False,
        )
        print(f"Saved route plot to: {route_path}")

        # Wizualizacje historii
        plot_fitness_history(
            history=results["history"],
            title="Historia funkcji przystosowania (bez optymalizacji)",
            save_path=fitness_path,
            show=False,
        )
        print(f"Saved fitness history plot to: {fitness_path}")

        plot_distance_history(
            history=results["history"],
            title="Historia długości trasy (bez optymalizacji)",
            save_path=distance_path,
            show=False,
        )
        print(f"Saved distance history plot to: {distance_path}")

        plot_time_history(
            history=results["history"],
            title="Czas wykonania generacji (bez optymalizacji)",
            save_path=time_path,
            show=False,
        )
        print(f"Saved time history plot to: {time_path}")

        # Wizualizacja historii współczynnika mutacji
        mutation_rate_path = os.path.join(
            save_path, "mutation_rate_history_without_opt.png"
        )
        plot_mutation_rate_history(
            history=results["history"],
            title="Historia współczynnika mutacji (bez optymalizacji)",
            save_path=mutation_rate_path,
            show=False,
        )
        print(f"Saved mutation rate history plot to: {mutation_rate_path}")
    except Exception as e:
        print(f"Error saving plots: {e}")  # Tworzenie animacji ewolucji
    if save_animation:
        print("Tworzenie animacji ewolucji...")
        create_evolution_animation(
            cities=cities,
            history=results["history"],
            save_path=os.path.join(save_path, "evolution_without_opt.gif"),
            interval=500,  # ms między klatkami
            title="Ewolucja trasy - algorytm genetyczny bez optymalizacji parametrów",
        )

    return results


def run_with_optimization(
    cities: np.ndarray,
    generations: int = 500,
    n_configs: int = 10,
    save_path: str = "/results",
    save_animation: bool = True,
    animation_interval: int = 10,
) -> Dict[str, Any]:
    """
    Uruchomienie algorytmu genetycznego z optymalizacją parametrów.

    Args:
        cities: Tablica współrzędnych miast
        generations: Liczba pokoleń
        n_configs: Liczba konfiguracji do przetestowania
        save_path: Ścieżka do zapisu wyników
        save_animation: Czy zapisać animację ewolucji
        animation_interval: Co ile generacji zapisywać klatkę animacji

    Returns:
        Słownik z wynikami
    """
    print("Uruchamianie optymalizacji parametrów...")

    # Utworzenie optymalizatora parametrów
    optimizer = ParameterOptimizer(
        cities=cities,
        use_gpu=False,
        n_trials=2,
        n_generations=300,  # Mniejsza liczba pokoleń dla optymalizacji
        save_path=save_path,
    )  # Optymalizacja parametrów
    best_params = optimizer.optimize(n_configs=n_configs)

    print("Najlepsze parametry:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Get the optimization results for visualization
    optimization_results = optimizer.get_results_summary()

    # Save the optimization results for plotting parameter influence
    results_dict = {
        "parameters": best_params,
    }

    # Add each parameter's values and distances for comparison plots
    for param in best_params.keys():
        results_dict[param] = optimization_results[param].tolist()

    results_dict["best_distance"] = optimization_results[
        "best_distance"
    ].tolist()  # Create parameter comparison plots
    plot_optimization_results(
        results=results_dict,
        save_dir=save_path,
        show=False,
        figsize=(10, 6),
        dpi=100,
    )

    # Save detailed parameter analysis to a text file
    data_path = os.path.join(os.path.dirname(save_path), "data")
    os.makedirs(data_path, exist_ok=True)
    save_parameter_comparison_results(
        opt_results=optimization_results,
        save_path=os.path.join(data_path, "parameter_analysis.txt"),
    )

    print("\nUruchamianie algorytmu z optymalnymi parametrami...")

    # Utworzenie instancji algorytmu genetycznego z optymalnymi parametrami
    start_time = time.time()

    print("Używanie implementacji CPU")
    ga = GeneticAlgorithm(cities=cities, generations=generations, **best_params)

    # Uruchomienie algorytmu
    results = ga.evolve(verbose=True, save_interval=animation_interval)

    elapsed_time = time.time() - start_time
    results["time"] = elapsed_time

    print(f"Czas wykonania: {elapsed_time:.2f} s")
    print(f"Najlepsza trasa: {results['best_distance']:.2f}")

    # Zapisanie wizualizacji
    os.makedirs(save_path, exist_ok=True)

    # Save path for each plot
    route_path = os.path.join(save_path, "best_route_with_opt.png")
    fitness_path = os.path.join(save_path, "fitness_history_with_opt.png")
    distance_path = os.path.join(save_path, "distance_history_with_opt.png")
    time_path = os.path.join(save_path, "time_history_with_opt.png")

    # Save plots with explicit save paths
    try:
        # Wizualizacja najlepszej trasy
        plot_route(
            cities=cities,
            route=results["best_route"],
            title=f"Najlepsza trasa (z optymalizacją): {results['best_distance']:.2f}",
            save_path=route_path,
            show=False,
        )
        print(f"Saved route plot to: {route_path}")

        # Wizualizacje historii
        plot_fitness_history(
            history=results["history"],
            title="Historia funkcji przystosowania (z optymalizacją)",
            save_path=fitness_path,
            show=False,
        )
        print(f"Saved fitness history plot to: {fitness_path}")

        plot_distance_history(
            history=results["history"],
            title="Historia długości trasy (z optymalizacją)",
            save_path=distance_path,
            show=False,
        )
        print(f"Saved distance history plot to: {distance_path}")

        plot_time_history(
            history=results["history"],
            title="Czas wykonania generacji (z optymalizacją)",
            save_path=time_path,
            show=False,
        )
        print(f"Saved time history plot to: {time_path}")

        # Wizualizacja historii współczynnika mutacji
        mutation_rate_path = os.path.join(
            save_path, "mutation_rate_history_with_opt.png"
        )
        plot_mutation_rate_history(
            history=results["history"],
            title="Historia współczynnika mutacji (z optymalizacją)",
            save_path=mutation_rate_path,
            show=False,
        )
        print(f"Saved mutation rate history plot to: {mutation_rate_path}")
    except Exception as e:
        print(f"Error saving plots: {e}")  # Tworzenie animacji ewolucji
    if save_animation:
        print("Tworzenie animacji ewolucji...")
        create_evolution_animation(
            cities=cities,
            history=results["history"],
            save_path=os.path.join(save_path, "evolution_with_opt.gif"),
            interval=500,  # ms między klatkami            title="Ewolucja trasy - algorytm genetyczny z optymalizacją parametrów",
        )

    # Store the optimization results in the returned dictionary
    results["optimization_results"] = optimization_results

    return results


def compare_results(
    results_without_opt: Dict[str, Any],
    results_with_opt: Dict[str, Any],
    save_path: str = "/results",
):
    """
    Porównanie wyników z i bez optymalizacji parametrów.

    Args:
        results_without_opt: Wyniki bez optymalizacji
        results_with_opt: Wyniki z optymalizacją
        save_path: Ścieżka do zapisu wizualizacji
    """
    print("\nPorównanie wyników:")
    print(f"Bez optymalizacji: {results_without_opt['best_distance']:.2f}")
    print(f"Z optymalizacją: {results_with_opt['best_distance']:.2f}")

    improvement = (
        (results_without_opt["best_distance"] - results_with_opt["best_distance"])
        / results_without_opt["best_distance"]
        * 100
    )
    print(f"Poprawa: {improvement:.2f}%")

    # Wizualizacja porównania
    compare_methods(
        method_names=["Bez optymalizacji", "Z optymalizacją"],
        distances=[
            results_without_opt["best_distance"],
            results_with_opt["best_distance"],
        ],
        times=[results_without_opt["time"], results_with_opt["time"]],
        title="Porównanie algorytmu z i bez optymalizacji parametrów",
        save_path=os.path.join(save_path, "comparison.png"),
        show=False,
    )

    # Porównanie zbieżności (historii najlepszych tras)
    plt.figure(figsize=(12, 6), dpi=100)

    generations_without_opt = np.arange(
        len(results_without_opt["history"]["best_distance"])
    )
    generations_with_opt = np.arange(len(results_with_opt["history"]["best_distance"]))

    plt.plot(
        generations_without_opt * 10,  # Co 10 generacji
        results_without_opt["history"]["best_distance"],
        "b-",
        linewidth=2,
        label="Bez optymalizacji",
    )
    plt.plot(
        generations_with_opt * 10,  # Co 10 generacji
        results_with_opt["history"]["best_distance"],
        "r-",
        linewidth=2,
        label="Z optymalizacją",
    )

    plt.title("Porównanie zbieżności algorytmu")
    plt.xlabel("Generacja")
    plt.ylabel("Długość trasy")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.savefig(
        os.path.join(save_path, "convergence_comparison.png"),
        dpi=100,
        bbox_inches="tight",
    )

    plt.close()


def save_parameters_info(
    params_without_opt: Dict[str, Any],
    params_with_opt: Dict[str, Any],
    save_path: str = "/results",
):
    """
    Zapisanie informacji o parametrach do pliku tekstowego.

    Args:
        params_without_opt: Parametry bez optymalizacji
        params_with_opt: Parametry z optymalizacją
        save_path: Ścieżka do zapisu pliku
    """
    with open(
        os.path.join(save_path, "parameters_comparison.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("Porównanie parametrów algorytmu genetycznego\n")
        f.write("===========================================\n\n")

        f.write("Parametry bez optymalizacji:\n")
        for param, value in params_without_opt.items():
            f.write(f"  {param}: {value}\n")

        f.write("\nParametry z optymalizacją:\n")
        for param, value in params_with_opt.items():
            f.write(f"  {param}: {value}\n")


def save_detailed_results(
    results_without_opt: Dict[str, Any],
    results_with_opt: Dict[str, Any],
    cities: np.ndarray,
    save_path: str = "data",
):
    """
    Zapisanie szczegółowych wyników do plików tekstowych w katalogu data.

    Args:
        results_without_opt: Wyniki bez optymalizacji
        results_with_opt: Wyniki z optymalizacją
        cities: Tablica współrzędnych miast
        save_path: Ścieżka do zapisu plików
    """
    # Upewnienie się, że katalog istnieje
    os.makedirs(save_path, exist_ok=True)

    # Zapisanie wyników bez optymalizacji
    with open(
        os.path.join(save_path, "results_without_opt.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("Wyniki algorytmu genetycznego bez optymalizacji parametrów\n")
        f.write("=====================================================\n\n")

        # Informacje ogólne
        f.write(f"Liczba miast: {len(cities)}\n")
        f.write(f"Liczba pokoleń: {results_without_opt['parameters']['generations']}\n")
        f.write(f"Czas wykonania: {results_without_opt['time']:.2f} s\n\n")

        # Parametry
        f.write("Parametry:\n")
        for param, value in results_without_opt["parameters"].items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")

        # Najlepsza trasa
        f.write(
            f"Najlepsza długość trasy: {results_without_opt['best_distance']:.2f}\n"
        )
        f.write(
            f"Najlepsza wartość funkcji przystosowania: {results_without_opt['best_fitness']:.10f}\n"
        )
        f.write(
            f"Średnia wartość funkcji przystosowania: {results_without_opt['mean_fitness']:.10f}\n\n"
        )

        # Najlepsza trasa (jako sekwencja miast)
        f.write("Najlepsza trasa (sekwencja miast):\n")
        f.write(", ".join(map(str, results_without_opt["best_route"])))
        f.write("\n\n")

        # Historia ewolucji (co 10 generacji)
        f.write("Historia ewolucji (co 10 generacji):\n")
        f.write(
            "Generacja,Najlepsza długość,Najlepsza funkcja przystosowania,Średnia funkcja przystosowania,Czas generacji (ms)\n"
        )

        for i, (dist, fit, mean_fit, gen_time) in enumerate(
            zip(
                results_without_opt["history"]["best_distance"],
                results_without_opt["history"]["best_fitness"],
                results_without_opt["history"]["mean_fitness"],
                results_without_opt["history"]["time_per_gen"],
            )
        ):
            gen_num = i * 10  # Co 10 generacji
            f.write(
                f"{gen_num},{dist:.2f},{fit:.10f},{mean_fit:.10f},{gen_time*1000:.2f}\n"
            )

    # Zapisanie wyników z optymalizacją
    with open(
        os.path.join(save_path, "results_with_opt.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("Wyniki algorytmu genetycznego z optymalizacją parametrów\n")
        f.write("==================================================\n\n")

        # Informacje ogólne
        f.write(f"Liczba miast: {len(cities)}\n")
        f.write(f"Liczba pokoleń: {results_with_opt['parameters']['generations']}\n")
        f.write(f"Czas wykonania: {results_with_opt['time']:.2f} s\n\n")

        # Parametry
        f.write("Parametry:\n")
        for param, value in results_with_opt["parameters"].items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")

        # Najlepsza trasa
        f.write(f"Najlepsza długość trasy: {results_with_opt['best_distance']:.2f}\n")
        f.write(
            f"Najlepsza wartość funkcji przystosowania: {results_with_opt['best_fitness']:.10f}\n"
        )
        f.write(
            f"Średnia wartość funkcji przystosowania: {results_with_opt['mean_fitness']:.10f}\n\n"
        )

        # Najlepsza trasa (jako sekwencja miast)
        f.write("Najlepsza trasa (sekwencja miast):\n")
        f.write(", ".join(map(str, results_with_opt["best_route"])))
        f.write("\n\n")

        # Historia ewolucji (co 10 generacji)
        f.write("Historia ewolucji (co 10 generacji):\n")
        f.write(
            "Generacja,Najlepsza długość,Najlepsza funkcja przystosowania,Średnia funkcja przystosowania,Czas generacji (ms)\n"
        )

        for i, (dist, fit, mean_fit, gen_time) in enumerate(
            zip(
                results_with_opt["history"]["best_distance"],
                results_with_opt["history"]["best_fitness"],
                results_with_opt["history"]["mean_fitness"],
                results_with_opt["history"]["time_per_gen"],
            )
        ):
            gen_num = i * 10  # Co 10 generacji
            f.write(
                f"{gen_num},{dist:.2f},{fit:.10f},{mean_fit:.10f},{gen_time*1000:.2f}\n"
            )

    # Zapisanie porównawczego pliku csv z danymi do analizy
    with open(
        os.path.join(save_path, "comparison_data.csv"), "w", encoding="utf-8"
    ) as f:
        f.write("metric,without_optimization,with_optimization,improvement_percent\n")

        # Długość trasy
        improvement = (
            (results_without_opt["best_distance"] - results_with_opt["best_distance"])
            / results_without_opt["best_distance"]
            * 100
        )
        f.write(
            f"best_distance,{results_without_opt['best_distance']:.2f},{results_with_opt['best_distance']:.2f},{improvement:.2f}\n"
        )

        # Funkcja przystosowania
        improvement = (
            (results_with_opt["best_fitness"] - results_without_opt["best_fitness"])
            / results_without_opt["best_fitness"]
            * 100
        )
        f.write(
            f"best_fitness,{results_without_opt['best_fitness']:.10f},{results_with_opt['best_fitness']:.10f},{improvement:.2f}\n"
        )

        # Średnia funkcja przystosowania
        improvement = (
            (results_with_opt["mean_fitness"] - results_without_opt["mean_fitness"])
            / results_without_opt["mean_fitness"]
            * 100
        )
        f.write(
            f"mean_fitness,{results_without_opt['mean_fitness']:.10f},{results_with_opt['mean_fitness']:.10f},{improvement:.2f}\n"
        )

        # Czas wykonania
        improvement = (
            (results_without_opt["time"] - results_with_opt["time"])
            / results_without_opt["time"]
            * 100
        )
        f.write(
            f"execution_time,{results_without_opt['time']:.2f},{results_with_opt['time']:.2f},{improvement:.2f}\n"
        )

    print(f"Szczegółowe wyniki zostały zapisane w katalogu: {save_path}")


def save_routes_csv(
    results_without_opt: Dict[str, Any],
    results_with_opt: Dict[str, Any],
    cities: np.ndarray,
    save_path: str = "data",
):
    """
    Zapisanie tras jako pliki CSV z koordynatami do wizualizacji w zewnętrznych narzędziach.

    Args:
        results_without_opt: Wyniki bez optymalizacji
        results_with_opt: Wyniki z optymalizacją
        cities: Tablica współrzędnych miast
        save_path: Ścieżka do zapisu plików
    """
    # Upewnienie się, że katalog istnieje
    os.makedirs(save_path, exist_ok=True)

    # Zapisanie trasy bez optymalizacji
    route_without_opt = results_without_opt["best_route"]
    # Dodanie pierwszego miasta na końcu, aby zamknąć trasę
    route_without_opt_closed = np.append(route_without_opt, route_without_opt[0])

    # Przygotowanie danych: współrzędne miast w kolejności trasy
    coords_without_opt = np.zeros((len(route_without_opt_closed), 3))
    for i, city_idx in enumerate(route_without_opt_closed):
        coords_without_opt[i, 0] = i  # Indeks w trasie
        coords_without_opt[i, 1:] = cities[city_idx]  # Współrzędne X, Y

    # Zapisanie do CSV
    np.savetxt(
        os.path.join(save_path, "route_without_opt.csv"),
        coords_without_opt,
        delimiter=",",
        header="order,x,y",
        comments="",
        fmt=["%.0f", "%.5f", "%.5f"],
    )

    # Zapisanie trasy z optymalizacją
    route_with_opt = results_with_opt["best_route"]
    # Dodanie pierwszego miasta na końcu, aby zamknąć trasę
    route_with_opt_closed = np.append(route_with_opt, route_with_opt[0])

    # Przygotowanie danych: współrzędne miast w kolejności trasy
    coords_with_opt = np.zeros((len(route_with_opt_closed), 3))
    for i, city_idx in enumerate(route_with_opt_closed):
        coords_with_opt[i, 0] = i  # Indeks w trasie
        coords_with_opt[i, 1:] = cities[city_idx]  # Współrzędne X, Y

    # Zapisanie do CSV
    np.savetxt(
        os.path.join(save_path, "route_with_opt.csv"),
        coords_with_opt,
        delimiter=",",
        header="order,x,y",
        comments="",
        fmt=["%.0f", "%.5f", "%.5f"],
    )

    # Zapisanie wszystkich miast do CSV (dla łatwiejszej wizualizacji)
    city_data = np.zeros((len(cities), 3))
    for i, city in enumerate(cities):
        city_data[i, 0] = i  # Indeks miasta
        city_data[i, 1:] = city  # Współrzędne X, Y

    np.savetxt(
        os.path.join(save_path, "all_cities.csv"),
        city_data,
        delimiter=",",
        header="id,x,y",
        comments="",
        fmt=["%.0f", "%.5f", "%.5f"],
    )


def save_parameter_comparison_results(
    opt_results: pd.DataFrame,
    save_path: str,
):
    """
    Zapisanie wyników porównania parametrów do pliku tekstowego.

    Args:
        opt_results: Ramka danych z wynikami optymalizacji parametrów
        save_path: Ścieżka do zapisu pliku
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Group by each parameter and calculate average distance
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("PORÓWNANIE WPŁYWU PARAMETRÓW NA DŁUGOŚĆ TRASY\n")
        f.write("=============================================\n\n")

        # Parameters to analyze
        params = [
            "population_size",
            "elite_size",
            "mutation_rate",
            "tournament_size",
            "crossover_type",
            "mutation_type",
            "selection_type",
        ]

        for param in params:
            if param in opt_results.columns:
                f.write(f"\nParameter: {param}\n")
                f.write("-" * (len(param) + 10) + "\n")

                # Group by parameter value and calculate statistics
                grouped = opt_results.groupby(param)["best_distance"].agg(
                    ["mean", "std", "min", "count"]
                )
                grouped = grouped.sort_values("mean")

                # Format and write the results
                f.write(
                    f"{'Value':<15} {'Mean Distance':<15} {'Std Dev':<15} {'Min Distance':<15} {'Count':<8}\n"
                )

                for value, (mean, std, min_dist, count) in grouped.iterrows():
                    f.write(
                        f"{str(value):<15} {mean:.2f}{' '*9} {std:.2f}{' '*9} {min_dist:.2f}{' '*9} {count:<8}\n"
                    )

                f.write("\n")

        # Add overall best parameters
        best_idx = opt_results["best_distance"].idxmin()
        best_config = opt_results.loc[best_idx]

        f.write("\nNAJLEPSZA KONFIGURACJA PARAMETRÓW\n")
        f.write("===============================\n\n")

        for param in params:
            if param in best_config:
                f.write(f"{param}: {best_config[param]}\n")

        f.write(f"\nNajlepsza długość trasy: {best_config['best_distance']:.2f}\n")


def main():
    """
    Główna funkcja programu.
    """
    parser = argparse.ArgumentParser(
        description="Problem komiwojażera z algorytmem genetycznym"
    )

    parser.add_argument(
        "--num-cities", type=int, default=50, help="Liczba miast (domyślnie: 50)"
    )
    parser.add_argument(
        "--generations", type=int, default=500, help="Liczba pokoleń (domyślnie: 500)"
    )
    parser.add_argument(
        "--opt-configs",
        type=int,
        default=10,
        help="Liczba konfiguracji do przetestowania w optymalizacji (domyślnie: 10)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="results",
        help="Ścieżka do zapisu wyników (domyślnie: results)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data",
        help="Ścieżka do zapisu szczegółowych danych do analizy (domyślnie: data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Ziarno dla generatora liczb losowych (domyślnie: 42)",
    )
    parser.add_argument(
        "--skip-animation", action="store_true", help="Pomiń tworzenie animacji"
    )
    parser.add_argument(
        "--animation-interval",
        type=int,
        default=10,
        help="Co ile generacji zapisywać klatkę animacji (domyślnie: 10)",
    )

    args = parser.parse_args()

    # Generowanie miast
    cities = generate_cities(num_cities=args.num_cities, seed=args.seed)

    # Zapisanie informacji o miastach
    os.makedirs(args.save_path, exist_ok=True)
    np.savetxt(
        os.path.join(args.save_path, "cities.csv"),
        cities,
        delimiter=",",
        header="x,y",
        comments="",
    )

    # Wizualizacja miast
    plt.figure(figsize=(10, 8), dpi=100)
    plt.scatter(cities[:, 0], cities[:, 1], s=100, c="red", alpha=0.7)

    # Numerowanie miast
    for i, city in enumerate(cities):
        plt.text(city[0] + 0.1, city[1] + 0.1, str(i), fontsize=8)

    plt.title(f"Rozmieszczenie {args.num_cities} miast")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(alpha=0.3)
    plt.savefig(
        os.path.join(args.save_path, "cities.png"), dpi=100, bbox_inches="tight"
    )
    plt.close()

    # Uruchomienie algorytmu bez optymalizacji
    results_without_opt = run_without_optimization(
        cities=cities,
        generations=args.generations,
        save_path=args.save_path,
        save_animation=not args.skip_animation,
        animation_interval=args.animation_interval,
    )

    # Uruchomienie algorytmu z optymalizacją
    results_with_opt = run_with_optimization(
        cities=cities,
        generations=args.generations,
        n_configs=args.opt_configs,
        save_path=args.save_path,
        save_animation=not args.skip_animation,
        animation_interval=args.animation_interval,
    )

    # Porównanie wyników
    compare_results(
        results_without_opt=results_without_opt,
        results_with_opt=results_with_opt,
        save_path=args.save_path,
    )  # Zapisanie informacji o parametrach
    save_parameters_info(
        params_without_opt=results_without_opt["parameters"],
        params_with_opt=results_with_opt["parameters"],
        save_path=args.save_path,
    )

    # Zapisanie szczegółowych wyników do plików tekstowych w katalogu data
    data_path = "data"
    os.makedirs(data_path, exist_ok=True)
    save_detailed_results(
        results_without_opt=results_without_opt,
        results_with_opt=results_with_opt,
        cities=cities,
        save_path=data_path,
    )

    # Zapisanie tras jako pliki CSV do wizualizacji w zewnętrznych narzędziach
    save_routes_csv(
        results_without_opt=results_without_opt,
        results_with_opt=results_with_opt,
        cities=cities,
        save_path=data_path,
    )  # Zapisanie wyników porównania parametrów do pliku tekstowego
    opt_results_path = os.path.join(args.save_path, "optimization_results.txt")

    # Check if optimization_results is available
    if "optimization_results" in results_with_opt:
        save_parameter_comparison_results(
            opt_results=results_with_opt["optimization_results"],
            save_path=opt_results_path,
        )
        print(f"Saved parameter comparison results to: {opt_results_path}")
    else:
        print("Optimization results not available for parameter comparison analysis.")

    print("\nWszystkie wyniki zostały zapisane w katalogu:", args.save_path)
    print(f"Szczegółowe dane do analizy zostały zapisane w katalogu: {data_path}")


if __name__ == "__main__":
    main()
