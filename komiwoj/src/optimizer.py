"""
Optymalizacja parametrów algorytmu genetycznego dla problemu komiwojażera.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
from sklearn.model_selection import ParameterGrid
import pandas as pd
from tqdm import tqdm
import joblib
import os
import sys

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ga import GeneticAlgorithm

# Define constants
_GPU_OPTIMIZED_AVAILABLE = False


class ParameterOptimizer:
    """
    Klasa do optymalizacji parametrów algorytmu genetycznego.
    """

    def __init__(
        self,
        cities: np.ndarray,
        use_gpu: bool = False,  # Always False now since we've removed GPU support
        n_trials: int = 3,
        n_generations: int = 100,
        save_path: str = "../results",
    ):
        """
        Inicjalizacja optymalizatora parametrów.

        Args:
            cities: Tablica współrzędnych miast (Nx2)
            use_gpu: Parametr zachowany dla kompatybilności (zawsze False)
            n_trials: Liczba powtórzeń dla każdego zestawu parametrów
            n_generations: Liczba pokoleń dla każdego uruchomienia algorytmu
            save_path: Ścieżka do zapisu wyników
        """
        self.cities = cities
        self.use_gpu = False  # Always use CPU
        self.n_trials = n_trials
        self.n_generations = n_generations
        self.save_path = save_path  # Zapewnienie istnienia katalogu do zapisu wyników
        os.makedirs(save_path, exist_ok=True)

        # Parametry do optymalizacji
        self.param_grid = {
            "population_size": [30, 50, 100, 150, 200, 250],
            "elite_size": [2, 5, 10, 15, 20, 25],
            "mutation_rate": [0.005, 0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
            "tournament_size": [2, 3, 5, 7, 10, 15],
            "crossover_type": ["OX", "PMX", "CX"],
            "mutation_type": ["swap", "insert", "inversion", "scramble"],
            "selection_type": ["tournament", "roulette", "rank"],
        }  # Wyniki optymalizacji
        self.results = []

    def _evaluate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ocena jednego zestawu parametrów.

        Args:
            params: Słownik z parametrami do oceny

        Returns:
            Wyniki dla danego zestawu parametrów
        """
        results = []

        for trial in range(self.n_trials):
            # Ustawienie ziarna dla powtarzalności
            seed = trial

            # Utworzenie instancji algorytmu genetycznego
            ga = GeneticAlgorithm(
                cities=self.cities,
                generations=self.n_generations,
                seed=seed,
                **params,
            )

            # Uruchomienie algorytmu
            start_time = time.time()
            result = ga.evolve(verbose=False)
            elapsed_time = time.time() - start_time

            # Zapisanie wyników
            trial_result = {
                "trial": trial,
                "best_distance": result["best_distance"],
                "best_fitness": result["best_fitness"],
                "mean_fitness": result["mean_fitness"],
                "time": elapsed_time,
                **params,
            }

            results.append(trial_result)

        # Obliczenie średnich wyników z wszystkich prób
        avg_result = {
            "best_distance": np.mean([r["best_distance"] for r in results]),
            "best_fitness": np.mean([r["best_fitness"] for r in results]),
            "mean_fitness": np.mean([r["mean_fitness"] for r in results]),
            "time": np.mean([r["time"] for r in results]),
            "std_distance": np.std([r["best_distance"] for r in results]),
            **params,
        }

        return avg_result

    def optimize(self, n_configs: Optional[int] = None) -> Dict[str, Any]:
        """
        Optymalizacja parametrów.

        Args:
            n_configs: Maksymalna liczba konfiguracji do przetestowania (None = wszystkie)

        Returns:
            Najlepsze parametry
        """
        # Generowanie siatki parametrów
        param_grid = list(ParameterGrid(self.param_grid))

        # Ograniczenie liczby konfiguracji, jeśli podano
        if n_configs is not None and n_configs < len(param_grid):
            # Losowy wybór konfiguracji
            np.random.seed(42)  # Dla powtarzalności
            param_grid = np.random.choice(param_grid, size=n_configs, replace=False)

        print(f"Testowanie {len(param_grid)} konfiguracji parametrów...")

        # Testowanie wszystkich konfiguracji
        for params in tqdm(param_grid):
            result = self._evaluate_params(params)
            self.results.append(result)

        # Utworzenie ramki danych z wynikami
        results_df = pd.DataFrame(self.results)

        # Zapisanie wyników do pliku
        results_df.to_csv(
            os.path.join(self.save_path, "parameter_optimization_results.csv"),
            index=False,
        )

        # Znalezienie najlepszej konfiguracji (najmniejsza odległość)
        best_idx = results_df["best_distance"].argmin()
        best_params = results_df.iloc[best_idx].to_dict()

        # Zapisanie najlepszych parametrów
        best_params_df = pd.DataFrame([best_params])
        best_params_df.to_csv(
            os.path.join(self.save_path, "best_parameters.csv"), index=False
        )

        # Zapisanie modelu najlepszych parametrów
        best_params_model = {k: best_params[k] for k in self.param_grid.keys()}
        joblib.dump(
            best_params_model, os.path.join(self.save_path, "best_parameters.joblib")
        )

        print(f"Najlepsze parametry: {best_params_model}")
        print(f"Najlepsza średnia odległość: {best_params['best_distance']:.2f}")

        return best_params_model

    def get_results_summary(self) -> pd.DataFrame:
        """
        Pobranie podsumowania wyników optymalizacji.

        Returns:
            Ramka danych z wynikami
        """
        if not self.results:
            raise ValueError("Brak wyników. Najpierw uruchom metodę 'optimize'.")

        return pd.DataFrame(self.results)

    @classmethod
    def load_best_params(cls, load_path: str = "../results") -> Dict[str, Any]:
        """
        Wczytanie najlepszych parametrów z pliku.

        Args:
            load_path: Ścieżka do wczytania parametrów

        Returns:
            Słownik z najlepszymi parametrami
        """
        params_path = os.path.join(load_path, "best_parameters.joblib")

        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Nie znaleziono pliku parametrów w {params_path}")

        return joblib.load(params_path)
