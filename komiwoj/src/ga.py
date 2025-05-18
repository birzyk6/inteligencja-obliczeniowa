"""
Implementacja algorytmu genetycznego dla problemu komiwojażera na CPU.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional


class GeneticAlgorithm:
    """
    Klasa implementująca algorytm genetyczny dla problemu komiwojażera na CPU.
    """

    def __init__(
        self,
        cities: np.ndarray,
        population_size: int = 100,
        elite_size: int = 20,
        mutation_rate: float = 0.01,
        generations: int = 500,
        tournament_size: int = 5,
        crossover_type: str = "OX",
        mutation_type: str = "swap",
        selection_type: str = "tournament",
        seed: Optional[int] = None,
    ):
        """
        Inicjalizacja algorytmu genetycznego.

        Args:
            cities: Tablica współrzędnych miast (Nx2)
            population_size: Rozmiar populacji
            elite_size: Liczba osobników elitarnych
            mutation_rate: Współczynnik mutacji
            generations: Liczba pokoleń
            tournament_size: Rozmiar turnieju dla selekcji turniejowej            crossover_type: Typ krzyżowania ("OX", "PMX", "CX")
            mutation_type: Typ mutacji ("swap", "insert", "inversion", "scramble")
            selection_type: Typ selekcji ("tournament", "roulette", "rank")
            seed: Ziarno dla generatora liczb losowych
        """
        self.cities = cities
        self.num_cities = len(cities)
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.selection_type = selection_type

        # Inicjalizacja generatora liczb losowych
        if seed is not None:
            np.random.seed(seed)
        self.rng = np.random.RandomState(seed)

        # Obliczenie macierzy odległości
        self.distance_matrix = self._calculate_distance_matrix()

        # Inicjalizacja populacji
        self.population = self._initialize_population()  # Historyczne dane
        self.history = {
            "best_fitness": [],
            "mean_fitness": [],
            "best_route": [],
            "best_distance": [],
            "time_per_gen": [],
            "mutation_rate": [],  # Track mutation rate changes
        }

        # Parametry dla adaptacyjnego współczynnika mutacji
        self.initial_mutation_rate = mutation_rate
        self.stagnation_counter = 0
        self.max_stagnation = (
            20  # Liczba generacji bez poprawy, po której zwiększymy mutation_rate
        )
        self.mutation_rate_increase = 0.02  # O ile zwiększyć mutation_rate
        self.max_mutation_rate = 0.25  # Maksymalny mutation_rate
        self.improvement_threshold = (
            0.001  # Próg poprawy, poniżej którego uznajemy stagnację
        )

    def _calculate_distance_matrix(self) -> np.ndarray:
        """
        Obliczenie macierzy odległości między miastami.

        Returns:
            Macierz odległości
        """
        n = self.num_cities
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Odległość euklidesowa
                    distance_matrix[i, j] = np.sqrt(
                        np.sum((self.cities[i] - self.cities[j]) ** 2)
                    )

        return distance_matrix

    def _initialize_population(self) -> List[np.ndarray]:
        """
        Inicjalizacja populacji losowych tras.

        Returns:
            Lista tras (permutacji miast)
        """
        population = []

        for _ in range(self.population_size):
            # Losowa permutacja miast (trasa)
            route = np.arange(self.num_cities)
            self.rng.shuffle(route)
            population.append(route)

        return population

    def _calculate_route_distance(self, route: np.ndarray) -> float:
        """
        Obliczenie długości trasy.

        Args:
            route: Trasa (permutacja miast)

        Returns:
            Długość trasy
        """
        total_distance = 0

        for i in range(self.num_cities):
            from_city = route[i]
            to_city = route[(i + 1) % self.num_cities]
            total_distance += self.distance_matrix[from_city, to_city]

        return total_distance

    def _calculate_fitness(self, route: np.ndarray) -> float:
        """
        Obliczenie wartości funkcji przystosowania dla trasy.

        Args:
            route: Trasa (permutacja miast)

        Returns:
            Wartość funkcji przystosowania
        """
        distance = self._calculate_route_distance(route)
        # Unikamy dzielenia przez zero
        return 1.0 / (distance + 1e-10)

    def _selection_tournament(self, fitness_scores: List[float]) -> List[int]:
        """
        Selekcja turniejowa.

        Args:
            fitness_scores: Lista wartości funkcji przystosowania

        Returns:
            Indeksy wybranych osobników
        """
        selected_indices = []

        for _ in range(self.population_size):
            # Losowy wybór osobników do turnieju
            tournament_indices = self.rng.choice(
                self.population_size, size=self.tournament_size, replace=False
            )

            # Wybór najlepszego osobnika z turnieju
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            best_tourney_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected_indices.append(best_tourney_idx)

        return selected_indices

    def _selection_roulette(self, fitness_scores: List[float]) -> List[int]:
        """
        Selekcja metodą koła ruletki.

        Args:
            fitness_scores: Lista wartości funkcji przystosowania

        Returns:
            Indeksy wybranych osobników
        """
        # Normalizacja wartości funkcji przystosowania
        total_fitness = sum(fitness_scores)
        relative_fitness = [f / total_fitness for f in fitness_scores]

        # Obliczenie skumulowanej sumy
        cumulative_sum = np.cumsum(relative_fitness)

        # Selekcja metodą koła ruletki
        selected_indices = []
        for _ in range(self.population_size):
            pick = self.rng.random()
            # Znalezienie pierwszego indeksu, dla którego skumulowana suma jest większa od pick
            for i, cum_sum in enumerate(cumulative_sum):
                if cum_sum > pick:
                    selected_indices.append(i)
                    break

        return selected_indices

    def _selection_rank(self, fitness_scores: List[float]) -> List[int]:
        """
        Selekcja rankingowa - osobniki są wybierane proporcjonalnie do ich rangi,
        a nie bezpośrednio do ich wartości przystosowania.

        Args:
            fitness_scores: Lista wartości funkcji przystosowania

        Returns:
            Indeksy wybranych osobników
        """
        # Obliczenie rang (im wyższe przystosowanie, tym wyższa ranga)
        ranks = np.argsort(np.argsort(fitness_scores))
        ranks = ranks + 1  # Rangi od 1 do n

        # Prawdopodobieństwo wyboru oparte na rangach
        probabilities = ranks / np.sum(ranks)

        # Wybór osobników
        selected_indices = self.rng.choice(
            len(fitness_scores),
            size=len(fitness_scores),
            p=probabilities,
            replace=True,
        )

        return selected_indices.tolist()

    def _selection(self, fitness_scores: List[float]) -> List[int]:
        """
        Selekcja osobników na podstawie funkcji przystosowania.

        Args:
            fitness_scores: Lista wartości funkcji przystosowania

        Returns:
            Indeksy wybranych osobników
        """
        if self.selection_type == "tournament":
            return self._selection_tournament(fitness_scores)
        elif self.selection_type == "roulette":
            return self._selection_roulette(fitness_scores)
        elif self.selection_type == "rank":
            return self._selection_rank(fitness_scores)
        else:
            raise ValueError(f"Nieznany typ selekcji: {self.selection_type}")

    def _crossover_OX(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Krzyżowanie metodą Order Crossover (OX).

        Args:
            parent1: Pierwszy rodzic (trasa)
            parent2: Drugi rodzic (trasa)

        Returns:
            Potomek (nowa trasa)
        """
        size = len(parent1)

        # Losowy wybór segmentu do zachowania
        start, end = sorted(self.rng.choice(size, size=2, replace=False))

        # Inicjalizacja potomka
        child = np.full(size, -1, dtype=np.int32)

        # Skopiowanie segmentu z pierwszego rodzica
        child[start : end + 1] = parent1[start : end + 1]

        # Uzupełnienie pozostałych miast w kolejności z drugiego rodzica
        copied_cities = set(parent1[start : end + 1])
        remaining_cities = [city for city in parent2 if city not in copied_cities]

        # Uzupełnienie miejsc przed i po segmencie
        idx = 0
        for i in range(size):
            if i < start or i > end:
                child[i] = remaining_cities[idx]
                idx += 1

        return child

    def _crossover_PMX(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Krzyżowanie metodą Partially Mapped Crossover (PMX).

        Args:
            parent1: Pierwszy rodzic (trasa)
            parent2: Drugi rodzic (trasa)

        Returns:
            Potomek (nowa trasa)
        """
        size = len(parent1)

        # Losowy wybór segmentu do mapowania
        start, end = sorted(self.rng.choice(size, size=2, replace=False))

        # Inicjalizacja potomka
        child = np.full(size, -1, dtype=np.int32)

        # Skopiowanie segmentu z pierwszego rodzica
        child[start : end + 1] = parent1[start : end + 1]

        # Mapowanie genów - używamy podejścia z GPU, które jest bardziej niezawodne
        for i in range(start, end + 1):
            if parent2[i] not in child:
                # Znajdź miejsce dla tego miasta
                j = i
                try:
                    # Używamy podejścia z pętlą while, ale z zabezpieczeniem przed nieskończoną pętlą
                    max_iterations = size  # zabezpieczenie przed nieskończoną pętlą
                    iteration = 0
                    while start <= j <= end and iteration < max_iterations:
                        j = np.where(parent2 == parent1[j])[0][0]
                        iteration += 1

                    # Sprawdź, czy j jest poza segmentem mapowania
                    if j < start or j > end:
                        child[j] = parent2[i]
                except (IndexError, ValueError):
                    # Jeśli napotkamy błąd (np. brak dopasowania), pomijamy
                    continue

        # Uzupełnienie pozostałych miast z drugiego rodzica
        for i in range(size):
            if child[i] == -1:
                child[i] = parent2[i]

        return child

    def _crossover_cx(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Krzyżowanie cykliczne (Cycle Crossover - CX).
        Zachowuje absolutne pozycje miast z rodziców.

        Args:
            parent1: Pierwszy rodzic
            parent2: Drugi rodzic

        Returns:
            Potomek
        """
        size = len(parent1)
        child = np.full(size, -1)  # Inicjalizacja dziecka wartościami -1

        # Indeks startowy
        start_idx = 0

        # Dopóki nie wszystkie pozycje są wypełnione
        while -1 in child:
            # Jeśli pozycja nie jest jeszcze wypełniona
            if child[start_idx] == -1:
                # Rozpoczęcie nowego cyklu
                cycle_start = parent1[start_idx]
                current = cycle_start

                # Wypełnianie cyklu z parent1
                while True:
                    idx_in_parent2 = np.where(parent2 == current)[0][0]
                    child[idx_in_parent2] = current
                    current = parent1[idx_in_parent2]

                    # Jeśli wróciliśmy do początku cyklu, przerywamy
                    if current == cycle_start:
                        break

            # Szukanie następnej niewypełnionej pozycji
            if -1 in child:
                start_idx = np.where(child == -1)[0][0]

        # Wypełnianie pozostałych pozycji z parent2
        for i in range(size):
            if child[i] == -1:
                child[i] = parent2[i]

        return child

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Krzyżowanie rodziców.

        Args:
            parent1: Pierwszy rodzic (trasa)
            parent2: Drugi rodzic (trasa)

        Returns:
            Potomek (nowa trasa)
        """
        if self.crossover_type == "OX":
            return self._crossover_OX(parent1, parent2)
        elif self.crossover_type == "PMX":
            return self._crossover_PMX(parent1, parent2)
        elif self.crossover_type == "CX":
            return self._crossover_cx(parent1, parent2)
        else:
            raise ValueError(f"Nieznany typ krzyżowania: {self.crossover_type}")

    def _mutate_swap(self, route: np.ndarray) -> np.ndarray:
        """
        Mutacja metodą zamiany (swap mutation).

        Args:
            route: Trasa do mutacji

        Returns:
            Zmutowana trasa
        """
        mutated_route = route.copy()

        for i in range(self.num_cities):
            # Dla każdego miasta, z prawdopodobieństwem mutation_rate
            if self.rng.random() < self.mutation_rate:
                # Wybierz losowe miasto do zamiany
                j = self.rng.randint(0, self.num_cities)
                # Zamiana miast
                mutated_route[i], mutated_route[j] = mutated_route[j], mutated_route[i]

        return mutated_route

    def _mutate_insert(self, route: np.ndarray) -> np.ndarray:
        """
        Mutacja metodą wstawienia (insertion mutation).

        Args:
            route: Trasa do mutacji

        Returns:
            Zmutowana trasa
        """
        mutated_route = route.copy()

        # Jeśli wylosowana liczba jest mniejsza od współczynnika mutacji
        if self.rng.random() < self.mutation_rate:
            # Losowy wybór dwóch indeksów
            i, j = sorted(self.rng.choice(self.num_cities, size=2, replace=False))

            # Przesunięcie fragmentu
            if i < j:
                temp = mutated_route[j]
                mutated_route[i + 1 : j + 1] = mutated_route[i:j]
                mutated_route[i] = temp

        return mutated_route

    def _mutate_inversion(self, route: np.ndarray) -> np.ndarray:
        """
        Mutacja metodą inwersji (inversion mutation).

        Args:
            route: Trasa do mutacji

        Returns:
            Zmutowana trasa
        """
        mutated_route = route.copy()

        # Jeśli wylosowana liczba jest mniejsza od współczynnika mutacji
        if self.rng.random() < self.mutation_rate:
            # Losowy wybór dwóch indeksów
            i, j = sorted(self.rng.choice(self.num_cities, size=2, replace=False))

            # Odwrócenie kolejności fragmentu
            if i < j:
                mutated_route[i : j + 1] = np.flip(mutated_route[i : j + 1])

        return mutated_route

    def _mutate_scramble(self, route: np.ndarray) -> np.ndarray:
        """
        Mutacja typu scramble - wybiera podsekwencję miast i miesza ich kolejność.

        Args:
            route: Trasa do mutacji

        Returns:
            Zmutowana trasa
        """
        route_length = len(route)

        # Wybór dwóch losowych punktów
        idx1, idx2 = self.rng.choice(route_length, size=2, replace=False)

        # Upewnienie się, że idx1 < idx2
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        # Wybór podsekwencji do przemieszania
        subsequence = route[idx1 : idx2 + 1].copy()

        # Przemieszanie podsekwencji
        self.rng.shuffle(subsequence)

        # Utworzenie nowej trasy
        new_route = route.copy()
        new_route[idx1 : idx2 + 1] = subsequence

        return new_route

    def _mutate(self, route: np.ndarray) -> np.ndarray:
        """
        Mutacja trasy.

        Args:
            route: Trasa do mutacji

        Returns:
            Zmutowana trasa
        """
        if self.mutation_type == "swap":
            return self._mutate_swap(route)
        elif self.mutation_type == "insert":
            return self._mutate_insert(route)
        elif self.mutation_type == "inversion":
            return self._mutate_inversion(route)
        elif self.mutation_type == "scramble":
            return self._mutate_scramble(route)
        else:
            raise ValueError(f"Nieznany typ mutacji: {self.mutation_type}")

    def evolve(self, verbose: bool = True, save_interval: int = 10) -> Dict[str, Any]:
        """
        Uruchomienie algorytmu genetycznego.

        Args:
            verbose: Czy wyświetlać postęp
            save_interval: Co ile generacji zapisywać najlepszą trasę do historii

        Returns:
            Słownik z wynikami
        """
        # Śledzenie poprzedniej najlepszej odległości dla adaptacyjnego mutation_rate
        previous_best_distance = float("inf")

        # Ewolucja populacji przez zadaną liczbę pokoleń
        for generation in range(self.generations):
            start_time = time.time()

            # Obliczenie wartości funkcji przystosowania dla całej populacji
            fitness_scores = [
                self._calculate_fitness(route) for route in self.population
            ]

            # Znalezienie najlepszego osobnika
            best_idx = np.argmax(fitness_scores)
            best_route = self.population[best_idx]
            best_distance = self._calculate_route_distance(best_route)

            # Adaptacyjny mutation_rate - zwiększanie gdy wyniki stagnują
            if generation > 0:
                # Sprawdzanie poprawy
                improvement = (
                    previous_best_distance - best_distance
                ) / previous_best_distance

                # Jeśli poprawa jest poniżej progu, zwiększamy licznik stagnacji
                if improvement < self.improvement_threshold:
                    self.stagnation_counter += 1
                else:
                    # Resetuj licznik jeśli jest znacząca poprawa
                    self.stagnation_counter = 0

                # Jeśli stagnacja trwa zbyt długo, zwiększamy mutation_rate
                if self.stagnation_counter >= self.max_stagnation:
                    old_rate = self.mutation_rate
                    self.mutation_rate = min(
                        self.mutation_rate + self.mutation_rate_increase,
                        self.max_mutation_rate,
                    )
                    self.stagnation_counter = 0  # Reset licznika

                    if verbose and old_rate != self.mutation_rate:
                        print(
                            f"Generation {generation}: Increasing mutation rate from {old_rate:.4f} to {self.mutation_rate:.4f}"
                        )

            # Aktualizacja poprzedniej najlepszej odległości
            previous_best_distance = best_distance

            # Zapisanie danych do historii
            if generation % save_interval == 0 or generation == self.generations - 1:
                self.history["best_fitness"].append(fitness_scores[best_idx])
                self.history["mean_fitness"].append(np.mean(fitness_scores))
                self.history["best_route"].append(best_route.copy())
                self.history["best_distance"].append(best_distance)
                self.history["mutation_rate"].append(
                    self.mutation_rate
                )  # Zapisanie aktualnego mutation_rate

            # Wyświetlenie postępu
            if verbose and (generation % 10 == 0 or generation == self.generations - 1):
                print(
                    f"Generation {generation}: Best distance = {best_distance:.2f}, Mutation rate = {self.mutation_rate:.4f}"
                )

            # Selekcja rodziców
            selected_indices = self._selection(fitness_scores)

            # Utworzenie nowej populacji, zaczynając od elitarnych osobników
            elite_indices = np.argsort(fitness_scores)[-self.elite_size :]
            next_population = [self.population[i].copy() for i in elite_indices]

            # Uzupełnienie pozostałej części populacji poprzez krzyżowanie i mutację
            while len(next_population) < self.population_size:
                # Losowy wybór dwóch rodziców
                parent1_idx = selected_indices[
                    self.rng.randint(0, len(selected_indices))
                ]
                parent2_idx = selected_indices[
                    self.rng.randint(0, len(selected_indices))
                ]

                # Krzyżowanie
                child = self._crossover(
                    self.population[parent1_idx], self.population[parent2_idx]
                )

                # Mutacja
                child = self._mutate(child)

                # Dodanie dziecka do nowej populacji
                next_population.append(child)

            # Aktualizacja populacji
            self.population = next_population[: self.population_size]

            # Zapisanie czasu generacji
            self.history["time_per_gen"].append(time.time() - start_time)

        # Obliczenie końcowych wyników
        final_fitness_scores = [
            self._calculate_fitness(route) for route in self.population
        ]
        best_idx = np.argmax(final_fitness_scores)
        best_route = self.population[best_idx]
        best_distance = self._calculate_route_distance(best_route)

        if verbose:
            print(f"Final best distance: {best_distance:.2f}")
            print(
                f"Average time per generation: {np.mean(self.history['time_per_gen'])*1000:.2f} ms"
            )

        # Przygotowanie wyników
        results = {
            "best_route": best_route.copy(),
            "best_distance": best_distance,
            "best_fitness": final_fitness_scores[best_idx],
            "mean_fitness": np.mean(final_fitness_scores),
            "history": self.history,
            "parameters": {
                "population_size": self.population_size,
                "elite_size": self.elite_size,
                "mutation_rate": self.mutation_rate,
                "generations": self.generations,
                "tournament_size": self.tournament_size,
                "crossover_type": self.crossover_type,
                "mutation_type": self.mutation_type,
                "selection_type": self.selection_type,
            },
        }

        return results

    def get_best_route(self) -> Tuple[np.ndarray, float]:
        """
        Pobranie najlepszej trasy z obecnej populacji.

        Returns:
            Krotka (najlepsza trasa, długość trasy)
        """
        fitness_scores = [self._calculate_fitness(route) for route in self.population]
        best_idx = np.argmax(fitness_scores)
        best_route = self.population[best_idx]
        best_distance = self._calculate_route_distance(best_route)

        return best_route, best_distance
