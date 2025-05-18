"""
Funkcje do wizualizacji wyników algorytmu genetycznego dla problemu komiwojażera.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from typing import List, Dict, Any, Tuple, Optional
import matplotlib

matplotlib.rcParams["font.family"] = "DejaVu Sans"


def plot_route(
    cities: np.ndarray,
    route: np.ndarray,
    title: str = "Trasa",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 100,
    color: str = "tab:blue",
    marker_size: int = 100,
    line_width: int = 1.5,
    ax=None,
):
    """
    Wizualizacja trasy dla problemu komiwojażera.

    Args:
        cities: Tablica współrzędnych miast (Nx2)
        route: Permutacja miast (indeksy)
        title: Tytuł wykresu
        save_path: Ścieżka do zapisu wizualizacji
        show: Czy wyświetlić wizualizację
        figsize: Rozmiar wykresu
        dpi: Rozdzielczość wykresu
        color: Kolor linii
        marker_size: Rozmiar znaczników miast
        line_width: Grubość linii
        ax: Obiekt osi do rysowania (opcjonalny)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        created_ax = True
    else:
        created_ax = False

    # Rysowanie miast
    ax.scatter(cities[:, 0], cities[:, 1], s=marker_size, c="red", alpha=0.7, zorder=2)

    # Rysowanie trasy
    for i in range(len(route)):
        from_city = route[i]
        to_city = route[(i + 1) % len(route)]
        ax.plot(
            [cities[from_city, 0], cities[to_city, 0]],
            [cities[from_city, 1], cities[to_city, 1]],
            c=color,
            linewidth=line_width,
            alpha=0.6,
            zorder=1,
        )

    # Numerowanie miast
    for i, city in enumerate(cities):
        ax.text(city[0] + 0.1, city[1] + 0.1, str(i), fontsize=8, zorder=3)

    # Ustawienia wykresu
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.3)

    # Dopasowanie zakresu osi
    margin = 0.1
    x_min, x_max = np.min(cities[:, 0]), np.max(cities[:, 0])
    y_min, y_max = np.min(cities[:, 1]), np.max(cities[:, 1])
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show and created_ax:
        plt.show()

    return ax


def plot_fitness_history(
    history: Dict[str, List],
    title: str = "Historia funkcji przystosowania",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 100,
):
    """
    Wizualizacja historii wartości funkcji przystosowania.

    Args:
        history: Słownik z historią algorytmu
        title: Tytuł wykresu
        save_path: Ścieżka do zapisu wizualizacji
        show: Czy wyświetlić wizualizację
        figsize: Rozmiar wykresu
        dpi: Rozdzielczość wykresu
    """
    plt.figure(figsize=figsize, dpi=dpi)

    generations = np.arange(len(history["best_fitness"]))

    plt.plot(
        generations,
        history["best_fitness"],
        "b-",
        linewidth=2,
        label="Najlepsze przystosowanie",
    )
    plt.plot(
        generations,
        history["mean_fitness"],
        "r--",
        linewidth=1.5,
        label="Średnie przystosowanie",
    )

    plt.title(title)
    plt.xlabel("Generacja")
    plt.ylabel("Wartość funkcji przystosowania")
    plt.grid(alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()


def plot_distance_history(
    history: Dict[str, List],
    title: str = "Historia długości trasy",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 100,
):
    """
    Wizualizacja historii długości najlepszej trasy.

    Args:
        history: Słownik z historią algorytmu
        title: Tytuł wykresu
        save_path: Ścieżka do zapisu wizualizacji
        show: Czy wyświetlić wizualizację
        figsize: Rozmiar wykresu
        dpi: Rozdzielczość wykresu
    """
    plt.figure(figsize=figsize, dpi=dpi)

    generations = np.arange(len(history["best_distance"]))

    plt.plot(generations, history["best_distance"], "g-", linewidth=2)

    plt.title(title)
    plt.xlabel("Generacja")
    plt.ylabel("Długość trasy")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()


def plot_time_history(
    history: Dict[str, List],
    title: str = "Czas wykonania generacji",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 100,
):
    """
    Wizualizacja czasu wykonania dla każdej generacji.

    Args:
        history: Słownik z historią algorytmu
        title: Tytuł wykresu
        save_path: Ścieżka do zapisu wizualizacji
        show: Czy wyświetlić wizualizację
        figsize: Rozmiar wykresu
        dpi: Rozdzielczość wykresu
    """
    plt.figure(figsize=figsize, dpi=dpi)

    generations = np.arange(len(history["time_per_gen"]))
    times_ms = np.array(history["time_per_gen"]) * 1000  # Konwersja na ms

    plt.plot(generations, times_ms, "m-", linewidth=1.5, alpha=0.7)
    plt.axhline(
        y=np.mean(times_ms),
        color="r",
        linestyle="--",
        label=f"Średni czas: {np.mean(times_ms):.2f} ms",
    )

    plt.title(title)
    plt.xlabel("Generacja")
    plt.ylabel("Czas (ms)")
    plt.grid(alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()


def create_evolution_animation(
    cities: np.ndarray,
    history: Dict[str, List],
    save_path: Optional[str] = None,
    interval: int = 200,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 100,
    title: str = "Ewolucja trasy w algorytmie genetycznym",
    save_interval: int = 10,
):
    """
    Tworzenie animacji ewolucji trasy w algorytmie genetycznym.

    Args:
        cities: Tablica współrzędnych miast (Nx2)
        history: Słownik z historią algorytmu
        save_path: Ścieżka do zapisu animacji
        interval: Interwał czasowy między klatkami (ms)
        figsize: Rozmiar wykresu
        dpi: Rozdzielczość wykresu
        title: Tytuł główny animacji
        save_interval: Interwał zapisu historii w algorytmie genetycznym

    Returns:
        Obiekt animacji
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    routes = history["best_route"]
    distances = history["best_distance"]

    # Inicjalizacja wykresu
    scatter = ax.scatter([], [], s=100, c="red", alpha=0.7, zorder=2)

    (line,) = ax.plot([], [], "b-", linewidth=1.5, alpha=0.6, zorder=1)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    gen_title = ax.set_title("")

    # Dopasowanie zakresu osi
    margin = 0.1
    x_min, x_max = np.min(cities[:, 0]), np.max(cities[:, 0])
    y_min, y_max = np.min(cities[:, 1]), np.max(cities[:, 1])
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.3)

    # Numerowanie miast
    for i, city in enumerate(cities):
        ax.text(city[0] + 0.1, city[1] + 0.1, str(i), fontsize=8, zorder=3)

    def init():
        scatter.set_offsets(cities)
        line.set_data([], [])
        gen_title.set_text("")
        return scatter, line, gen_title

    def update(frame):
        route = routes[frame]

        # Przygotowanie danych dla linii
        x_route = [cities[city, 0] for city in route]
        y_route = [
            cities[city, 1] for city in route
        ]  # Dodanie ostatniego połączenia (powrót do pierwszego miasta)
        x_route.append(cities[route[0], 0])
        y_route.append(cities[route[0], 1])
        line.set_data(x_route, y_route)
        # Oblicz prawdziwy numer generacji (frame * save_interval)
        actual_generation = frame * save_interval
        gen_title.set_text(
            f"Generacja {actual_generation}: Długość trasy = {distances[frame]:.2f}"
        )

        return scatter, line, gen_title

    # Create a list of frames with a 2-second pause at the end
    frames = list(range(len(routes)))

    # Add extra frames at the end (10 frames = 2 seconds at fps=5)
    pause_frames = [len(routes) - 1] * 10
    all_frames = frames + pause_frames

    # Utworzenie animacji
    animation = FuncAnimation(
        fig, update, frames=all_frames, init_func=init, blit=True, interval=interval
    )

    if save_path:
        animation.save(save_path, writer="pillow", fps=5)

    plt.close()

    return animation


def plot_parameter_comparison(
    parameter_name: str,
    parameter_values: List,
    distances: List[float],
    title: str = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 100,
):
    """
    Wizualizacja wpływu parametru na jakość rozwiązania.

    Args:
        parameter_name: Nazwa parametru
        parameter_values: Lista wartości parametru
        distances: Lista odpowiadających długości tras
        title: Tytuł wykresu
        save_path: Ścieżka do zapisu wizualizacji
        show: Czy wyświetlić wizualizację
        figsize: Rozmiar wykresu
        dpi: Rozdzielczość wykresu
    """
    plt.figure(figsize=figsize, dpi=dpi)

    # Konwersja wartości parametru na stringi dla etykiet
    x_labels = [str(val) for val in parameter_values]

    plt.bar(x_labels, distances, color="skyblue", alpha=0.7)
    plt.plot(x_labels, distances, "ro-", linewidth=2, markersize=8)

    # Dodanie etykiet wartości
    for i, distance in enumerate(distances):
        plt.text(
            i, distance + 0.5, f"{distance:.2f}", ha="center", va="bottom", fontsize=9
        )

    plt.title(title or f"Wpływ parametru '{parameter_name}' na długość trasy")
    plt.xlabel(parameter_name)
    plt.ylabel("Długość trasy")
    plt.grid(alpha=0.3, axis="y")

    # Optymalizacja wyświetlania etykiet dla dużej liczby wartości
    if len(parameter_values) > 6:
        plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()


def plot_parameter_boxplot(
    parameter_name: str,
    parameter_values: List,
    distance_values: List[List[float]],
    title: str = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 100,
):
    """
    Wizualizacja wpływu parametru na jakość rozwiązania za pomocą boxplotów.

    Args:
        parameter_name: Nazwa parametru
        parameter_values: Lista wartości parametru
        distance_values: Lista list z długościami tras dla każdej wartości parametru
        title: Tytuł wykresu
        save_path: Ścieżka do zapisu wizualizacji
        show: Czy wyświetlić wizualizację
        figsize: Rozmiar wykresu
        dpi: Rozdzielczość wykresu
    """
    plt.figure(figsize=figsize, dpi=dpi)

    # Konwersja wartości parametru na stringi dla etykiet
    x_labels = [str(val) for val in parameter_values]

    # Utworzenie boxplotów
    box = plt.boxplot(distance_values, labels=x_labels, patch_artist=True)

    # Kolorowanie boxplotów
    colors = plt.cm.viridis(np.linspace(0, 1, len(parameter_values)))
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

    # Obliczenie średniej dla każdej wartości parametru i dodanie linii łączącej średnie
    means = [np.mean(dist) for dist in distance_values]
    plt.plot(range(1, len(means) + 1), means, "r-", linewidth=2, label="Średnia")

    # Dodanie etykiet wartości średnich
    for i, mean_val in enumerate(means):
        plt.text(
            i + 1,
            mean_val + 0.5,
            f"{mean_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.title(title or f"Wpływ parametru '{parameter_name}' na długość trasy (boxplot)")
    plt.xlabel(parameter_name)
    plt.ylabel("Długość trasy")
    plt.grid(alpha=0.3, axis="y")
    plt.legend()

    # Optymalizacja wyświetlania etykiet dla dużej liczby wartości
    if len(parameter_values) > 6:
        plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()


def plot_optimization_results(
    results: Dict[str, Any],
    save_dir: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 100,
):
    """
    Wizualizacja wyników optymalizacji parametrów.

    Args:
        results: Słownik z wynikami optymalizacji
        save_dir: Katalog do zapisu wizualizacji
        show: Czy wyświetlić wizualizację
        figsize: Rozmiar wykresu
        dpi: Rozdzielczość wykresu
    """
    # Przygotowanie danych dla wizualizacji
    parameters = results["parameters"]

    # Lista parametrów do wizualizacji
    param_names = ["population_size", "elite_size", "mutation_rate", "tournament_size"]
    categorical_params = ["crossover_type", "mutation_type", "selection_type"]

    # Wizualizacja parametrów liczbowych (bar chart + boxplot)
    for param in param_names:
        if param in parameters:
            # Zebranie danych dla danego parametru
            param_values = sorted(list(set(results[param])))
            param_distances = []
            param_distances_all = []  # Dla boxplotów - pełne listy wartości

            for value in param_values:
                # Filtrowanie wyników dla danej wartości parametru
                filtered_indices = [
                    i for i, v in enumerate(results[param]) if v == value
                ]
                # Średnia dla wykresu słupkowego
                distances = [results["best_distance"][i] for i in filtered_indices]
                avg_distance = np.mean(distances)
                param_distances.append(avg_distance)
                param_distances_all.append(distances)

            # Tworzenie tytułu
            title = f"Wpływ parametru '{param}' na długość trasy"

            # Tworzenie ścieżek do zapisu
            save_path_bar = None
            save_path_box = None
            if save_dir:
                save_path_bar = os.path.join(save_dir, f"param_{param}.png")
                save_path_box = os.path.join(save_dir, f"param_{param}_boxplot.png")

            # Wykres słupkowy - średnie wartości
            plot_parameter_comparison(
                parameter_name=param,
                parameter_values=param_values,
                distances=param_distances,
                title=title,
                save_path=save_path_bar,
                show=show,
                figsize=figsize,
                dpi=dpi,
            )

            # Boxplot - rozkład wartości
            if all(len(dist) > 1 for dist in param_distances_all):
                plot_parameter_boxplot(
                    parameter_name=param,
                    parameter_values=param_values,
                    distance_values=param_distances_all,
                    title=f"{title} (boxplot)",
                    save_path=save_path_box,
                    show=show,
                    figsize=figsize,
                    dpi=dpi,
                )

    # Wizualizacja parametrów kategorycznych (tylko bar chart)
    for param in categorical_params:
        if param in parameters:
            # Zebranie danych dla danego parametru
            param_values = sorted(list(set(results[param])))
            param_distances = []

            for value in param_values:
                # Filtrowanie wyników dla danej wartości parametru
                filtered_indices = [
                    i for i, v in enumerate(results[param]) if v == value
                ]
                avg_distance = np.mean(
                    [results["best_distance"][i] for i in filtered_indices]
                )
                param_distances.append(avg_distance)

            # Tworzenie tytułu
            title = f"Wpływ parametru '{param}' na długość trasy"

            # Tworzenie ścieżki do zapisu
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, f"param_{param}.png")

            # Wizualizacja
            plot_parameter_comparison(
                parameter_name=param,
                parameter_values=param_values,
                distances=param_distances,
                title=title,
                save_path=save_path,
                show=show,
                figsize=figsize,
                dpi=dpi,
            )


def compare_methods(
    method_names: List[str],
    distances: List[float],
    times: List[float],
    title: str = "Porównanie metod",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 100,
):
    """
    Porównanie różnych metod (np. CPU vs GPU, różne zestawy parametrów).

    Args:
        method_names: Nazwy metod
        distances: Lista długości tras dla każdej metody
        times: Lista czasów wykonania dla każdej metody
        title: Tytuł wykresu
        save_path: Ścieżka do zapisu wizualizacji
        show: Czy wyświetlić wizualizację
        figsize: Rozmiar wykresu
        dpi: Rozdzielczość wykresu
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # Wykres długości tras
    bars1 = ax1.bar(method_names, distances, color="skyblue", alpha=0.7)
    ax1.set_title("Długość trasy")
    ax1.set_ylabel("Długość")
    ax1.grid(alpha=0.3, axis="y")

    # Dodanie etykiet wartości
    for bar, dist in zip(bars1, distances):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{dist:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Wykres czasów wykonania
    bars2 = ax2.bar(method_names, times, color="salmon", alpha=0.7)
    ax2.set_title("Czas wykonania")
    ax2.set_ylabel("Czas (s)")
    ax2.grid(alpha=0.3, axis="y")

    # Dodanie etykiet wartości
    for bar, time_val in zip(bars2, times):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{time_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Ustawienia ogólne
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Dostosowanie miejsca dla tytułu

    # Optymalizacja wyświetlania etykiet dla dużej liczby metod
    if len(method_names) > 4:
        for ax in [ax1, ax2]:
            ax.set_xticklabels(method_names, rotation=45, ha="right")

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()


def plot_mutation_rate_history(
    history: Dict[str, List],
    title: str = "Historia współczynnika mutacji",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 100,
):
    """
    Wizualizacja zmian współczynnika mutacji w trakcie ewolucji.

    Args:
        history: Słownik z historią algorytmu
        title: Tytuł wykresu
        save_path: Ścieżka do zapisu wizualizacji
        show: Czy wyświetlić wizualizację
        figsize: Rozmiar wykresu
        dpi: Rozdzielczość wykresu
    """
    plt.figure(figsize=figsize, dpi=dpi)

    # Jeżeli historia współczynnika mutacji jest pusta, nie rysujemy wykresu
    if not history.get("mutation_rate"):
        plt.title("Brak danych o współczynniku mutacji")
        plt.figtext(
            0.5,
            0.5,
            "Brak danych o współczynniku mutacji",
            ha="center",
            va="center",
            fontsize=12,
        )
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        return

    generations = np.arange(len(history["mutation_rate"]))
    save_interval = (
        len(history["best_distance"]) // len(generations) if len(generations) > 0 else 1
    )

    # Rysowanie wykresu
    plt.plot(generations * save_interval, history["mutation_rate"], "g-", linewidth=2)

    # Jeśli występują zmiany mutation_rate, oznaczamy je dodatkowo punktami
    mutation_rates = np.array(history["mutation_rate"])
    changes = np.where(mutation_rates[1:] != mutation_rates[:-1])[0] + 1
    if len(changes) > 0:
        plt.plot(changes * save_interval, mutation_rates[changes], "ro", markersize=6)

    plt.title(title)
    plt.xlabel("Generacja")
    plt.ylabel("Współczynnik mutacji")
    plt.grid(alpha=0.3)

    # Dodanie informacji o częstotliwości adaptacji
    if len(changes) > 0:
        plt.figtext(
            0.02,
            0.02,
            f"Liczba zmian: {len(changes)}",
            ha="left",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7),
        )

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
