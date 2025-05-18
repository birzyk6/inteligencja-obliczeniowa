# Problem Komiwojażera z wykorzystaniem algorytmu genetycznego

## Opis projektu

Ten projekt implementuje rozwiązanie klasycznego problemu komiwojażera (Traveling Salesman Problem, TSP) przy użyciu algorytmu genetycznego. Problem komiwojażera polega na znalezieniu najkrótszej trasy, która przechodzi przez wszystkie miasta dokładnie raz i wraca do miasta początkowego.

Główne cechy projektu:

-   Implementacja algorytmu genetycznego na CPU
-   Optymalizacja parametrów algorytmu genetycznego
-   Wizualizacja wyników i animacje ewolucji rozwiązania
-   Analiza wpływu różnych parametrów na jakość rozwiązania

## Struktura projektu

```
komiwoj/
│
├── src/                        # Kod źródłowy
│   ├── ga.py                   # Implementacja algorytmu genetycznego
│   ├── optimizer.py            # Moduł optymalizacji parametrów
│   ├── visualization.py        # Funkcje do wizualizacji wyników
│   └── main.py                 # Główny plik wykonawczy
│
├── data/                       # Dane do analizy
│
├── results/                    # Wyniki i wizualizacje
│
└── requirements.txt            # Zależności projektu
```

## Wymagania

-   Python 3.8+
-   NumPy
-   Matplotlib
-   pandas
-   scikit-learn
-   tqdm
-   joblib

Instalacja zależności:

```bash
pip install -r requirements.txt
```

## Użycie

### Uruchomienie podstawowe

```bash
python src/main.py
```

### Parametry wiersza poleceń

-   `--num-cities`: Liczba miast (domyślnie: 50)
-   `--generations`: Liczba pokoleń (domyślnie: 500)
-   `--opt-configs`: Liczba konfiguracji do przetestowania w optymalizacji (domyślnie: 10)
-   `--save-path`: Ścieżka do zapisu wyników (domyślnie: ../results)
-   `--seed`: Ziarno dla generatora liczb losowych (domyślnie: 42)
-   `--skip-animation`: Pomiń tworzenie animacji
-   `--animation-interval`: Co ile generacji zapisywać klatkę animacji (domyślnie: 10)

Przykład:

```bash
python src/main.py --num-cities 50 --generations 1000 --opt-configs 20
```

## Algorytm genetyczny

Implementacja algorytmu genetycznego obejmuje następujące operacje:

1. **Inicjalizacja populacji**: Losowe generowanie populacji początkowej.
2. **Selekcja**: Wybór osobników do reprodukcji. Zaimplementowane metody:
    - Selekcja turniejowa
    - Selekcja metodą koła ruletki
3. **Krzyżowanie**: Tworzenie nowych osobników poprzez kombinację genów rodziców. Zaimplementowane metody:
    - Order Crossover (OX)
    - Partially Mapped Crossover (PMX)
4. **Mutacja**: Wprowadzanie losowych zmian w genotypie. Zaimplementowane metody:
    - Mutacja przez zamianę (swap)
    - Mutacja przez wstawienie (insertion)
    - Mutacja przez inwersję (inversion)
5. **Elitaryzm**: Zachowanie najlepszych osobników między pokoleniami.

## Optymalizacja parametrów

Moduł optymalizacji pozwala na znalezienie optymalnych parametrów algorytmu genetycznego:

-   Rozmiar populacji
-   Liczba osobników elitarnych
-   Współczynnik mutacji
-   Rozmiar turnieju (dla selekcji turniejowej)
-   Typ krzyżowania
-   Typ mutacji
-   Typ selekcji

## Wizualizacje

Projekt generuje różne wizualizacje pomocne w analizie wyników:

-   Wizualizacja najlepszej trasy
-   Wykresy historii funkcji przystosowania
-   Wykresy historii długości trasy
-   Wykresy czasu wykonania
-   Animacje ewolucji rozwiązania
-   Porównania metod i parametrów

## Autorzy

Bartosz Irzyk

## Licencja

MIT
