# System oceny ryzyka inwestycji oparty na logice rozmytej

## Wprowadzenie

System wspomagania decyzji inwestycyjnych opracowany w ramach tego projektu wykorzystuje logikę rozmytą do modelowania niejednoznaczności i niepewności występujących w procesach decyzyjnych związanych z inwestycjami.

## Zmienne lingwistyczne i funkcje przynależności

System wykorzystuje trzy zmienne wejściowe i jedną zmienną wyjściową.

### Zmienne wejściowe

#### 1. Zmienność rynku (market_volatility)

Zmienność rynku odzwierciedla poziom nieprzewidywalności i wahań na rynku, na którym działa przedsiębiorstwo.

**Funkcje przynależności:**

-   Niska (low): funkcja Gaussa z centrum w 10 i szerokością 15
-   Średnia (medium): funkcja Gaussa z centrum w 50 i szerokością 15
-   Wysoka (high): funkcja Gaussa z centrum w 90 i szerokością 15

#### 2. Kondycja finansowa przedsiębiorstwa (financial_health)

Kondycja finansowa ocenia stabilność i siłę finansową przedsiębiorstwa.

**Funkcje przynależności:**

-   Słaba (poor): funkcja Z-kształtna z parametrami [20, 40]
-   Przeciętna (average): funkcja Gaussa z centrum w 50 i szerokością 15
-   Doskonała (excellent): funkcja S-kształtna z parametrami [60, 80]

#### 3. Potencjał wzrostu branży (industry_growth)

Potencjał wzrostu branży określa perspektywy rozwojowe sektora, w którym działa przedsiębiorstwo.

**Funkcje przynależności:**

-   Spadek (declining): funkcja dzwonowa (gbellmf) z parametrami [20, 2, 10]
-   Stabilność (stable): funkcja dzwonowa (gbellmf) z parametrami [20, 2, 50]
-   Dynamiczny wzrost (booming): funkcja dzwonowa (gbellmf) z parametrami [20, 2, 90]

### Zmienna wyjściowa

#### Ocena ryzyka inwestycji (investment_risk)

Ocena ryzyka inwestycji określa poziom bezpieczeństwa lub ryzyka związanego z daną inwestycją.

**Funkcje przynależności:**

-   Bardzo bezpieczna (very_safe): funkcja Gaussa z centrum w 10 i szerokością 10
-   Bezpieczna (safe): funkcja Gaussa z centrum w 30 i szerokością 10
-   Umiarkowane ryzyko (moderate): funkcja Gaussa z centrum w 50 i szerokością 10
-   Ryzykowna (risky): funkcja Gaussa z centrum w 70 i szerokością 10
-   Bardzo ryzykowna (very_risky): funkcja Gaussa z centrum w 90 i szerokością 10

## Baza reguł

System zawiera 18 reguł wnioskowania, które obejmują różne kombinacje zmiennych wejściowych i określają odpowiednie wartości zmiennej wyjściowej. Poniżej przedstawiono kilka przykładowych reguł:

1. JEŻELI zmienność rynku jest niska I kondycja finansowa jest doskonała I potencjał wzrostu branży jest dynamiczny, TO ryzyko inwestycji jest bardzo bezpieczne.

2. JEŻELI zmienność rynku jest niska I kondycja finansowa jest doskonała I potencjał wzrostu branży jest stabilny, TO ryzyko inwestycji jest bezpieczne.

3. JEŻELI zmienność rynku jest średnia I kondycja finansowa jest przeciętna I potencjał wzrostu branży jest stabilny LUB dynamiczny, TO ryzyko inwestycji jest umiarkowane.

4. JEŻELI zmienność rynku jest wysoka I kondycja finansowa jest słaba, TO ryzyko inwestycji jest bardzo ryzykowne.

Pełny zestaw reguł znajduje się w kodzie źródłowym systemu.

## Metoda defuzyfikacji

W projekcie zastosowano metodę centroidu (środka ciężkości) do defuzyfikacji. Jest to jedna z najbardziej popularnych metod defuzyfikacji, która oblicza wartość wyjściową jako środek ciężkości obszaru pod krzywą funkcji przynależności zmiennej wyjściowej. Metoda ta oferuje kompromis między dokładnością a złożonością obliczeniową.

## Przykładowe wnioskowanie

Aby zilustrować działanie systemu, przeprowadzono analizę trzech przykładowych scenariuszy inwestycyjnych:

### Scenariusz 1: Niska zmienność rynku, doskonała kondycja finansowa, dynamiczny wzrost branży

-   Zmienność rynku: 20
-   Kondycja finansowa: 80
-   Potencjał wzrostu branży: 90
-   **Wynik oceny ryzyka: 16.67/100 (Bardzo bezpieczna inwestycja)**

### Scenariusz 2: Średnia zmienność rynku, przeciętna kondycja finansowa, stabilny wzrost branży

-   Zmienność rynku: 50
-   Kondycja finansowa: 50
-   Potencjał wzrostu branży: 50
-   **Wynik oceny ryzyka: 49.72/100 (Umiarkowane ryzyko inwestycji)**

### Scenariusz 3: Wysoka zmienność rynku, słaba kondycja finansowa, spadek branży

-   Zmienność rynku: 80
-   Kondycja finansowa: 30
-   Potencjał wzrostu branży: 20
-   **Wynik oceny ryzyka: 87.57/100 (Bardzo ryzykowna inwestycja)**

## Wizualizacja wyników

System generuje następujące wizualizacje:

1. **Funkcje przynależności** - dla wszystkich zmiennych wejściowych i wyjściowej, pokazujące zaawansowane kształty funkcji wykorzystujących modele Gaussa, Z-kształtne, S-kształtne i dzwonowe.

2. **Wykresy konturowe** - pokazujące poziomy ryzyka dla różnych kombinacji zmienności rynku i kondycji finansowej, przy trzech różnych poziomach potencjału wzrostu branży (10 - spadek, 50 - stabilność, 90 - dynamiczny wzrost).

3. **Wykres powierzchniowy 3D** - pokazujący zależność między zmiennością rynku, kondycją finansową a ryzykiem inwestycyjnym przy stałym poziomie potencjału wzrostu branży równym 50 (stabilność).

### Wykresy funkcji przynależności dla poszczególnych zmiennych

#### Zmienność rynku

![Funkcje przynależności - Zmienność rynku](./results/market_volatility_membership.png)

Na powyższym wykresie przedstawiono funkcje przynależności dla zmiennej "Zmienność rynku":

-   **Niska** (kolor niebieski): funkcja Gaussa z centrum w punkcie 10
-   **Średnia** (kolor pomarańczowy): funkcja Gaussa z centrum w punkcie 50
-   **Wysoka** (kolor zielony): funkcja Gaussa z centrum w punkcie 90

#### Kondycja finansowa

![Funkcje przynależności - Kondycja finansowa](./results/financial_health_membership.png)

Na powyższym wykresie przedstawiono funkcje przynależności dla zmiennej "Kondycja finansowa":

-   **Słaba** (kolor niebieski): funkcja Z-kształtna z parametrami [20, 40]
-   **Przeciętna** (kolor pomarańczowy): funkcja Gaussa z centrum w punkcie 50
-   **Doskonała** (kolor zielony): funkcja S-kształtna z parametrami [60, 80]

#### Potencjał wzrostu branży

![Funkcje przynależności - Potencjał wzrostu branży](./results/industry_growth_membership.png)

Na powyższym wykresie przedstawiono funkcje przynależności dla zmiennej "Potencjał wzrostu branży":

-   **Spadek** (kolor niebieski): funkcja dzwonowa z parametrami [20, 2, 10]
-   **Stabilność** (kolor pomarańczowy): funkcja dzwonowa z parametrami [20, 2, 50]
-   **Dynamiczny wzrost** (kolor zielony): funkcja dzwonowa z parametrami [20, 2, 90]

#### Ryzyko inwestycji

![Funkcje przynależności - Ryzyko inwestycji](./results/investment_risk_membership.png)

Na powyższym wykresie przedstawiono funkcje przynależności dla zmiennej wyjściowej "Ryzyko inwestycji":

-   **Bardzo bezpieczne** (kolor niebieski): funkcja Gaussa z centrum w punkcie 10
-   **Bezpieczne** (kolor pomarańczowy): funkcja Gaussa z centrum w punkcie 30
-   **Umiarkowane** (kolor zielony): funkcja Gaussa z centrum w punkcie 50
-   **Ryzykowne** (kolor czerwony): funkcja Gaussa z centrum w punkcie 70
-   **Bardzo ryzykowne** (kolor fioletowy): funkcja Gaussa z centrum w punkcie 90

### Zbiorczy wykres funkcji przynależności

Poniżej przedstawiono zbiorcze funkcje przynależności dla wszystkich zmiennych wejściowych i wyjściowej:

![Funkcje przynależności](./results/membership_functions.png)

Na powyższym wykresie widoczne są:

-   Funkcje przynależności dla zmienności rynku (pierwsze od góry)
-   Funkcje przynależności dla kondycji finansowej (drugie od góry)
-   Funkcje przynależności dla potencjału wzrostu branży (trzecie od góry)
-   Funkcje przynależności dla ryzyka inwestycji (ostatnie na dole)

### Wykresy konturowe

Poniżej przedstawiono wykresy konturowe pokazujące zależność ryzyka inwestycyjnego od zmienności rynku i kondycji finansowej przy trzech różnych poziomach potencjału wzrostu branży:

![Wykresy konturowe ryzyka inwestycyjnego](./results/risk_contour_plots.png)

Na wykresach wyraźnie widać, jak zmienia się ryzyko inwestycyjne w zależności od kombinacji zmienności rynku (oś X) i kondycji finansowej (oś Y) przy:

-   Potencjale wzrostu branży = 10 (spadek) - pierwszy wykres
-   Potencjale wzrostu branży = 50 (stabilność) - środkowy wykres
-   Potencjale wzrostu branży = 90 (dynamiczny wzrost) - ostatni wykres

Jaśniejsze kolory oznaczają wyższe ryzyko, ciemniejsze - niższe ryzyko inwestycyjne.

### Wykres powierzchniowy 3D

Poniżej przedstawiono wykres powierzchniowy 3D pokazujący zależność między zmiennością rynku, kondycją finansową a ryzykiem inwestycyjnym przy stałym poziomie potencjału wzrostu branży równym 50 (stabilność):

![Wykres powierzchniowy 3D ryzyka inwestycyjnego](./results/risk_3d_plot.png)

Wykres ten pozwala na przestrzenną wizualizację wpływu zmienności rynku i kondycji finansowej na ryzyko inwestycyjne. Wysokość powierzchni odpowiada poziomowi ryzyka - im wyżej, tym wyższe ryzyko inwestycyjne.

## Wnioski

1. **Skuteczność zaawansowanych funkcji przynależności** - Zastosowanie funkcji Gaussa, Z-kształtnych, S-kształtnych i dzwonowych pozwala na bardziej dokładne modelowanie niepewności niż funkcje trójkątne. Jak widać na indywidualnych wykresach zmiennych, funkcje te zapewniają płynne przejścia między kategoriami lingwistycznymi, co lepiej oddaje charakter rzeczywistych ocen eksperckich.

2. **Precyzyjne wartości liczbowe** - System generuje precyzyjne oceny liczbowe (np. 16.67/100, 49.72/100, 87.57/100), które można łatwo interpretować jako stopień przynależności do danej kategorii ryzyka, dając bardziej zniuansowane wyniki niż systemy binarne.

3. **Znaczenie kondycji finansowej** - Analiza wyników pokazuje, że kondycja finansowa przedsiębiorstwa ma kluczowe znaczenie dla oceny ryzyka inwestycji. Przejście od słabej (30) do doskonałej (80) kondycji finansowej może zmienić ocenę ryzyka z "bardzo ryzykownej" na "bardzo bezpieczną", nawet przy innych niezmienionych parametrach. Widać to wyraźnie zarówno na wykresach konturowych, jak i na wykresie 3D, gdzie gradient zmian jest szczególnie stromy wzdłuż osi kondycji finansowej.

4. **Wpływ zmienności rynku** - Wysoka zmienność rynku (80) wyraźnie zwiększa ryzyko inwestycji, nawet przy dobrych pozostałych parametrach. Kształt funkcji przynależności dla zmienności rynku (widoczny na indywidualnym wykresie) pokazuje, jak płynnie zmienia się ocena zmienności od niskiej do wysokiej, co przekłada się na odpowiednią gradację ryzyka.

5. **Interakcje między zmiennymi** - System ujawnia złożone, nieliniowe interakcje między trzema zmiennymi wejściowymi, co widoczne jest szczególnie na wykresach konturowych i powierzchniowym. Dzięki indywidualnym wykresom funkcji przynależności można lepiej zrozumieć sposób modelowania każdej ze zmiennych i ich wzajemne oddziaływanie.

6. **Charakterystyka funkcji przynależności** - Porównując wykresy poszczególnych zmiennych, można zauważyć:

    - Funkcje Gaussa dla zmienności rynku zapewniają symetryczne i płynne przejścia między kategoriami
    - Funkcje Z-kształtna i S-kształtna dla kondycji finansowej modelują asymetryczne zjawiska, gdzie przejście między kategoriami nie jest symetryczne
    - Funkcje dzwonowe dla potencjału wzrostu branży oferują węższe zakresy centralnych wartości, co odpowiada bardziej precyzyjnemu określeniu stabilności branży

7. **Elastyczność systemu** - Implementacja w języku Python z wykorzystaniem biblioteki scikit-fuzzy umożliwia łatwą modyfikację i rozszerzanie systemu o dodatkowe zmienne, funkcje przynależności czy reguły wnioskowania. Indywidualne wykresy zmiennych ułatwiają precyzyjne dostrojenie kształtu funkcji przynależności.

8. **Potencjalne kierunki rozwoju** - System mógłby zostać rozbudowany o:
    - Dodatkowe zmienne wejściowe (np. wskaźniki makroekonomiczne, ryzyko polityczne)
    - Dynamiczną aktualizację reguł na podstawie danych historycznych
    - Interfejs użytkownika umożliwiający interaktywną analizę scenariuszy
    - Integrację z systemami pozyskiwania danych rynkowych w czasie rzeczywistym

System oparty na logice rozmytej z wykorzystaniem zaawansowanych funkcji przynależności stanowi wartościowe narzędzie wspomagające podejmowanie decyzji inwestycyjnych. Jak pokazują szczegółowe wykresy poszczególnych zmiennych, takie podejście pozwala na precyzyjne modelowanie niepewności w sposób przypominający ludzkie rozumowanie oraz generowanie intuicyjnych i łatwych do interpretacji wyników liczbowych, które uwzględniają niejednoznaczność i subiektywny charakter ocen eksperckich.

## Interpretacja wykresów

Analiza wygenerowanych wykresów pozwala na wyciągnięcie następujących wniosków:

1. **Z indywidualnych wykresów funkcji przynależności** widać, że:

    - Dla zmienności rynku funkcje Gaussa mają optymalne pokrycie przestrzeni decyzyjnej, z największym nakładaniem się w środkowych obszarach wartości (40-60), gdzie niepewność oceny jest największa
    - Dla kondycji finansowej wyraźna jest asymetria funkcji - łatwiej jest jednoznacznie określić "słabą" kondycję finansową niż precyzyjnie rozróżnić między "przeciętną" i "doskonałą"
    - Dla potencjału wzrostu branży funkcje dzwonowe zapewniają ostre rozgraniczenie między kategoriami, co wskazuje na możliwość bardziej zdecydowanej klasyfikacji w tej zmiennej
    - Dla ryzyka inwestycji równomierne rozłożenie pięciu funkcji Gaussa zapewnia płynne przejście między kategoriami ryzyka

2. **Zbiorczy wykres funkcji przynależności** potwierdza dobre dopasowanie funkcji do modelowanych zjawisk, pokazując jak wszystkie zmienne współdziałają w systemie.

3. **Wykresy konturowe** ujawniają, że:

    - Przy spadającej branży (pierwszy wykres) nawet dobra kondycja finansowa nie jest w stanie całkowicie zniwelować ryzyka inwestycyjnego przy wysokiej zmienności rynku
    - W przypadku stabilnej branży (środkowy wykres) istnieje wyraźna "bezpieczna strefa" w lewym górnym rogu (niska zmienność, dobra kondycja)
    - Dla rozwijającej się branży (ostatni wykres) bezpieczna strefa jest znacznie większa, co potwierdza pozytywny wpływ potencjału wzrostu na ocenę ryzyka

4. **Wykres powierzchniowy 3D** pozwala zauważyć:
    - Wyraźne "doliny" niskiego ryzyka w obszarze niskiej zmienności i dobrej kondycji finansowej
    - Strome "zbocza" wskazujące na szybki wzrost ryzyka przy pogorszeniu się któregokolwiek z parametrów
    - Stosunkowo płaskie "płaskowyże" wysokiego ryzyka przy złej kondycji finansowej i wysokiej zmienności rynku

Dodane indywidualne wykresy funkcji przynależności dla poszczególnych zmiennych znacząco ułatwiają zrozumienie i interpretację modelu rozmytego. Pozwalają lepiej prześledzić proces wnioskowania od danych wejściowych, poprzez funkcje przynależności, do końcowej oceny ryzyka inwestycyjnego.
