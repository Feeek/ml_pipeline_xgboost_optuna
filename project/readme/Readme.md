# Salary Prediction Project – HR Analytics with XGBoost & SHAP

## Cel projektu
Celem analizy było stworzenie pełnego procesu (pipeline), który:
- przetwarza dane o wynagrodzeniach w branży IT/Data,
- identyfikuje czynniki najsilniej wpływające na pensje,
- przewiduje wysokość wynagrodzeń z wykorzystaniem modelu XGBoost,
- wyjaśnia decyzje modelu przy użyciu metod interpretowalności (SHAP).

Projekt ma charakter zarówno analityczny (EDA), jak i predykcyjny (ML), a wnioski są formułowane w języku zrozumiałym także dla HR i biznesu.

---

## Pipeline
1. **ETL** – czyszczenie i standaryzacja danych, usuwanie duplikatów.  
2. **EDA** – analiza rozkładów, korelacji, outlierów, różnic regionalnych.  
3. **Feature Engineering** – klastrowanie ról do obszarów kompetencji (`p_ai`, `p_data`, `p_software`, `p_academic`).  
4. **Segmentacja** – podział zbioru na dwa clustery: **US+CA+UK** vs **Rest**.  
5. **Modelowanie** – XGBoost + wstępne strojenie hiperparametrów (Optuna).  
6. **Interpretacja** – analiza SHAP (ważność cech globalnie i lokalnie).  
7. **Wnioski** – kluczowe czynniki wynagrodzeń, ocena jakości modelu, kierunki rozwoju.

---

## Dane
- Rozmiar: **52,938 wierszy × 11 kolumn**  
- Braki danych: **0**  
- Dominujące cechy próby:
  - lokalizacja: **USA** (ponad 40 tys. rekordów),
  - stanowiska: Data Scientist / Data Engineer / Data Analyst,
  - forma zatrudnienia: głównie **Full-time**,
  - firmy: najczęściej **Medium**,
  - waluta: **USD**.

---

## Wyniki modeli
| Segment        | RMSE   | MAE   | R²     | Interpretacja |
|----------------|--------|-------|--------|---------------|
| **US+CA+UK**   | 51,700 | 42,503| 0.231  | Model uchwycił część wzorców, ale sporo zmienności pozostaje niewyjaśnione. |
| **Rest**       | 40,890 | 31,401| 0.006  | Model praktycznie nie wyjaśnia płac – dane są zbyt różnorodne i mało liczne. |

**Wnioski:**  
- Najlepsze wyniki uzyskano dla regionu **US+CA+UK**, gdzie dane są liczne i spójne.  
- Segment **Rest** wymaga dodatkowych informacji (np. koszty życia, wskaźniki gospodarcze).  

---

## Kluczowe obserwacje z EDA
- **Stanowisko (`job_title`)** i **doświadczenie (`experience_level`)** – główne determinanty płac.  
- **Geografia** – pensje w USA i Kanadzie znacznie przewyższają Europę kontynentalną; w grupie Rest wyróżnia się Australia.  
- **Rozmiar firmy** – duże organizacje częściej oferują wyższe i bardziej zróżnicowane płace.  
- **Tryb pracy** – (On-site/Remote/Hybrid) nie różnicuje istotnie wynagrodzeń w porównaniu z lokalizacją i rolą.  

---

## Kierunki rozwoju
- **Strojenie hiperparametrów** – pełna optymalizacja Optuna (200–500 prób).  
- **Lepsza transformacja celu** – logarytmizacja wynagrodzeń, skalowanie per-region.  
- **Nowe cechy** – koszty życia, makroekonomia, dokładniejsze klasyfikacje stanowisk.  
- **Walidacja czasowa** – testowanie generalizacji na ostatnich latach, monitoring dryfu danych.  

---

## Struktura repozytorium
- `etl.py` – przygotowanie danych, czyszczenie, standaryzacja.  
- `eda.py`, `region_eda.py` – analiza eksploracyjna i wizualizacje.  
- `feature_eng.py` – inżynieria cech, klastrowanie ról.  
- `xgb.py` – modelowanie XGBoost + Optuna.  
- `main_pres.ipynb` – finalny notatnik prezentacyjny (opis + wyniki).  

---

## Podsumowanie
Projekt pokazuje, jak zbudować kompletny proces: **od danych surowych, przez analizę i inżynierię cech, po modelowanie i interpretację wyników**.  
Wyniki wskazują, że prognozowanie płac jest możliwe z umiarkowaną skutecznością w regionach z dużą liczbą obserwacji (USA, Kanada, UK), natomiast w innych wymaga dalszego wzbogacania danych.  
Dzięki analizie SHAP wnioski są **transparentne i zrozumiałe także dla HR i biznesu**.
