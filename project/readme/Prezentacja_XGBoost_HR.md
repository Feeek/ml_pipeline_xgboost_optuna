# Predykcja wynagrodzen 'Data Jobs' (XGBoost) – HR Usecase

## 1. Cel
- Predykcja wynagrodzen w USD dla roli data/IT
- Pipeline: ETL + EDA + przygotowanie danych pod XGBoost
- Wartosc dla HR: szybsze i spojne widełki ofert, wsparcie negocjacji

**Opis punktu:** Jednym zdaniem: budujemy wiarygodny pipeline danych i analize, na ktorej oprzemy model prognozujacy pensje.

## 2. Problem biznesowy (HR)
- Rozrzut stawek i dlugi time-to-offer
- Ryzyko przepłacenia lub niedoszacowania
- Potrzeba: rekomendacja widelek dla profilu kandydata

**Opis punktu:** Celem jest ustandaryzowanie decyzji i skrocenie procesu ofertowania.

## 3. Zrodla danych (Kaggle)
- ruchi798/data-science-job-salaries – ds_salaries.csv
- sazidthe1/data-science-salaries – data_science_salaries.csv
- arnabchaki/data-science-salaries-2025 – salaries.csv
- Ladowanie przez kagglehub (cache w .kaggle-cache)

**Opis punktu:** Trzy spojne zrodla podnosza pokrycie rynku i aktualnosc.

## 4. ETL – Extract (etl.py)
- extract(url, file): pobranie pliku z Kaggle
- Autodetekcja i usuniecie sztucznej kolumny indeksu
- Kazdy DataFrame trafia do self.datasets

**Opis punktu:** Usuwamy smieciowe indeksy powstale po eksportach.

## 5. ETL – Transform (etl.py)
- Mapowania: mappings/columns.json, mappings/values.json
- Standaryzacja wartosci (skroty -> etykiety)
- Normalizacja krajow przez country_converter (cache)
- Zmiana nazw kolumn na standard

**Opis punktu:** Celem jest ujednolicenie struktur i etykiet.

## 6. ETL – Load (etl.py)
- Konkatenacja wszystkich zestawow
- drop_duplicates, reset_index
- Sortowanie po work_year rosnaco

**Opis punktu:** Finalnie jedna czysta tabela gotowa do analizy.

## 7. EDA – klasa EDA (eda.py)
- describe(): braki, wymiar, statystyki
- correlations(): Pearson/Spearman/Kendall + kategorie (correlation ratio)
- outliers(): boxploty numerycznych, top-N kategorii

**Opis punktu:** Jedna klasa, komplet podstawowych raportow EDA.

## 8. EDA – korelacje z celem
- Numeryczne: corrwith(target) (Pearson/Spearman/Kendall)
- Kategoryczne: correlation ratio (eta) przez factorize + wariancje

**Opis punktu:** Pozwala porownywac w jednym szkielecie liczby i kategorie.

## 9. Orkiestracja (main.py)
- ETLPipeline(): extract x3 -> transform -> load
- EDA(dataset): describe, correlations(salary_in_usd), outliers

**Opis punktu:** Jeden skrypt uruchamia caly proces – idealny pod demo.

## 10. Analiza w czasie (dump.py)
- Zaladowanie datasetu z ETL
- Usuniecie kolumn ryzykownych do przecieku (np. salary)
- Label-encoding kategorii
- Heatmapa korelacji + barplot korelacji z celem per rok

**Opis punktu:** Rentgen korelacji i trendow rok po roku.

## 11. Model i strojenie – gdzie wpinamy (High level)
- Po ETL/EDA mamy jednolite kolumny i zrozumiale cechy
- Split train/valid/test, obrobka outlierow
- XGBoost + Optuna (RMSE w USD)

**Opis punktu:** To przygotowanie pod produkcyjny model predykcyjny.

## 12. Uzycie w HR
- Sugerowanie widelek oferty dla profilu
- What-if: remote vs on-site, region, rozmiar firmy
- Audyt czynnikow (feature importance, SHAP)

**Opis punktu:** Wartosc: szybciej, spojniej i z uzasadnieniem decyzyjnym.

## 13. KPI i ryzyka
- KPI: ↓ time-to-offer, ↑ spojnosci widelek, oszczednosci
- Ryzyka: bias, drift rynkowy, RODO
- Mitigacje: audyt cech, monitoring RMSE, retrainy okresowe

**Opis punktu:** Odpowiedzialne wykorzystanie i nadzor jakosci.

## 14. Rozwuj
- Featury: is_remote, same_country, grupy rol
- Log-cel + KFold + regularyzacja
- Dashboard (Power BI/Streamlit), integracja z ATS/HRIS

**Opis punktu:** Plan rozwoju i produktowe wdrozenie dla HR.

