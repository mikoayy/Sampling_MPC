# Sampling MPC

## Środowisko

Python 3.10+ (testowane na 3.14).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Skrypty uruchamia się z katalogu głównego repozytorium.

## Zadania

| Zadanie | Uruchomienie |
|---|---|
| 1–2. Ustawienie stanu początkowego [3.14, 0] | `python zad2_stan_poczatkowy.py` |
| 3–4. Sterownik PD i swing-up przez sterowanie energią + PD | `python zad3_4_pd_energia.py` |
| 5. Random shooting | `python zad5_random_shooting.py` |
| 6. Heatmapa nagrody w funkcji liczby i długości rolloutów | `python zad6_heatmapa.py` → `results/heatmap_rs.png` |
| 7. MPPI | `python zad7_mppi.py` |
| 8. Odporność MPPI na błąd modelu | `python zad8_odpornosc.py` → `results/odpornosc.png` |
| 9. MPPI na Hopperze | `python zad9_hopper.py` (wariant tylko z balansowaniem: `--balans`) |

Sterowniki są w `controllers/`; model używany przez random shooting i MPPI jest w `controllers/model.py`.
