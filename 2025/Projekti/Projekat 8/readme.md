**UNIVERZITET U ZENICI**
**POLITEHNIČKI FAKULTET**

**Projekat: Predikcija broja saobraćajnih nesreća u SAD-u**
**Predmet: Rudarenje podataka**
**Profesor: doc. dr. Adnan Dželihodžić**
**Asistenti: mr. Faris Hambo, mr. Ahmed Mujić**
**Student: Ajla Huskić, II-133**
**Studij: Softversko inženjerstvo, II ciklus studija**
**Zenica, juni 2025.**

---

## 2. Opis projekta

Cilj ovog projekta je razvoj modela za predikciju broja saobraćajnih nesreća u određenim vremenskim i geografskim uslovima. Projekt je realizovan u okviru kursa **Rudarenje podataka**, s fokusom na obradu i analizu realnog skupa podataka iz SAD-a, te izgradnju prediktivnog modela koji koristi kombinaciju faktora kao što su vremenski uslovi, doba dana, mjesec i lokacija.
Model koristi **Random Forest regresiju**, koja omogućava obradu nelinearnih odnosa između ulaznih varijabli i ciljne vrijednosti (broj nesreća). Osim osnovnog modeliranja, istražene su različite metode za čišćenje podataka, transformaciju i evaluaciju performansi modela kako bi se poboljšala tačnost i robusnost predikcija.

## 3. Opis dataseta

Dataset korišten u projektu dolazi sa platforme **Kaggle** i nosi naziv [US Accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents). Ovaj skup sadrži informacije o preko 7 miliona saobraćajnih nesreća zabilježenih u SAD-u od 2016. do 2023. godine.

Glavne kolone koje su iskorištene:

* `Start_Time`, `Start_Lat`, `Start_Lng`: vrijeme i lokacija nesreće
* `Weather_Condition`: opis vremenskih uslova
* `Humidity(%)`, `Visibility(mi)`, `Wind_Speed(mph)`: meteorološki faktori
* `Distance(mi)`: dužina nesreće u miljama

Dataset je detaljan, ali sadrži i znatan broj `NaN` vrijednosti, outliera, te redundantnih kolona koje su filtrirane tokom pripreme. Konačna forma dataseta (`nesrece_model_podaci.csv`) sadrži agregirane podatke (grupisane po vremenu i prostoru) koji predstavljaju broj nesreća u definisanim uslovima.

## 4. Opis problema

Problem koji se rješava je **predikcija broja saobraćajnih nesreća** u zadatoj geografskoj zoni (grupisanoj po `Lat_Group` i `Lng_Group`), u određenom satu (`Hour`), mjesecu (`Month`) i vremenskim uslovima (`Weather_Condition`).
Radi se o regresijskom problemu jer je ciljna varijabla `accident_count` kontinuirana i može poprimiti širok raspon vrijednosti.
Izazovi uključuju:

* **neravnomjernu distribuciju podataka** (većina vremensko-lokacijskih kombinacija ima 0 ili mali broj nesreća),
* **veliki broj kategorija vremenskih uslova**,
* prisustvo **outliera** koji značajno utiču na evaluacione metrike.

## 5. Opis algoritma

Za predikciju je korišten algoritam **Random Forest Regressor**, koji koristi ansambl stabala odlučivanja za regresiju. Ovaj model je robustan na šum u podacima, može raditi s nelinearnim odnosima i ne zahtijeva skaliranje ulaznih varijabli.

Pipeline modela:

1. **One-hot encoding** nad `Weather_Condition`
2. **Numeričke varijable**: `Lat_Group`, `Lng_Group`, `Hour`, `Month`
3. **Random Forest** sa `n_estimators=100`, `max_depth=15`

Model je treniran na 80% podataka i testiran na 20% neviđenih podataka.

## 6. Metode korištene za poboljšanje algoritma

### 1. ČIŠĆENJE OUTLIERA

* Testirani su različiti pragovi: IQR metoda (stroga), percentil metoda (blaža)
* Blaža metoda (0.1% – 99.9%) sačuvala je više podataka i dala viši R²

### 2. LOG-TRANSFORMACIJA CILJNE VARIJABLE

* `log1p()` primijenjen na `accident_count`, a zatim `expm1()` nakon predikcije
* Stabilizuje velike vrijednosti, smanjuje uticaj outliera

### 3. DODAVANJE VREMENSKIH KOMPONENTI

* `Hour` i `Month` uvedeni kao numeričke varijable

### 4. VIZUALNA ANALIZA

* Scatter plot i histogram reziduala korišteni za evaluaciju modela i otkrivanje pristrasnosti

## 7. Rezultati i interpretacija

Model je evaluiran pomoću dvije metrike:

* **RMSE** (Root Mean Squared Error)
* **R² Score** (koeficijent determinacije)

| Verzija modela                      | RMSE  | R² Score | Napomena                                 |
| ----------------------------------- | ----- | -------- | ---------------------------------------- |
| Strogo čišćenje outliera + bez loga | 6.38  | 0.83     | manja greška, ali slabija generalizacija |
| Blaže čišćenje + log-transformacija | 14.08 | **0.91** | bolji fit na generalni uzorak, veći R²   |

📈 Scatter plot pokazuje da većina predikcija prati pravac `y = x` (idealno), što ukazuje na dosljedne rezultate.
📉 Histogram reziduala pokazuje simetričnu distribuciju grešaka, bez izražene pristrasnosti modela.

---

**README dokument izrađen u okviru projekta iz Rudarenja podataka - 2025**
