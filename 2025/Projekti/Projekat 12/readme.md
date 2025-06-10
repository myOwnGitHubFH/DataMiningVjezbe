
# Predikcija odlaska zaposlenika




## Autori

- Selimanović Harun
- Smajlović Admir


## Opis projekta

Cilj ovog projekta je analizirati podatke o zaposlenicima i razviti model koji može predvidjeti da li će zaposleni napustiti firmu ili ne. Analiza uključuje vizualizaciju podataka, pripremu za modeliranje i primjenu mašinskih algoritama za klasifikaciju.
## Opis dataseta

Korišten je **HR Analytics Dataset**, koji sadrži informacije o zaposlenicima u firmi.
Kolone koje sa nalaze unutar ovog dataseta su:
- **satisfaction_level** - nivo zadovoljstva poslom 
- **last_evaluation** - ocjena učinka pri posljednjoj evaluaciji
- **number_project** - broj projekata na kojima je zaposleni radio
- **average_montly_hours** - prosječan broj mjesečnih radnih sati
- **time_spend_company** - broj godina u firmi
- **Work_accident** - da li je doživio povredu na radu (0 ili 1)
- **promotion_last_5years** - da li je unaprijeđen u posljednjih 5 godina
- **Department** - odjel u kojem radi
- **salary** - nivo plate (low, medium, high)
- **left** - ciljana varijabla – da li je zaposleni napustio firmu (1) ili ne (0)

## Opis problema

Problem koji rješavamo je binarna klasifikacija: na osnovu dostupnih podataka, predviđamo da li će zaposleni napustiti firmu ili ostati.Ovo je značajno za HR timove, jer omogućava proaktivno djelovanje u cilju zadržavanja ključnog osoblja.
## Opis algoritama

Za rješavanje problema korišten je Random Forest Classifier, popularan algoritam mašinskog učenja zasnovan na stabalima odlučivanja.
Ovaj algoritam je izabran jer:
- Se dobro nosi sa nelinearnim relacijama
- Otporniji je na overfitting u odnosu na obična stabla
- Može raditi i sa numeričkim i sa kategorizovanim podacima

Također su testirani i drugi algoritmi poput Logističke regresije i KNN ali je Random Forest dao najbolje rezultate.

## Metode korištene za poboljšanje algoritma

Da bi se poboljšala tačnost i robusnost modela, korištene su sljedeće metode:
- Feature engineering: transformacija varijabli (salary, Department) u numerički format
- Balansiranje klase: korišten je SMOTE (Synthetic Minority Over-sampling Technique)
- Tuning hiperparametara: korišten je GridSearchCV za odabir najboljih parametara za Random Forest (npr. broj stabala, dubina stabla)
- K-Fold Cross Validation: za pouzdaniju procjenu performansi modela

## Rezultati i njihova interpretacija

Nakon treniranja modela i evaluacije na test setu, postignuti su sljedeći rezultati:

**Random Forest**
- Accuracy: 0.9884

**Logistic Regression**
- Accuracy: 0.7604

**Support Vector Machine**
- Accuracy: 0.7847

**Decision Tree**
- Accuracy: 0.9747

Kao što se vidi Random Forest ima najveću preciznost dok najmanju preciznost ima Logistic Regression.

Vizualizacije koje su korištene uključuju:
- Confusion matrix – za tačne i pogrešne klasifikacije
- ROC kriva – za prikaz odnosa između true positive i false positive rate
**Feature importance – najuticajnije varijable su:**
- satisfaction_level
- number_project
- time_spend_company


