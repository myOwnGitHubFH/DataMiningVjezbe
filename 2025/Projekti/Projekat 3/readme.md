1. Autor/i projekta
Mladen Lazić
Amer Delić
2. Opis projekta
Sentiment analiza nad Large Movie Review Dataset-om
Potrebno je na osnovu recenzija filmova iz dataseta napraviti model koji će neku novu recenziju moći klasificirati kao recenziju sa pozitivnim ili negativnim sentimentom
3. Opis dataseta
Large Movie Review je dataset koji sadrži 50000 recenzija na filmove uz ocjene sentimenta za (sentiment score)
50000 podataka podijeljeno je jednako po 25000 na trening i testne
Trening i testni su dalje podijeljeni na pozitivne i negativne također jednako
Dataset sadrži i dodatnih 50000 recenzija za nenadrzirano učenje, ali to nije bila tematika vog projekta
Dataset je formatiran kao jedan folder u kojem se nalaze dva podfoldera (trening i test), a u ta dva su još po dva podfoldera (pozitivni i negativni)
Svaka recenzija je u obliku txt file-a pa je bilo potrebno izvršiti ekstrakciju svake i smještanje u pogodne varijable
4. Opis problema
Problematika ovog projekta je sentiment analiza. Tj. kako da na osnovu recenzija na filmove klasificiramo nove recenzije.
Sentiment analiza ima dodira sa natural language processingom jer je potrebno pratiti kontekst samog teksta kako bi došli do željenih rezultata
5. Opis algoritma
Algoritam tj. neurnonska mreža odabrana za rješavanje problema je LSTM
LSTM je vrsta RNN koja se koristi za zadatke sa sekvencijalnim podacima, procesiranje jezika, prepoznavanje govora i predviđanja na osnovu datog vremena. RNN kao što je već navedeno ima jedan hidden state koji se prožima kroz sve slojeve neuralne mreže što izaziva poteškoće za dugoročnije pamćenje. LSTM modeli saniraju ovu manu uz pomoć memory cell (memorijske ćelije). Memorijska ćelija čuva informacije kroz duži vremenski period što nam omogućava bolje predikcije.
Memorijska ćelija se sastoji od tri dijela: input gate, forget gate i output gate. Input gate upravlja informacijama koje se dodaju u ćeliju, forget gate upravlja informacijama koje se uklanjaju iz ćelije, a output gate upravlja informacijama koje se ispisuju iz memorijske ćelije. Ova arhitektura omogućava LSTM-u da čuva i odbacuje određene informacije dok se kreće kroz mrežu što ustvari omogućava dugoročnije pamćenje.

LSTM je odabran jer imamo pristup većem datasetu jer se na takvim najbolje pokazuje.
Nije povoljan za manje datasetove jer nema dovoljno podataka za učenje

6. Metode koristene za poboljasanje algoritma

Za poboljšanje smo prvobitno iskoristili sentiment score koji nam je dat u datasetu
Model nije radio dobro dok nismo imali pripremljene težine pri inicijalnom polazu modela
Nakon što smo odradili postaljanje težina preciznost je postala veća, a overfitting je bio izbjegnut
Iskorištono je također early stopping tj. rano zaustavljanje gdje smo model postavili na 10 epoha i pratili gubitak u svakoj epohi.
Ako bi gubitak počeo rast prekinuli bi trening i uzeli bi model u najboljoj epohi
Podešeni su parametri u samom LSTM sloju gdje smo ubacili dropout koji je dodatna prevencija protiv overfittinga
7. Rezultati i njhova interpretacija
Model je dostigao preciznost od 89% te preciznost na validacijskim podacima od 81% što je zadovoljavajuće
U prethodnim iteracijama preciznost je bila viša, ali na validacijskim podacima bi pala skoro na 60% ostavljajući velik jaz.
Ovo je bio znak overfittinga koji smo rješavali kroz navedene metode.
Još jedna metrika bila je score koji smo dobijali pri ubacivanju eksternih podataka
Što je score veći znači da je sentiment pozitivniji, što je manji znači negativniji
Prethodno smo imali slučajeve da je za Movie is bad davao pozitive oko 60% score
Ali je kasnije riješeno tuningom modela kao što je navedeno prethodno.