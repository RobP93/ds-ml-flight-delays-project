## Observations:

- Logarithm to achieve better normal distribution?
- Many of the codes in status not available
- Date features split into month, weekday, hour?
- Country codes for departure and arrival airports?
- Testflüge? WKL0000, TUN --> TUN, Route planning, repositioning, cargo, maintenance, test
- Verspätung erst ab 30 Minuten?
- DEL = Deleted oder Delayed?
- Target bei DEP?

fltid Bedeutung:
1-99: Premium oder Langstreckenflüge
0100-0999: Interkontinental oder wichtige Flüge
1000-1299: Kurzstrecke u. Inland
3000-5999: Reginalflüge oder Tochtergesellschaften
6000+: Codeshare (durch andere Airline ausgeführt)
8000: Charter
9000: Positionierungsflüge


Done:
- Alle Spalten vollständig --> Kein Imputing notwendig
- STA, STD und DATOP in datetime formatieren (Done)
- Zusätzliche Features engineeren (Stunde, Wochentag, Monat) (Done)
- Geplante Flugdauer engineeren: STA - STD (Done)
- Alle Spaltennamen formatiert: Leerzeichen entfernt, Lowercase (Done)
- Leerzeichen am Anfang und Ende aus Spalte "fltid" entfernt (Done)

Open:
- Checken ob domestic oder international Flug (anhand von Ländercode des Flughafens)
- Drop "id" column


Backlog/To Discuss:
- Zusätzliche Features engineeren (Morgen/Abend, Feiertag)

- Wetterdaten
- Verspätungen bei mehrfachem Einstz der gleichen Maschine an einem Tag
- Feature "Delayed" engineeren: Nur ATA und DEP behalten, Rest entfernen, da keine relevanten Infos


Verworfen:
- Aircarft Class extrahieren aus AC


## Baseline Modell
- Mean pro Abflughafen (!!!Mean nur auf Train Daten berechnen!!!)
