#### Observations / General Notes:

- Logarithm to achieve better normal distribution?

- Many of the codes in status not available
- Date features split into month, weekday, hour?

- Country codes for departure and arrival airports?

- Testflüge? WKL0000, TUN --> TUN, Route planning, repositioning, cargo, maintenance, test
- Verspätung erst ab 30 Minuten?

- DEL = Deleted oder Delayed?
- Target bei DEP?



##### Done:
- Alle Spalten vollständig --> Kein Imputing notwendig ✅
- STA, STD und DATOP in datetime formatieren ✅
- Zusätzliche Features engineeren (Stunde, Wochentag, Monat) ✅
- Geplante Flugdauer engineeren: STA - STD ✅
- Alle Spaltennamen formatiert: Leerzeichen entfernt, Lowercase ✅
- Leerzeichen am Anfang und Ende aus Spalte "fltid" entfernt ✅
- Drop "id" column ✅

##### Open:
- Checken ob domestic oder international Flug (anhand von Ländercode des Flughafens)
- Zusätzliche Features engineeren (Morgen/Abend, Kalenderwochen, Werktag/Wochenende)
- Feature "Delayed" engineeren: e.g. Yes/No, Delay Clusters (e.g. up to 30 mins, 30 to 120 mins, more than 120 mins, etc.)

##### Backlog/To Discuss:


##### Verworfen:
- Aircarft Class extrahieren aus AC --> Nicht super relevant im Moment
- Feiertage, Ferien --> Sehr große lokale Unterschiede
- Wetterdaten --> Eher relevant für Outlier, komplex an Daten zu kommen
- Verspätungen bei mehrfachem Einstz der gleichen Maschine an einem Tag --> Wird über Morgen/Tag/Abend Kategorisierung abgedeckt


#### Baseline Modell
- Mean pro Abflughafen (!!!Mean nur auf Train Daten berechnen!!!)