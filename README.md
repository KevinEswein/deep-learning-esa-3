# ESA 3: Language Model mit LSTM

- **Thema:**
Trainieren eines Language Models (LM) zur Wortvorhersage mit einem rekurrenten Long Short-Term Memory (LSTM) Network
und dem TensorFlow.js (TFJS) Framework/API.
- **Bearbeitungszeit:** 30–35 Stunden, je nach Vorkenntnissen und Erfahrung.
- **Voraussetzungen:** Kapitel 1–16 bis einschließlich ARC - Architectures.
- **Kompetenzerwerb/Lernziele:** Nach der Bearbeitung der Aufgabe sollten Sie:
  - Verstehen, wie man Sprache mit einem rekurrenten Netzwerk wie Long Short-Term Memory (LSTM) und allgemeiner in einem
    KI-System verarbeitet.
  - Wissen, was ein Language-Modell (LM) ist, und wie man ein solches trainiert.
  - Ein Long Short-Term Memory (LSTM) Netzwerk aufsetzen und trainieren können.
- **Vorbereitung:**
Recherchieren Sie zu Language-Models und Wortvorhersage (Next Word Prediction), siehe Hintergrundwissen.

## Aufgabenstellung
Erstellen Sie ein Language Models (LM) zur Wortvorhersage. Trainieren Sie dazu als Modell ein Long Short-Term Memory
(LSTM) Netzwerk auf der Basis der Daten (siehe den Punkt „Daten“ unten) zur Wortvorhersage (Next Word Prediction).
Mittels des trainierten LSTM Language-Models kann autoregressiv ein Text generiert werden, in dem das jeweils
vorhergesagte Wort an den vorhandenen Text angehängt wird.

## Modell und Optimierung
Nutzen Sie für Ihr Modell die folgende Netzwerkarchitektur und Parametern für den Lernalgorithmus:

- Stacked LSTM: 2 hidden Layer (in sich rekursiv) mit je 100 LSTM Units (Sie können auch mit anderen/größeren Architekturen experimentieren).
- Softmax Output mit der Dimension des Dictionaries.
- Als Loss nutzen Sie Cross-Entropy.
- Lernrate und Optimizer: Adam mit Lernrate(Learning Rate)=0.01 und Batch-Size=32 (Sie können auch mit anderen experimentieren).
- Anzahl der Trainings-Epochen (Epochs): Ausprobieren, dazu den Loss beobachten (Sie können dazu den Tensorflow (TF) Visor nutzen).

## Interaktion
I1) Der Nutzer kann einen Text (Prompt) eingeben. Dieser sollte nur aus vollständigen, durch Leerzeichen getrennten
Wörtern (Tokens) bestehen. Er kann dann jederzeit (am Ende eines vollständig eingegebenen Wortes) den Button
„Vorhersage“ betätigen und erhält eine Darstellung der wahrscheinlichsten nun folgenden Wörter mit deren
Wahrscheinlichkeiten. Er kann eines dieser Wörter auswählen, sodass es an den Text angehängt wird. Daraufhin beginnt
automatisch eine neue Wortvorhersage.  

I2) Der Nutzer kann mittels des „Weiter“ Buttons das wahrscheinlichste vorhergesagte Wort annehmen. Diese wird an den
bisher eingegebenen Text angehängt, darauf beginnt automatisch eine neue Wortvorhersage. Der Nutzer kann also über
wiederholtes Betätigen des „Weiter“ Buttons einen Text generieren lassen.

I3) Der Nutzer kann über einen „Auto“ Button automatisch bis zu 10 Wörter vorhersagen lassen. Diese automatische
Vorhersage kann mittels eines „Stopp“ Buttons unterbrochen werden.

I4) Über ein „Reset“ Button werden der eingegebene Text und das Netzwerk zurückgesetzt.

**Buttons:** I1) Vorhersage, I2) Weiter, I3) Auto, Stopp und I1) die Auswahl eines der nächsten Wörter.

## Experimente + Fragestellungen
1) Experimentieren Sie mit der Netzwerkarchitektur. Dokumentieren und begründen Sie Ihre finale Architektur.


2) Notieren Sie als Resultat, wie oft die Vorhersage genau richtig ist (k=1), und wie oft das korrekte nächste Wort
unter den ersten k Worten, die sie vorhersagen liegt, mit k gleich 5, 10, 20 und 100. Sie können auch die Perplexity
(siehe Hintergrund) als Maß Ihrer Resultate nutzen.


3) Können Sie Ihre ursprünglichen Trainingsdaten mittels des trainierten Models rekonstruieren? (überlegen Sie, ob sich
daraus ein Datenschutzproblem ergibt).

## Visualisierung
Sie können, dazu außer der API von TF, z. B. folgende Bibliotheken zur Visualisierung der Ergebnisse als Diagramm
nutzen: Plotly, D3.

## Diskussion
Diskutieren Sie Ihre Ergebnisse (unter den Resultaten auf der gleichen HTML-Seite, max. 10 Sätze). Was haben Sie
beobachtet/gelernt?

## Dokumentation
Nutzen Sie die gleiche HTML-Seite (unter der Diskussion) wie zur Abgabe Ihrer Lösung zur Dokumentation der folgenden
Aspekte:

1) Technisch: Listen Sie alle verwendeten Frameworks auf und erklären Sie kurz (1–3 Sätze) wozu Sie diese verwenden.
Dokumentieren Sie technische Besonderheiten Ihrer Lösung.


2) Fachlich: Erläutern Sie Ihre Implementierung der Logik und alles, was für ihre Lösung wichtig ist (Ansatz, Resultate,
Quellen, etc.)

Schreiben Sie bitte nichts in die Moodle Abgabe-Felder.

**Hinweise**
Wortvorhersage ist eine Multi-Class Classification. Nutzen Sie als Objektivfunktion den Categorical Cross-Entropy Loss.

**Fehlerbehandlung, Test und QA**
Stellen Sie sicher, dass die Eingabe das richtige Format hat.

**User Experience (UX):**
Beachten Sie die Human/Mensch-Computer-Interaction (HCI) Kriterien beim Interaktionsdesign: ISO 9241-11 Anforderungen
an die Gebrauchstauglichkeit und ISO 9241-110 Interaktionsprinzipien. Ihre Anwendung sollte funktional
(Aufgabenangemessenheit) und benutzerfreundlich (Usability) und mit angemessenem Feedback und einer [kontextsensitive]
Hilfe ausgestattet sein.

**Gestaltung:** Achten Sie auf eine sinnvolle Semantik bei der Farbgestaltung und ein übersichtliches Layout. Siehe
dazu: material.io - Design Guidance and Code

**Libraries:** Sie können, dazu außer der API von TF (Visor), z. B. folgende Bibliotheken nutzen: Plotly, D3, Chart.js,
etc.

**Arbeitsumgebung:** JS-, HTML-IDE (z. B. Atom, WebStorm, Visual Studio Code), [local] Web-Server.  
Zum Lernen auf einem Server, z. B.: Google colab

**Testumgebung:** Chrome [unter macOS].

## Bewertungskriterien + Punkte
1. Funktionsfähigkeit und Vollständigkeit der Anwendung entsprechend der Aufgabenstellung (15 Punkte)
2. Modellperformance (5 Punkte)
3. Experimente, Resultate und Diskussion (10 Punkte)
4. Dokumentation, technisch und fachlich (5 Punkte)
5. User Experience (UX) und User Interaktion (HCI, Interaktionsdesign, Dialoggestaltung, Usability, Hilfe) (5 Punkte)
6. Gestaltung und Visualisierung (Farben, Formen, Screen-Layout, Text, Semantik) (5 Punkte)

**Gesamtpunktzahl:** 45 Punkte

