# Line Follower mit Reinforcement Learning

Praktikumsprojekt **Künstliche Intelligenz** — Hochschule Bochum

**Teammitglieder:** Majd Fares (018383717), Mohamad Aldabaa (018383957)

---

## Kurzbeschreibung

Ein Agent (2D-Auto) lernt mit Reinforcement Learning, einer Linie zu folgen —
ausschließlich anhand von **5 binären Bodensensoren**, ohne Karte, Position
oder einprogrammierte Regeln. Ziel ist das Absolvieren von **2 vollständigen
Runden** auf einer gewellten Rundstrecke.

Trainiert wurden zwei Algorithmen (**PPO** und **DQN**, jeweils
Stable-Baselines3) und mit zwei Baselines verglichen (Random Policy,
regelbasierte Policy). Beide Agenten erreichen eine Erfolgsrate von **100 %**
auf der Trainingsstrecke — auch von zufälligen Startpositionen aus — und
generalisieren auf Streckenformen, die im Training nie vorkamen.

Ein dokumentierter Fall von **Reward Hacking** (Schlangenlinien-Fahren unter
der ursprünglichen Reward-Funktion v1) und dessen Behebung durch eine
fortschrittsbasierte Reward-Funktion (v2) sind Teil der Auswertung —
Details in der Dokumentation (`docs/dokumentation.pdf`).

## Verwendetes Framework

| Aspekt | Beschreibung |
|---|---|
| Framework | Python + Gymnasium (eigene Umgebung) |
| Simulation / Rendering | Pygame (vereinfachtes kinematisches Modell) |
| RL-Bibliothek | Stable-Baselines3 (PPO, DQN) |
| Sprache | Python 3 |

## Installation

```bash
# Projektordner entpacken, dann im Ordner:
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## Verwendung

```bash
# Training (Algorithmus in train.py oben wählen: ALGO = "PPO" oder "DQN")
python train.py

# Evaluation: Baselines vs. trainierte Agenten + Generalisierungstest
python evaluate.py

# Demo: trainierten Agenten fahren sehen
python watch.py          # PPO
python watch.py dqn      # DQN

# Umgebung manuell testen (Pfeiltasten links/rechts, R = Reset)
python manual_drive.py
```

Fertig trainierte Modelle liegen in `models/`, Trainingsplots und
Episoden-Logs in `results/`. Training und Evaluation sind über feste
Seeds reproduzierbar.

## Projektstruktur

```
LineFollower/
├── README.md
├── requirements.txt
├── track.py                 # Streckengeometrie + Roboter-Kinematik + Sensorik
├── line_follower_env.py     # Gymnasium-Umgebung (Reward, Episodenlogik)
├── train.py                 # Training (PPO/DQN) + Trainingsplot
├── evaluate.py              # Baseline-Vergleich + Generalisierungstest
├── watch.py                 # Demo des trainierten Agenten
├── manual_drive.py          # manueller Test der Umgebung
├── models/                  # trainierte Modelle (.zip)
│   └── ppo_v1_presence.zip  # v1-Modell (Reward-Hacking-Nachweis)
├── results/                 # Trainingsplots + Episoden-Logs (v1 und v2)
└── docs/
    └── dokumentation.pdf
```

## Bekannte Einschränkungen

- Vereinfachtes kinematisches Modell: konstante Geschwindigkeit, keine
  Trägheit, drei diskrete Lenkaktionen.
- Die Beobachtung (5 binäre Sensoren, 32 mögliche Zustände) ist bewusst
  minimal; auf sehr scharfen Kurven kann die Linie zwischen zwei
  Zeitschritten aus dem Sensorbereich rutschen.
- Die Rundenzählung erfolgt über den Winkelfortschritt um das
  Streckenzentrum und setzt daher ringförmige Strecken voraus.
- DQN zeigte auf unbekannten, kurvenreicheren Strecken eine geringere
  Robustheit (88 % Erfolg) als PPO (100 %) — siehe Diskussion in der
  Dokumentation.
