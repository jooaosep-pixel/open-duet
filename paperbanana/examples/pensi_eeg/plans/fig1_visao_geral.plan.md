## Layout
- **Canvas:** 16:9 aspect ratio, horizontal orientation.
- **Panels:** Three main stages arranged from left to right, separated by subtle background shading or thin dividers.
- **Reading Order:** Left-to-right flow, with a feedback loop returning from the final stage to the first.
- **Palette:** Purple (#6A4BB8) for data/database, Blue (#33639A) for modeling, Orange (#C44A2F) for metrics, Graphite (#1B1F3A) for text/lines.

## Components
1. **Stage 1: Construção do Banco (Purple)**
   - **Input Icons:** A stack of documents labeled "Sinais EDF (256 Hz)" and a user icon labeled "Dados do Paciente".
   - **Process Box:** A central box labeled "Fluxo de Anotação" containing three sub-nodes: "Leitor 1", "Leitor 2 (Cego)", and "Leitor Sênior" (connected by small arrows).
   - **Output Shape:** A cylinder icon labeled "Banco de Dados Congelado" with a small padlock/hash symbol.

2. **Stage 2: Modelagem (Blue)**
   - **Feature Box:** A small rectangle labeled "Energia (8 faixas x 18 canais)".
   - **Split Path (Top):** A box labeled "Detector Personalizado" with sub-text "1 modelo por criança" and "Validação Temporal".
   - **Split Path (Bottom):** A box labeled "Detector Generalista" with sub-text "Modelo Global" and "Leave-one-child-out".

3. **Stage 3: Avaliação (Orange)**
   - **Metrics List:** A single grouped box containing the labels: "F1 por evento", "Alarmes Falsos / 24h", and "Latência (SzCORE)".
   - **Error Node:** A small red-tinted circle labeled "Análise de Erros".

## Connections
1. **Data flow:** "Sinais EDF" and "Dados do Paciente" -> "Fluxo de Anotação" (Solid, Graphite).
2. **Storage:** "Fluxo de Anotação" -> "Banco de Dados Congelado" (Solid, Graphite).
3. **Training:** "Banco de Dados Congelado" -> "Energia (8 faixas x 18 canais)" (Solid, Graphite).
4. **Branching:** "Energia..." splits into two arrows -> "Detector Personalizado" and "Detector Generalista" (Solid, Blue).
5. **Inference:** Both detectors -> "Metrics List" (Solid, Orange).
6. **Feedback Loop:** "Análise de Erros" -> (Long dashed line along the bottom) -> "Fluxo de Anotação" (Dashed, Graphite, labeled "Refinamento do Manual").

## Annotations
- **Panel Titles:** (a) Coleta e Anotação, (b) Treinamento de Modelos, (c) Avaliação de Desempenho.
- **Legend:** 
  - Solid line: Fluxo de dados.
  - Dashed line: Feedback de melhoria.
  - Purple: Dados.
  - Blue: Aprendizado de Máquina.
  - Orange: Métricas.

## Avoid
- No photos of children or hospital rooms.
- No complex mathematical formulas for Logistic Regression.
- No mention of specific software names not in the text.
- No English labels (all must be in Portuguese).
- No overlapping arrows that obscure text.