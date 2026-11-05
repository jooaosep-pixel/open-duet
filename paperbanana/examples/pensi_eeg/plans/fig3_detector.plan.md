## Layout
- **Canvas:** 16:9 aspect ratio, white background.
- **Orientation:** Left-to-right linear pipeline.
- **Stages:** 5 main stages (Input, Feature Extraction, Vectorization, Classification, Output).
- **Color Palette:** Purple (#6A4BB8) for processing modules, Blue (#33639A) for data structures, Orange (#C44A2F) for the model and alarms, Graphite (#1B1F3A) for text and signal lines.

## Components
1. **Input Signal:** A set of 18 parallel wavy lines (Graphite) representing EEG channels. Label: "EEG Bruto (EDF, 256 Hz)".
2. **Windowing:** A vertical purple dashed box enclosing a segment of the signal. Label: "Janela de 2s".
3. **Feature Grid:** A 3D grid or stack of 3 matrices (Blue). Each matrix is 8x18. Label: "Energia (8 faixas x 18 canais)".
4. **Concatenation Block:** A grouping bracket or box (Purple) that merges the 3 matrices into one long bar. Label: "Concatenação (3 épocas)".
5. **Feature Vector:** A single long, thin horizontal rectangle (Blue). Label: "Vetor de Medidas (432 valores)".
6. **Classifier:** A rectangular block (Orange). Label: "Regressão Logística".
7. **Decision Logic:** A purple diamond or rounded box. Label: "Limiar e Persistência".
8. **Alarm Output:** A bold orange bell icon or box. Label: "Alarme".
9. **Explanation Panel:** A small heatmap (Graphite/Orange) showing a grid of weights. Label: "Pesos por Faixa e Canal".

## Connections
1. **Signal -> Windowing:** Solid arrow (Graphite).
2. **Windowing -> Feature Grid:** Solid arrow (Purple), label: "Extração de Energia".
3. **Feature Grid -> Concatenation Block:** Solid arrow (Blue).
4. **Concatenation Block -> Feature Vector:** Solid arrow (Blue).
5. **Feature Vector -> Classifier:** Solid arrow (Graphite).
6. **Classifier -> Decision Logic:** Solid arrow (Orange), label: "Probabilidade".
7. **Decision Logic -> Alarm Output:** Solid arrow (Orange).
8. **Classifier -> Explanation Panel:** Dashed arrow (Graphite), label: "Explicabilidade".

## Annotations
- **Title (Optional):** "Fluxo do Detector de Crises".
- **Legend:** 
    - (a) Extração de características por janela.
    - (b) Classificação linear e decisão temporal.
- **Notes:** Small text at the bottom: "Modelo Personalizado ou Generalista".

## Avoid
- No GPUs or neural network "brain" icons (the text specifies "common computer, no GPU").
- No English labels (all text must be in Portuguese).
- No complex 3D renderings; keep it to clean 2D vector shapes and flat colors.
- No mention of deep learning or backpropagation.