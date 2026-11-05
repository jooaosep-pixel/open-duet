## Layout
The diagram follows a horizontal, left-to-right pipeline across a 21:9 canvas. It is organized into four main functional zones: **Entrada** (Input), **Anotação Primária** (Primary Annotation), **Controle de Qualidade** (Quality Control), and **Validação e Escala** (Validation and Scaling). 

## Components
1.  **Entrada (Left):** Two stacked icons/boxes. Top: "Laudo Médico" (Purple #6A4BB8). Bottom: "Exame EDF" (Graphite #1B1F3A).
2.  **Anotação Primária (Center-Left):** A large box labeled "Leitor Qualificado". Inside, a small sub-label "Manual (SCORE/ACNS 2021)" (Graphite #1B1F3A).
3.  **Amostragem (Center):** A small diamond or split-point labeled "Sorteio (15-33%)".
4.  **Dupla Leitura (Center-Right):** A box labeled "2º Leitor (Cego)" (Blue #33639A).
5.  **Adjudicação (Top-Right):** A box labeled "Leitor Sênior" (Orange #C44A2F) positioned above the main flow to handle conflicts.
6.  **Consistência (Right):** A box labeled "Verificador Automático" (Graphite #1B1F3A).
7.  **Piloto (Far-Right):** A box labeled "Fase 0 (~50 exames)" (Purple #6A4BB8) representing the final gate before scaling.
8.  **Resultado (Far-Right Edge):** A cylinder icon labeled "Banco de Dados Final".

## Connections
1.  **Laudo Médico & Exame EDF -> Leitor Qualificado:** Solid arrows (Purple) indicating the report guides the reading.
2.  **Leitor Qualificado -> Amostragem:** Solid arrow (Purple) labeled "Marcações de Crise".
3.  **Amostragem -> 2º Leitor (Cego):** Solid arrow (Blue) labeled "Fração Sorteada".
4.  **Amostragem -> Verificador Automático:** Solid arrow (Purple) bypassing the double reading for the remaining fraction (~70%).
5.  **2º Leitor (Cego) & Leitor Qualificado -> Comparison Point:** Dashed lines meeting at a junction.
6.  **Comparison Point -> Leitor Sênior:** Solid arrow (Orange) labeled "Discordância".
7.  **Comparison Point -> Verificador Automático:** Solid arrow (Purple) labeled "Concordância (Padrão Ouro)".
8.  **Leitor Sênior -> Verificador Automático:** Solid arrow (Orange) labeled "Adjudicação".
9.  **Verificador Automático -> Fase 0:** Solid arrow (Graphite) labeled "Check de Consistência".
10. **Fase 0 -> Banco de Dados Final:** Thick solid arrow labeled "Escalonamento".

## Annotations
- **(a) Preparação:** Above the Input/Primary Annotation zone.
- **(b) Consenso:** Above the Quality Control zone.
- **(c) Produção:** Above the Validation/Scaling zone.
- **Legend:** 
    - Purple (#6A4BB8): Fluxo Principal.
    - Blue (#33639A): Dupla Leitura.
    - Orange (#C44A2F): Adjudicação.
    - Graphite (#1B1F3A): Regras Técnicas.

## Avoid
- No realistic human figures or EEG waveforms.
- No English labels (all must be in Portuguese as requested).
- Do not include the specific kappa/percentage metrics (Halford/Scheuer) inside the boxes; keep labels short.
- No complex 3D shapes; use clean 2D vector style.