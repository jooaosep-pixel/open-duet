"""Built-in examples so the app is usable before the benchmark is downloaded."""

EXAMPLES: list[dict[str, str]] = [
    {
        "title": "Retrieval-augmented reasoning agent",
        "intent": (
            "Figure 1: Overview of ReFlect. A planner decomposes the question into sub-queries, a retriever fetches evidence from "
            "the corpus, and a verifier scores each candidate answer; low-confidence answers are sent back to the planner for another round."
        ),
        "method_text": (
            "## 3 Method\n\nReFlect is an iterative retrieval-augmented reasoning framework with three components. Given a question q, "
            "the **Planner** (an LLM) produces a set of sub-queries {s_1, ..., s_k} that cover the information needed to answer q. "
            "Each sub-query is sent to the **Retriever**, a dense retriever over a document corpus D, which returns the top-m passages "
            "P_i for sub-query s_i. The Planner then drafts a candidate answer a conditioned on q and the union of retrieved passages. "
            "The **Verifier**, a separate LLM scored with a confidence head, assigns a confidence c in [0,1] to the candidate answer and "
            "produces a short critique. If c is above a threshold tau, the answer is returned. Otherwise the critique and the retrieved "
            "passages are appended to the Planner's memory and a new round begins, with the Planner issuing refined sub-queries. "
            "We cap the number of rounds at T=3. The training objective combines a supervised loss on final answers with a "
            "reinforcement learning reward that penalises the number of retrieval rounds."
        ),
    },
    {
        "title": "Masked diffusion for molecular graphs",
        "intent": (
            "Figure 2: Training and sampling of GraphMask-Diff. During training, atom and bond tokens are randomly masked and a graph "
            "transformer learns to predict them; at sampling time the graph is unmasked iteratively from a fully masked state."
        ),
        "method_text": (
            "## 3 GraphMask-Diff\n\nWe represent a molecule as a graph G=(V,E) where each node v carries an atom type token and each edge "
            "carries a bond type token. The forward process is a discrete masking diffusion: at time t, each token is independently "
            "replaced by a special [MASK] token with probability beta_t, where beta_t increases from 0 to 1. The denoising network f_theta "
            "is a graph transformer with alternating node-attention and edge-attention layers; it receives the partially masked graph "
            "together with a time embedding and outputs categorical distributions over atom types for masked nodes and over bond types "
            "for masked edges. Training minimises the cross-entropy of the predicted tokens against the ground-truth tokens on masked "
            "positions only. Sampling starts from a fully masked graph of a given size, and over K steps the model predicts all tokens, "
            "commits a fraction of the most confident predictions, and re-masks the rest, until no masks remain. Valency constraints are "
            "enforced by a projection step after each commit."
        ),
    },
    {
        "title": "Vision-language policy for robot manipulation",
        "intent": (
            "Figure 1: Architecture of PixelAct. Multi-view camera images and a language instruction are encoded, fused by a transformer "
            "with learnable action queries, and decoded into a chunk of future end-effector actions."
        ),
        "method_text": (
            "## 3 Approach\n\nPixelAct maps observations to actions. At each timestep the policy receives RGB images from three cameras "
            "(wrist, front, side) and a natural-language instruction. Each image is encoded by a shared ViT backbone into patch tokens; "
            "the instruction is encoded by a frozen text encoder. A fusion transformer attends over the concatenation of image tokens, "
            "language tokens and proprioceptive state tokens. We append H learnable action queries to the sequence; after fusion, "
            "each action query is decoded by a small MLP into a 7-DoF end-effector command, yielding an action chunk of horizon H=16. "
            "The policy is trained with an L1 loss on the action chunk against demonstrations, plus an auxiliary contrastive loss that "
            "aligns the pooled visual representation with the language embedding. At inference we execute the first 8 actions of each "
            "chunk before re-planning."
        ),
    },
]

PLOT_EXAMPLE: dict[str, str] = {
    "title": "Grouped bar chart of benchmark accuracy",
    "intent": (
        "Grouped bar chart comparing accuracy (%) of three methods on four benchmarks, with error bars (std) and the best method per "
        "benchmark highlighted. Legend at the top, y axis from 50 to 100."
    ),
    "method_text": (
        '{"benchmarks": ["GSM8K", "MATH", "HumanEval", "MMLU"],\n'
        ' "methods": {"Baseline": [71.2, 42.5, 60.1, 68.3], "Ours (w/o critic)": [78.9, 48.7, 66.4, 71.0], "Ours": [84.3, 55.2, 72.8, 74.6]},\n'
        ' "std": {"Baseline": [1.1, 1.4, 1.6, 0.8], "Ours (w/o critic)": [0.9, 1.3, 1.5, 0.7], "Ours": [0.8, 1.2, 1.1, 0.6]}}'
    ),
}
