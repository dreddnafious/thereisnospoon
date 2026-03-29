# Machine Learning: A Practitioner's Mental Model

A high-signal reference built from first principles. Not a textbook — a mental model for engineers who want to reason about ML systems the way they reason about software systems.

**How to read this.** Each section builds on the ones before it. The Neuron and Composition sections establish the geometric intuition that everything else depends on. Learning as Optimization requires the chain rule, which requires derivatives. The Combination Rule Family and Transformer sections assume you understand the neuron, composition, and optimization. Gates assume all of the above. If something doesn't make sense, check the earlier section it references.

**Notation.** `σ` = sigmoid function. `⊙` = element-wise (Hadamard) multiplication. `Φ(z)` = cumulative distribution function of the standard normal. `||x||` = magnitude (norm) of vector x. `∑` = sum. `ε` = infinitesimally small value.

---

## The Neuron

A neuron computes one number from a vector of inputs.

```
z = w · x + b
output = f(z)
```

**The dot product** (`w · x`) measures how much the input **x** points in the same direction as the weight vector **w**.

```
w · x = |w| |x| cos(θ)
```

Two vectors in, one scalar out. The result depends on three things: the direction of **w** (what the neuron is selective for), the magnitude of **w** (how aggressively it responds), and the input itself. If the input is perfectly aligned but tiny, z is small. If the input is large but orthogonal to **w**, z is zero.

A neuron is a half-space detector. The set of all inputs where `w · x + b = 0` forms a boundary — a line in 2D, a plane in 3D, a hyperplane in higher dimensions. On one side z is positive, on the other negative. The magnitude of z tells you how far from the boundary you are. The neuron gives you a side (which half) and a distance (how deep).

![A single neuron divides 2D input space along a hyperplane. Left: z values as a gradient showing positive and negative regions. Right: after ReLU, the negative side is clamped to zero.](figures/01_neuron_hyperplane.png)

**The bias** is a scalar, not a vector. It shifts the boundary. Without it, the hyperplane always passes through the origin. With it, the boundary can be positioned anywhere. The bias is a threshold — how aligned does the input need to be before the neuron responds positively?

**The nonlinearity** (`f`) transforms z into the neuron's output. Without it, stacking layers collapses into a single linear operation. The nonlinearity breaks linearity and gives the network expressive power.

### Activation Functions as Design Decisions

Each activation function is a different policy for what to do with the side-and-distance information that z provides. Think of gradient flow as a signal propagating through a pipeline. Each nonlinearity is a valve.

**ReLU: `max(0, z)`** — A gate valve. Fully open or fully closed. If z > 0, pass it through unchanged. If z < 0, output zero. The gradient is 1 (active) or 0 (dead). Active neurons pass the gradient perfectly — no attenuation. This is why ReLU solved vanishing gradients. Failure mode: neurons can die permanently. If weights update so z is negative for every input, gradient is always zero and the neuron never recovers.

**Sigmoid: `1 / (1 + e^(-z))`** — A pressure regulator. Squashes everything to (0, 1). Smooth, bounded. But the gradient approaches zero for large |z| — the neuron *saturates*. Confident neurons stop learning. In deep networks, saturation compounds: gradients shrink exponentially across layers. This is the vanishing gradient problem.

**tanh** — Same shape as sigmoid but centered on zero, range (-1, 1). Same saturation problem, but zero-centering helps gradient flow. Better than sigmoid in practice, but both saturate.

**RePU: `max(0, z)^p`** — An amplifier. Polynomial activation. A single neuron can represent curves, not just lines. But the gradient grows with z, so large activations produce large gradients. Risk: gradient explosion — the opposite failure mode from sigmoid.

**GELU: `z * Φ(z)`** — A proportional valve. Multiplies z by the probability that z is "large" under a standard normal distribution. Near zero, it smoothly modulates rather than hard-switching. No dead zone, no hard cutoff. Transformers use GELU because attention produces many near-zero signals, and the smooth response lets the network learn fine distinctions near the decision boundary.

![Activation functions and their derivatives side by side. Each function is a different valve on the signal pipeline.](figures/02_activation_functions.png)

### The Polarizing Filter Analogy

A neuron is a polarizing filter. Light has a polarization direction (the input). The filter has a preferred axis (the weight vector). Malus's Law says transmitted intensity is `I_in * cos²(θ)` — the component aligned with the filter passes through. Everything else is silently ignored. The weight vector defines what the neuron "cares about." The dot product extracts how much of the input aligns with that axis.

---

## Composition: Depth, Width, and Paper Folding

### Width: More Cuts Per Layer

A single neuron makes one cut. Two neurons make two cuts, creating up to four regions. n neurons give up to 2^n regions. A layer is a collection of simultaneous cuts partitioning input space.

But every boundary is flat. A single layer can only carve convex regions — like facets of a gem. If the data has curved or non-convex structure, a single layer must approximate it with many small flat cuts, like tracing a circle with a polygon.

### Depth: Folding Space

Take a flat sheet of paper. This is your input space.

First neuron in a layer: draw a line, fold along it. One side stays up, the other folds flat. Points on the dead side lose their identity along that dimension. Points on the active side keep their position.

More neurons: more folds, more creases. The paper is now a creased, partially flattened object.

The next layer draws straight lines on this folded paper and cuts. Unfold the paper. The straight cuts are no longer straight — they're bent at every crease. The folds turned straight cuts into complex boundaries.

**Each layer folds. Each subsequent layer cuts. The folds make simple cuts produce complex boundaries in the original space. The folds are the activation function.** Without the activation function there's no fold — the space passes through unchanged and depth adds nothing.

![Paper folding: original space, one ReLU fold, two ReLU folds. Straight cuts in folded space become kinked cuts when unfolded.](figures/03_paper_folding.png)

### Width vs Depth Tradeoff

A single wide layer can approximate any function (universal approximation theorem). But "enough width" can mean absurdly many neurons. A function with n nested oscillations takes O(n) neurons deep but O(2^n) neurons wide. Depth gives exponential efficiency for hierarchically structured functions.

**Too wide, too shallow**: massive capacity, no structural bias. Solutions tend to be brittle — lots of finely tuned cuts rather than a clean hierarchy. Training is harder because the loss landscape is broad and flat.

**Too deep, too narrow**: forced into a sequential pipeline. Information must survive every layer — if it's not useful at layer 3 but needed at layer 10, it can be destroyed (bottleneck problem). Longer chain rule products mean worse gradient pathology. Needs tricks like skip connections or normalization.

**Sweet spot**: moderate depth with moderate width. Hierarchical problems (images, language) reward depth. Flat discrimination problems (tabular data) often do better with width.

---

## Learning as Optimization

### Derivatives and the Chain Rule

A derivative is rise over run at the limit — how much the output changes per infinitesimal change in input. It's a local exchange rate.

For `f(x) = x²`: nudge x by ε, expand `(x + ε)² = x² + 2xε + ε²`, subtract f(x), divide by ε, let ε vanish. What survives is 2x. The power rule generalizes: `d/dx[x^n] = nx^(n-1)`.

Common derivative rules:
```
f(x) = x^n   →  f'(x) = nx^(n-1)    power rule
f(x) = e^x   →  f'(x) = e^x          exponential (rate of change equals itself)
f(x) = ln(x) →  f'(x) = 1/x          logarithm
f(x) = sin(x)→  f'(x) = cos(x)       trig
f(x) = c     →  f'(x) = 0            constant
```

**The chain rule**: for composed functions, the total rate of change is the product of the local rates. Like a gear train — each gear has a ratio, the overall ratio is the product. Like unit conversion — miles/gallon × gallons/hour = miles/hour.

```
f(g(x)):  df/dx = df/dg * dg/dx
```

For a neuron: `z = wx + b`, `a = ReLU(z)`, `L = (a - y)²`

```
dL/dw = dL/da * da/dz * dz/dw
      = 2(a-y) * (1 or 0) * x
```

The weight update is proportional to how wrong the output was, gated by whether the neuron was active, scaled by the input. Each term makes sense alone. The chain rule multiplies them.

![The chain rule as a pipeline. Forward signal flows right, backward gradient flows left. Total gradient is the product of local rates.](figures/04_chain_rule_pipeline.png)

**Backpropagation** is the chain rule applied layer by layer from output to input. Each layer receives an error signal from above, computes its weight updates, and passes a transformed error signal backward. The activation function's derivative gates the backward signal at each layer — this is why the choice of nonlinearity matters for learning, not just representation.

### The Loss Landscape

Every possible configuration of weights maps to a loss value. This surface has as many dimensions as there are weights. Training is finding a low point.

**Gradient descent**: compute the slope in every direction, step opposite to steepest ascent. Repeated small steps downhill.

### Why Things Get Stuck

**Vanishing gradients**: chain rule product of factors less than 1 shrinks exponentially through layers. Sigmoid's max derivative is 0.25 — five layers gives at most 0.25⁵ ≈ 0.001 of the original signal. ReLU's derivative is 1 when active, so gradients pass through at full strength.

**Exploding gradients**: chain rule product of factors greater than 1 grows exponentially. Large weights or polynomial activations can cause this. Weight updates become enormous and the network diverges.

**Saddle points**: zero gradient, but not a minimum. Like sitting on a horse saddle — minimum front-to-back, maximum side-to-side. In high dimensions, saddle points vastly outnumber true minima. For a point to be a true minimum, every direction must curve up. In a million dimensions, a random critical point will have roughly half its directions curving up and half down.

**Sharp vs flat minima**: both have low loss, but sharp minima are fragile — small perturbations in data send loss shooting up. Flat minima are robust and generalize better. A high learning rate bounces out of sharp minima but can settle into broad ones — implicit selection for generalization.

---

## Generalization

### Why It Works (And Shouldn't)

Classical theory says: more parameters than data points means overfitting. Neural networks violate this — massively overparameterized yet they generalize. Not fully understood, but several contributing factors:

**The optimizer is biased.** Gradient descent from random initialization finds simpler functions first — smooth, low-frequency patterns before noise. Early stopping captures pattern without noise. The optimizer has an implicit bias toward simplicity.

**The parameterization constrains.** A million parameters don't act independently. They interact through layers — changing one weight affects the function globally. The network can't easily be an arbitrary lookup table. The parameter count overstates the effective complexity.

**Flat minima.** SGD with mini-batch noise settles in broad regions of the loss landscape. These correspond to solutions robust to perturbation — robust to train/test differences.

### Regularization as Design Philosophy

Not a checklist of techniques. A single principle: **constrain the search space so the solutions the optimizer finds are ones that generalize.** Each method does this differently, addressing different failure modes:

**L2 regularization (weight decay)** — penalizes large weights, pushing toward smaller, smoother functions. Among all solutions that fit the data, prefer the simpler one.

**Dropout** — randomly zeroes neurons during training. Forces redundancy — no single neuron can memorize a specific feature. An ensemble method hiding inside a single network.

**Data augmentation** — expands training set with transformations. Not a regularizer on the model — a regularizer on the data. Fills gaps in the sparse sample of the true distribution.

**Early stopping** — stop when validation loss starts rising. The network learned the pattern and is now fitting noise.

---

## Representation: What Networks Actually Store

### Features as Directions

A randomly initialized network has no meaningful directions. Training carves out directions in activation space that reduce loss. A feature is a direction the network consistently uses to represent a property of the input. Features aren't designed — they're the residue of optimization.

The individual node is not the unit of meaning. The vector produced by the entire layer is. A node is one coordinate. A feature is a direction in the full vector space, spread across all nodes. Two features are similar if their directions are nearly parallel. They're unrelated if nearly perpendicular.

### Superposition

The number of nearly-perpendicular directions in high-dimensional space vastly exceeds the number of dimensions. A 4096-dimensional layer can encode far more than 4096 features by superimposing them — packing more features than dimensions, arranged to minimize mutual interference.

Features that share substrate tend to be features that rarely co-occur in the data (low collision risk). Related features cluster in nearby directions (geometric neighborhoods). The structure is learned, not random.

**Superposition is load-bearing for generalization.** Shared substrate forces the network to capture structure — relationships between features are baked into the geometry. Novel inputs activate blended patterns and get reasonable responses. The cost: interference between features when they do co-occur, which is one contributor to hallucinations (among multiple causes).

Vector arithmetic works approximately in neural representations (king - man + woman ≈ queen) because directions encode semantic relationships. The algebra is a side effect of optimization, not a guarantee. Some directions compose cleanly, others don't.

### Distributed Representation and Its Consequences

Knowledge is distributed across the geometry of entire layers, not localized in individual neurons. This means:

**You can't surgically modify knowledge.** Editing one direction risks perturbing others that share the same neurons.

**Continual learning is hard.** New features need directions. Creating them perturbs existing directions. Old knowledge degrades. This is catastrophic forgetting — a consequence of distributed representation, not a failure of the optimizer. Approaches exist (EWC: protect important weights; progressive networks: freeze old, add new; memory replay: mix old and new data) but none solve the fundamental problem.

**Knowledge decoupling is hard.** Knowledge (which features exist) and computation (how features combine) are the same weights. A weight matrix simultaneously defines which directions exist and how the next layer transforms them.

### Transfer Learning

A trained network's weights provide better initialization than random — features are partially formed, and the optimizer has less distance to travel. This works when source and target tasks share structure. It fails when they don't.

Transplanting trained layers into an untrained network: freeze them and they work as fixed feature extractors. Don't freeze them and random downstream layers send incoherent gradients that destroy the trained features. A layer's weights are adapted to the layers around it — remove that context and the contract is broken.

### Training Is Not Deterministic

GPU floating point operations don't guarantee execution order — identical inputs can produce different rounding errors that compound. Two runs from different initializations converge to functionally similar but geometrically different solutions. The loss landscape has many equally good minima — which one the optimizer finds depends on the path taken.

---

## The Combination Rule Family

Every architecture does three things: combine inputs according to some rule, apply a nonlinearity, repeat. The only difference between architectures is step 1 — the combination rule. That's where the inductive bias lives.

**Dense** — every input connects to every output. No assumption. Maximum flexibility, maximum parameters.

**Convolution** — small sliding dot product with shared weights. Assumes locality and translation invariance. Far fewer parameters. Hierarchical feature composition: edges → textures → objects across layers. Pooling, stride, and dilation control how the receptive field expands. Breaks down when locality is wrong, when translation invariance is wrong, or when data isn't on a grid.

**Recurrence** — sequential state-carrying. `h_t = f(W_h · h_(t-1) + W_x · x_t + b)`. A fixed-size vector summarizes all history. Each step is a lossy relay — the entire upstream history must fit in one vector. Eigenvalues of W_h determine information decay rates per direction (eigenvalue < 1: information decays; > 1: it explodes; = 1: preserved). Vanilla RNNs fail beyond ~10-20 steps. LSTMs add an additive cell state (conveyor belt) with learned gates (forget, input, output) — same trick as residual connections. GRUs simplify to one update gate. Attention replaced recurrence for parallelism, no compression bottleneck, and shorter gradient paths. Recurrence still wins for streaming, constant memory, and strict causality.

**Attention** — dynamic, content-dependent combination. Three projections per element: query (what am I looking for?), key (what do I contain?), value (what do I provide?). Dot product of query against all keys determines relevance. Softmax normalizes to a distribution. Output is weighted sum of values. Q/K/V are the complete decomposition of a routing operation — no additional projections have proven necessary.

**Graph operations** — message passing over explicit topology. Nodes collect from neighbors, aggregate, update. Convolution generalized to irregular structure. Excels when relationships are known (molecules, physics). Oversmoothing limits depth — too many rounds and all nodes converge.

**State-space models** — continuous-time recurrence from control theory. `dx/dt = Ax + Bu, y = Cx + Du`. Can be computed as recurrence (O(1) memory) or convolution (parallel training). HiPPO provides mathematically optimal history compression. Mamba added input-dependent gating for content-sensitive processing. Bridges recurrence and convolution.

**Sparse/structured matrices** — constrained connectivity for efficiency. Block-diagonal (independent groups), low-rank (bottleneck factorization), butterfly (hierarchical pairwise mixing in O(N log N)). Sometimes matches data structure, sometimes purely a compute approximation.

---

## The Transformer

A transformer block: self-attention (route information between elements), feed-forward network (apply stored knowledge per element), both wrapped in residual connections and layer normalization. Repeated N times.

### Self-Attention

Each element in the sequence generates Q, K, V vectors via learned projection matrices. Each element's query is dot-producted against all keys to determine relevance. Softmax produces attention weights. Output is weighted sum of values. Connectivity is dynamic — determined by content, computed fresh for every input.

![Self-attention on a 5-token sequence. Left: raw Q·K scores. Center: after softmax. Right: "it" attends mostly to "cat" — pronoun resolution emerges from dot product alignment.](figures/05_attention_mechanism.png)

### Multi-Head Attention

Multiple independent attention operations in parallel, each on a slice of the vector. Each head learns a different relevance pattern (syntactic, semantic, positional). Outputs are concatenated losslessly, then a projection remixes them into a unified representation. This isn't finer resolution of the same measurement — it's multiple *different* measurements. Each head asks a different question about which elements are relevant.

### Positional Encoding

Attention is permutation-invariant — it has no concept of order. Position must be injected. Sinusoidal (fixed mathematical patterns), learned embeddings (vector per position), RoPE (rotation applied to Q/K so dot products reflect relative distance — existing linear algebra property recognized as exact solution), ALiBi (distance-based penalty on attention scores).

### Feed-Forward Layers

Per-element, position-independent. Expansion to higher dimension (512 → 2048), GELU activation, contraction back (2048 → 512). This is a volumetric lookup — the vector activates a neighborhood in a learned feature space. The activation region is irregular and content-dependent — spines of varying length in non-uniform directions, with fuzzy boundaries from GELU. Features within that region are blended and compressed back.

Attention routes information between elements. The FFN applies stored knowledge per element. Division of labor is clean.

![FFN as volumetric lookup. Left: input vector. Center: expanded space with activated neighborhood (bright) and silent features (gray). Right: architecture — expand, activate, contract.](figures/06_ffn_volumetric.png)

FFN layers are where factual knowledge lives. Specific rows of W₁ activate for specific facts. This is the basis for model editing techniques (ROME, MEMIT) — treating the FFN as a key-value memory and writing directly to it.

### Residual Connections

`output = x + Sublayer(x)`. Each layer adds a delta rather than replacing. The vector is a running sum: `x + delta_1 + delta_2 + ... + delta_N`. Preservation is the default — a layer must actively write to contribute, but does nothing to preserve. Gradient flows directly through addition (derivative = 1) regardless of depth. Individual deltas recoverable by subtracting consecutive intermediate vectors.

![Residual connections. Left: each block adds a delta to the running sum — preservation is the default. Right: gradient flows at full strength through the additive highway.](figures/07_residual_connections.png)

### Encoder-Decoder vs Decoder-Only

Encoder-decoder: encoder processes full input with bidirectional attention, decoder generates output token by token with causal masking, cross-attention connects them. Stronger inductive bias for tasks with distinct input/output (translation).

Decoder-only: single decoder with causal masking. Input and output are the same sequence. Simpler, more general. At sufficient scale, the generality wins.

---

## Encoding

The encoder converts raw data into the vector format the downstream architecture operates on. It determines the ceiling — the model can only learn from what the encoder preserves.

Three decisions: what is an element (granularity), what does each element's vector capture (raw vs contextualized), and what structural information is preserved (order, adjacency, hierarchy).

**The encoder must preserve what you need, and you decide that before choosing the encoder, not after.** A pretrained encoder optimized for one objective may actively destroy information another task requires. End-to-end training solves the encoder-model alignment problem — the gradient tells the encoder what the downstream model needs. Without end-to-end training, you're hand-engineering the interface.

**Per-modality encoders**: text (tokenize + embed), images (pixels for CNN, patches for ViT), audio (spectrogram + conv front-end), graphs (define nodes, edges, features). Each specialized for its data type.

**Multimodal alignment**: adapter per modality projecting into shared dimensionality. The projection is easy — the alignment is hard. Contrastive training (CLIP) provides the signal that makes semantically matched inputs from different modalities land in the same region of the shared space.

---

## Learning Rules

**Backpropagation** — exact global gradient via chain rule. Dominant because nothing else matches its efficiency at scale. Downsides: requires differentiability (can't backprop through discrete decisions), requires storing all activations (memory scales with depth), backward pass as expensive as forward, sequential layer-by-layer backward (synchronization bottleneck), catastrophic forgetting (gradient only sees current batch), no learning at inference time, gradient pathology scales with depth.

**Hebbian learning** — "neurons that fire together wire together." `Δw = η · x_i · x_j`. Local, no global error signal. Learns correlations, not task mappings. Natural associative memory, supports continual learning. Capacity ~0.14N patterns for N neurons.

**Modern Hopfield networks** — exponential energy function dramatically increases capacity. The update rule is mathematically equivalent to transformer attention. Attention *is* Hopfield retrieval.

**Contrastive learning** — training objective, not a weight update rule. Make similar things close, dissimilar things far. Learns representations without labels. Uses backprop for weight updates, but the training signal comes from data structure, not human labels.

**Energy-based models** — define energy over configurations, learn the landscape, inference is energy minimization. Enables iterative refinement at inference — the model can "think longer" about hard inputs. Multiple low-energy states signal ambiguity. Cost: slower than single forward pass.

**Learned loss functions** — when the objective is too complex to specify mathematically (what makes a good image? a helpful response?), train a network to learn the objective from examples. GANs do this (discriminator is a learned loss). RLHF does this (reward model is a learned loss). General principle: when you can judge quality but can't formalize it, learn the loss function.

---

## Frameworks

The training objective — what the model optimizes for. Independent of architecture and learning rule.

**Supervised** — inputs and correct outputs. Model predicts, compares to answer, updates. Limited by labeled data.

**Self-supervised** — hide part of input, predict the hidden part. Labels come from data itself. Masked language modeling (BERT), next-token prediction (GPT). Power is scale — unlabeled data is effectively unlimited.

**Reinforcement learning** — no correct answers, only rewards. Model takes actions, receives sparse/delayed reward, learns a policy. Much harder than supervised — weak learning signal, credit assignment problem. RLHF uses RL to align LLMs with human preferences that next-token prediction doesn't capture.

**GANs** — generator produces fake data, discriminator distinguishes real from fake. Adversarial dynamic produces sharp, high-quality samples. Training is notoriously unstable (mode collapse, balancing). Dominated image generation 2016-2021, then replaced by diffusion.

**Diffusion** — define a noise-adding process that destroys data over T steps. Train a network to reverse each step. Generation: start from noise, iteratively denoise. Training is stable (simple regression objective — predict the noise). Quality matches GANs. Cost: hundreds of denoising steps per sample. Key insight: iterative refinement allocates compute proportionally to difficulty.

---

## Topology for the Problem

Architecture choice is driven by data structure, task requirements, data quantity, and compute constraints.

| Question | Signal | Architecture |
|----------|--------|-------------|
| Data on a grid? | Spatial locality, translation invariance | Convolutions |
| Sequential data? | Order matters, history needed | Attention, recurrence, or SSMs |
| Known relational structure? | Explicit edges between entities | Graph networks |
| Tabular data? | Flat features, no spatial/temporal structure | Dense layers or gradient-boosted trees |
| Unknown/complex structure? | No clear structural prior | Transformer |
| Very long sequences? | Memory and compute constraints | SSMs or sparse attention |
| Streaming / real-time? | Can't buffer full input | Recurrence or SSMs |
| Very little data? | Need strong inductive bias | Architecture with matching assumptions |
| Abundant data? | Can learn structure from scale | Transformer |

**Practical reality:** most practitioners start with a transformer. If it works, they stop. At sufficient scale, architecture matters less — the transformer learns what other architectures would have given for free. The counterpoint: that's wasteful when the budget is limited and the right inductive bias would do the same job with 10x fewer parameters.

**Hybrid architectures** use the cheapest operation at each level. Convolutions for local patterns. Attention for global routing. Recurrence/SSMs for long-range persistence. The design principle: match the operation to the structure at each stage of processing.

---

## Gates as Control Systems

### What a Gate Is

A gate is a function that modulates a signal through multiplication. That's the full definition. Everything else is variations on scope and complexity.

**Scalar gate** — one number applied uniformly: `output = g · x`. The whole signal is scaled together.

**Vector gate** — per-dimension independent control: `output = g ⊙ x`. Each dimension is scaled independently. This is the LSTM forget gate. The cost-expressiveness sweet spot for most applications.

**Matrix gate** — full cross-dimension mixing: `output = G · x`. Can rotate, project, and recombine dimensions. Attention is a matrix gate. More expressive, more expensive (d² vs d parameters).

A gate can be computed by an arbitrarily complex function — a two-layer network, a small transformer, anything. The boundary between "gate" and "subnetwork" is semantic: the output is used multiplicatively to modulate another signal. The practical constraint: the gate should be cheaper than the thing it's gating.

### Composing Gates — Soft Logic

Continuous differentiable analogs of digital logic:

- **NOT**: `1 - g` (invert)
- **AND**: `g₁ · g₂` (both must be high)
- **OR**: `g₁ + g₂ - g₁ · g₂` (either suffices)
- **Conditional**: `g · x₁ + (1 - g) · x₂` (soft if-then-else, interpolate between paths)

These compose into arbitrary soft logic. Everything is continuous — no hard 0s and 1s. This makes it differentiable and trainable, but the logic is never perfectly crisp.

### Branching and Routing

**Soft routing**: all paths execute, gate blends results. Differentiable but expensive — every path runs.

**Hard routing**: one path executes. Cheap but non-differentiable — can't backprop through argmax.

**Top-k routing** (MoE): select top 2 of 64 experts, blend those 2. Mostly hard, small soft blend. Practical compromise.

**Gumbel-softmax**: bridge between soft and hard. Add random noise, apply softmax with temperature. High temperature = soft (gradient flows). Low temperature = nearly hard. Anneal during training: start soft, end hard. At inference, use hard argmax. Training taught the router to make good discrete decisions through soft approximations.

**Analytical gate initialization**: compute the weight configuration that makes a gate fire for a target input class by backpropping through just the gate network. Set routing deterministically, let experts train under fixed routing. Decompose the learning problem — routing is set analytically, processing trains independently.

### Recursion Within a Forward Pass

**Adaptive computation time (ACT)**: a halting gate per element decides "am I done?" at each iteration. Elements that halt stop updating. Different elements take different numbers of passes. Easy inputs exit early, hard inputs go deep.

**Universal transformers**: one block with shared weights, applied repeatedly with learned halting. Recursion — the same function applied until a stopping condition.

**Fixed-point iteration**: apply a block repeatedly until the output stops changing. `x* = f(x*)` — the output the function maps to itself. This is energy minimization — the system relaxing toward equilibrium. No learned halting needed; convergence is the stopping condition.

**Deep equilibrium models (DEQ)**: find the fixed point directly using root-finding algorithms. Implicit differentiation gives the gradient at the fixed point without backpropagating through all iterations. Constant memory regardless of effective depth. The function must be contractive (each application brings points closer) or the iteration diverges.

### The Math Toolbox for Gates

**Sigmoid** — maps any real number to (0, 1). The universal "make a proportion" function. Steepness controlled by input magnitude. Use anywhere you need "how much."

**Softmax** — maps N real numbers to a distribution summing to 1. The "choose one of N" function. Temperature parameter controls sharpness: high T = spread out, low T = winner-take-all, T→0 = argmax.

**Sparsemax** — like softmax but can output exact zeros. Projects onto the probability simplex. Use when you want genuinely sparse routing — some options get exactly zero weight.

**Straight-through estimator** — forward pass uses hard decision (round, argmax), backward pass pretends the hard decision didn't happen and passes gradient through the continuous input. Theoretically unjustified, practically effective. Use anywhere hard decisions are needed in differentiable pipelines.

### Linear Algebra of Gating

**Projection** — map a vector onto a subspace, discarding components outside it. Like casting a 3D shadow onto 2D — depth is lost but influenced the shadow's shape. Q/K/V projections in attention are literally this: different projection angles producing different views of the same high-dimensional object. Multi-head attention: multiple projections from different angles to reconstruct more of the original structure.

**Masking** — element-wise multiplication, zeroing specific axes. Axis-aligned projection. Cheaper than full projection (d vs d² operations) but less expressive — can only keep or kill individual dimensions, not diagonal directions.

**Rotation** — change direction without changing magnitude. Reorient the representation between "frames of reference." RoPE is a rotation. Orthogonal matrix gates reorganize information without losing it.

**Interpolation** — `g · x₁ + (1-g) · x₂` traces a straight line between two points in vector space. Everywhere: residual connections, gate blends, soft routing. **Spherical interpolation (slerp)** follows the arc of a sphere, preserving magnitude — used in diffusion sampling and model weight merging.

![Geometric gating operations: projection (shadow onto subspace), masking (zero specific axes), rotation (change direction, keep magnitude), interpolation (blend between representations).](figures/08_gating_operations.png)

All gating is some combination of: scaling, masking, projection, rotation, interpolation. The design skill is matching the geometric operation to the information flow requirement — identifying what functional behavior you need, then choosing the mathematical shape that provides it.

---

## Appendix: Diagnosing and Fixing Training Problems

### Reading the Loss Curve

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| Loss plateaus early, stays flat | Saddle point or dead gradients | Increase learning rate, add momentum, try Adam |
| Loss drops then hits a ceiling | Network too small or learning rate problem | Temporarily make network bigger; try LR schedule; check data quality |
| Loss oscillates wildly | Learning rate too high | Reduce LR; use adaptive optimizer (Adam, AdaGrad) |
| Training loss drops, validation diverges | Overfitting | Regularization, dropout, data augmentation, early stopping |
| Loss is NaN or infinity | Exploding gradients | Gradient clipping, reduce LR, check weight initialization |

### The Toolkit

**Learning rate** — the single most important hyperparameter. Schedules (high → low) almost always beat fixed rates. Warmup (low → high → low) helps when early gradients from random weights are unreliable.

**Momentum** (typically 0.9) — converts gradient descent from a ball rolling with friction into one rolling with inertia. Smooths noise, accelerates through consistent slopes, carries through flat regions.

**Adam** — combines momentum with adaptive per-parameter learning rates. Default choice for most problems. Tradeoff: can converge to sharper minima than SGD with momentum, which can hurt generalization. Some practitioners use Adam early, then switch to SGD for final training.

**Gradient clipping** — safety valve, not a fix. If you need aggressive clipping, something else is wrong.

**Batch/layer normalization** — reshape the loss landscape itself. Smooth the surface, reduce severity of valleys and plateaus. Neither is universal: batch norm breaks with small batches and sequential data; layer norm assumes all features should have similar scale. Both trade representational freedom for trainability. Usually worth it. Not always.
