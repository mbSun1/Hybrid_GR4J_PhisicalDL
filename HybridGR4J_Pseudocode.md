# Hybrid Hydrological Model Pseudocode for Runoff Simulation

## Model Objective
Simulate daily runoff {Q̂ₜ} from meteorological inputs {Pₜ, Tₜ, Dₜ}, combining physical modeling (PBM-RNN) with neural network correction (Conv1D).

---

## Input
- Meteorological sequence: {Pₜ, Tₜ, Dₜ}, t = 1, ..., T
  - Pₜ: Precipitation (mm/day)
  - Tₜ: Temperature (°C)
  - Dₜ: Day length (hour)
- Observed runoff sequence: {Q_obsₜ}

## Output
- Physical parameters θ⁽ᴳ⁾ = {X₁, X₂, X₃}
- Neural network parameters θ⁽ᴺ⁾ (Conv1D weights and biases + unit hydrograph kernels)
- Simulated runoff sequence: {Q̂ₜ}

---

## Stage 1: Model Initialization

1. Define PBM-RNN layer:
   Qₜ = PBMRNN(Pₜ, Tₜ, Dₜ; θ⁽ᴳ⁾)

2. Initialize θ⁽ᴳ⁾:
   - X₁ ∈ [1, 2000]
   - X₂ ∈ [-20, 20]
   - X₃ ∈ [1, 300]

3. Compute potential evapotranspiration:
   - PETₜ = Hamon(Tₜ, Dₜ)

4. Build input vector:
   - xₜ = [Pₜ, Tₜ, PETₜ]

5. Initialize neural network parameters θ⁽ᴺ⁾, including:
   - Unit hydrograph convolution kernels: W_uh1, W_uh2
   - Convolution layers: W₁, b₁, W₂, b₂

---

## Stage 2: Physical Modeling & Routing (PBM-RNN + Unit Hydrograph)

For t = 1 to T:

1. Compute net precipitation and evapotranspiration:
   - If Pₜ ≥ PETₜ:
     - Pnₜ = Pₜ - PETₜ
     - Enₜ = 0
   - Else:
     - Pnₜ = 0
     - Enₜ = PETₜ - Pₜ

2. Update soil moisture storage:
   - Psₜ = f1(Pnₜ, Sₛₜ, X₁)
   - Esₜ = f2(Enₜ, Sₛₜ, X₁)
   - Percₜ = infiltration(Sₛₜ, X₁)
   - Sₛₜ ← Sₛₜ + Psₜ - Esₜ - Percₜ

3. Update groundwater storage:
   - Fₜ = recharge(Sᵣₜ, X₂, X₃)
   - Qᵣₜ = baseflow(Sᵣₜ, X₃)
   - Sᵣₜ ← Sᵣₜ + Fₜ - Qᵣₜ

4. Separate runoff components:
   - uh1ₜ = 0.9 × (Percₜ + Pnₜ - Psₜ)
   - uh2ₜ = 0.1 × (Percₜ + Pnₜ - Psₜ)

5. Define unit hydrographs:
   - UH₁ = exp(-i / 5), i = 0..29
   - UH₂ = exp(-i / 10), i = 0..29
   - Normalize UH₁, UH₂

6. Routing via convolution:
   - Q_uh1ₜ = UH₁ * uh1ₜ
   - Q_uh2ₜ = UH₂ * uh2ₜ

7. Final physical flow:
   - Q_physₜ = Qᵣₜ + Q_uh1ₜ + Q_uh2ₜ

---

## Stage 3: Neural Network Correction (ConvNet)

1. Build input:
   - CNN_inputₜ = [Pₜ, Tₜ, PETₜ, Q_physₜ]

2. ConvNet structure:
   - Q̂ₜ = ELU(W₂ · ELU(W₁ · xₜ + b₁) + b₂)

---

## Stage 4: Model Training

1. Set hyperparameters:
   - Learning rate η, Epochs N_epoch, Batch size B

2. Training loop:
   - For epoch in 1..N_epoch:
     - For batch (x, Q_obs):
       - Forward pass:
         - Q_phys = PRNN(x; θ⁽ᴳ⁾)
         - Q̂ = ConvNet([x, Q_phys]; θ⁽ᴺ⁾)
       - Loss:
         - ℒ = 1 - NSE(Q̂, Q_obs)
       - Backpropagation & update:
         - θ⁽ᴳ⁾ ← θ⁽ᴳ⁾ - η · ∇_{θ⁽ᴳ⁾} ℒ
         - θ⁽ᴺ⁾ ← θ⁽ᴺ⁾ - η · ∇_{θ⁽ᴺ⁾} ℒ

---

## Stage 5: Evaluation & Output

1. Model prediction: Q̂_test = Model(test_x)
2. Metrics: NSE, RMSE, KGE, Pearson R
3. Save: predictions, parameters, evaluation plots

---

## Appendix1: Unit Hydrograph Convolution

Q_uhₜ = ∑ (i = 1 to W_uh) [ UHᵢ × Pᵣ^(t − i + 1) ]

Where:
- Qₜ: routed runoff at time t
- UHᵢ: the i-th weight of the unit hydrograph
- Pᵣ^(t−i+1): effective runoff input at time (t−i+1)
- W_uh: number of steps in the unit hydrograph (kernel length)

---

## Appendix2: Neural Network Parameter Set

θ⁽ᴺ⁾ = {W₁, b₁, W₂, b₂, W_uh1, W_uh2}

Where:
- W₁, b₁: First Conv1D layer (kernel=10) weights and bias
- W₂, b₂: Second Conv1D layer (kernel=1) weights and bias
- W_uh1: Unit hydrograph kernel for fast flow (uh1_conv.weight)
- W_uh2: Unit hydrograph kernel for slow flow (uh2_conv.weight)