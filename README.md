# LiBeB Spallation Production Framework

### High-fidelity modeling of Li, Be, and B synthesis in solar and stellar flares

This repository contains a modular, scientifically rigorous Python framework for computing **light-element (Li, Be, B) production via nuclear spallation** in stellar atmospheres.

The physics implemented here is based on the classical flare gamma-ray literature (Ramaty, Kozlovsky, Murphy, Dermer, Mandzhavidze, Tatischeff, etc.) and extends it with modern numerical modeling:

- Depth-resolved transport of flare-accelerated ions  
- Bethe–Bloch stopping power with optional depth dependence  
- Rigidity-based shock spectra & stochastic K₂ spectra  
- Range-inversion \(E_0(E, X)\) solver  
- Modular cross-section framework  
- Slice-wise production \(Q'(X)\) and cumulative yields \(Q(X)\)  
- CSV-based spectral sweeps  

---

# 1. Scientific Background

Energetic ions (p, \(^{3}\)He, \(^{4}\)He) accelerated in flares collide with ambient nuclei (C, N, O, He), producing Li, Be, and B via **nuclear spallation**.

The instantaneous production rate for a reaction channel is:

$$
Q'(X) = \int_{E_{\min}}^{E_{\max}} 
\sigma(E)\, \phi(E, X)\, dE .
$$

The cumulative yield to depth \(X\):

$$
Q(X) = \int_0^X Q'(X')\, dX' .
$$

The framework numerically evaluates these integrals with physically consistent **spectra**, **stopping powers**, **range inversions**, and **cross sections**.

---

# 2. Core Physical Components

## 2.1 Bethe–Bloch stopping power

The framework computes the energy-loss per grammage using:

$$
S(E) = -\frac{dE}{dX}.
$$

A detailed Bethe–Bloch implementation includes:

- Projectile charge \(z\)  
- Mixture-averaged target \(Z/A\)  
- Electron density profiles \(\rho(X)\)  
- Effective ionization potential \(I(X)\)  
- Optional density effects  

This supports physically accurate propagation through chromospheric and photospheric layers.

---

## 2.2 Range-inversion \(E_0(E,X)\) solver

Continuous slowing down gives:

$$
X = \int_E^{E_0} \frac{dE'}{S(E')}.
$$

We define the *range function*:

$$
R(E) = \int_0^{E} \frac{dE'}{S(E')}.
$$

Then invert numerically:

$$
E_0 = R^{-1}(R(E) + X).
$$

This is the central step enabling depth-dependent spectra.

---

## 2.3 Particle Spectra

### **Stochastic (2nd-order Fermi) Acceleration**

$$
\phi_{\text{stoch}}(E) \propto 
K_2\!\left( \sqrt{\frac{E}{\alpha T}} \right).
$$

### **Shock Acceleration (rigidity-based)**

$$
\phi_{\text{sh}}(E)
\propto \frac{1}{\beta(E)}\, p(E)^{-s}
\exp\!\left[-\frac{E}{E_{0,i}}\right].
$$

Species-dependent cutoffs satisfy the Murphy–Dermer–Ramaty (1987) condition:

$$
v(E_{0,i})\,R(E_{0,i}) = v(E_{0,p})\,R(E_{0,p}).
$$

This allows physically consistent cutoffs for p, He-3, and He-4.

---

## 2.4 Cross Sections

All forward LiBeB-producing channels are supported:
$$
\(p + \mathrm{CNO} \rightarrow \mathrm{LiBeB}\)
\(^{3}\mathrm{He} + ^4\mathrm{He} \rightarrow ^6\mathrm{Li}\)
\(^{4}\mathrm{He} + ^4\mathrm{He} \rightarrow ^6\mathrm{Li}, ^7\mathrm{Li}\)
\(\alpha + \mathrm{CNO} \rightarrow \mathrm{LiBeB}\)
$$

Cross sections are read from CSV and interpolated with:

- Linear  
- PCHIP  
- Log–log  

---

# 3. Q′ and Q Calculation

Slice-wise production:

$$
Q'(X) = \int \sigma(E)\, \phi(E, X)\, dE.
$$

Cumulative yield:

$$
Q(X) = \int_0^X Q'(X')\, dX'.
$$

The framework produces:

- Per-slice \(Q'(X)\)  
- Integrated Q  
- CSV output for parameter sweeps  
- Depth-profiles for each channel  

---