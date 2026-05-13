# Section 3 — Notation Table (Reviewer #1, item #15)

| Symbol | Meaning | First use |
|--------|---------|-----------|
| \(t\) | Hourly time index, \(t=1,\dots,T\) | §3 |
| \(p_t\) | Closing price at time \(t\) | §3.0 |
| \(r_t\) | Hourly log-return \(\log(p_t/p_{t-1})\) | §3.0 |
| \(s_t\) | Unified state vector \([\mathbf{x}^{\text{tech}};\mathbf{x}^{\text{img}};\mathbf{x}^{\text{senti}}]\) | §3.0 |
| \(\mathbf{x}^{\text{tech}}_t\) | Technical-indicator features | §3.1 |
| \(\mathbf{x}^{\text{img}}_t\) | Candlestick-image features (ResNet-18) | §3.1 |
| \(\mathbf{x}^{\text{senti}}_t\) | FinBERT sentiment features | §3.1 |
| \(R_t\) | Regime label \(\in\{\text{Bear,Sideways,Bull}\}\) | §3.2 |
| \(\theta_{\text{conf}}\) | Regime-switch confidence threshold | §3.2 |
| \(\pi_\theta(a\mid s)\) | PPO policy with parameters \(\theta\) | §3.3 |
| \(w_k\) | Dynamic weight for agent \(k\) | §3.4 |
| \(W\) | Performance window for weight update (default 30 days) | §3.4 |
| \(T_{\text{temp}}\) | Softmax temperature for weighting (default 10) | §3.4 |
| \(\hat A_t\) | Generalised Advantage Estimate | §3.3 |
| \(\rho_t\) | Importance ratio \(\pi_\theta / \pi_{\theta_{\text{old}}}\) | §3.3 |
| \(\epsilon, c_1, c_2\) | PPO clip, value, entropy coefficients | §3.3 |
| \(\lambda_{\text{reg}}\) | Regime-specific turnover penalty | §3.3 |
| Acc, Prec, Rec, F1 | Standard classification metrics | §4 |

**Standard classification metrics (Reviewer #1).** For \(C\) classes
let \(\mathrm{TP}_c,\mathrm{FP}_c,\mathrm{FN}_c,\mathrm{TN}_c\) denote
class-\(c\) confusion counts. Then

\[
\mathrm{Acc} = \frac{\sum_c \mathrm{TP}_c}{N},\quad
\mathrm{Prec}_c=\frac{\mathrm{TP}_c}{\mathrm{TP}_c+\mathrm{FP}_c},\quad
\mathrm{Rec}_c=\frac{\mathrm{TP}_c}{\mathrm{TP}_c+\mathrm{FN}_c},\quad
\mathrm{F1}_c=\frac{2\,\mathrm{Prec}_c\mathrm{Rec}_c}{\mathrm{Prec}_c+\mathrm{Rec}_c}.
\]

We report macro-averaged Precision/Recall/F1 to weight all three
regimes equally, consistent with López de Prado (2018, AFML, §9).
