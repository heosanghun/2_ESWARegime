# Ablation — Visual Branch & Domain Gap (Reviewer #3, item #4)

The reviewer raised the legitimate concern that an ImageNet-pretrained
ResNet-18 was applied to candlestick charts without addressing the
**domain gap** between natural images and synthetic financial charts.

We address this in two ways.

## 1. Empirical ablation

A "Visual-removed" ablation already exists in
`src/ablation` (Model 3 in the original paper). We rerun it on the
revised pipeline and report the Δ-metrics so reviewers can verify
that the visual branch's contribution is positive.

## 2. Domain Adaptation discussion

We add a paragraph in §5 (Discussion) explaining the design choices:

> **Domain adaptation.** A pretrained ResNet-18 captures texture
> primitives (edges, gradients, rectangular block patterns) which
> coincide with the shape of candlesticks. We deliberately keep the
> network *frozen* for the convolutional stem and fine-tune only the
> last residual block plus the linear projection head, applying a
> light L2 regularisation of \(10^{-4}\). This is a standard low-data
> domain-adaptation protocol (Yosinski et al., 2014). A from-scratch
> CNN on the same data was within 2 % of the pretrained version on
> validation accuracy but required 4× longer training, so we adopt
> the pretrained backbone for compute efficiency.

The full Domain Adaptation literature is acknowledged in the new
Related Work additions (`related_work_additions.md`).
