# Multi-omics integration strategies
Tags: integration, fusion, early fusion, late fusion, intermediate, DIABLO, MOFA, uncertainty weighting

- **Early fusion (concatenation)**: join features of all layers into one matrix. Simple, but layers with more features or more noise can dominate, and every sample needs every layer.
- **Late fusion (decision level)**: train one model per layer and combine predictions (voting/averaging). Naturally handles samples with missing layers. Weighting each layer by its confidence/uncertainty reduces the influence of noisy layers.
- **Intermediate fusion**: learn a joint latent space (e.g. DIABLO in mixOmics, MOFA).

The mixOmics breast.TCGA dataset used here was introduced to demonstrate DIABLO (Singh et al., Bioinformatics 2019).

Source: Singh et al. 2019 (doi:10.1093/bioinformatics/bty1054); Argelaguet et al., Mol Syst Biol 2018 (MOFA).
