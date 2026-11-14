# Characteristics of the omics layers in TCGA-BRCA
Tags: RNA-seq, mRNA, miRNA-seq, RPPA, protein, measurement noise, missing data, heterogeneity, platform

- **mRNA (RNA-seq)**: genome-wide, high dynamic range; usually the most informative layer for intrinsic subtype (the subtypes were defined from mRNA).
- **miRNA (miRNA-seq)**: ~1-2k mature miRNAs, many lowly expressed (noisier); captures post-transcriptional regulation.
- **Protein (RPPA, reverse-phase protein array)**: ~150-200 targeted antibodies including phospho-proteins; directly measures functional pathway activity but has a limited panel, antibody-specific noise, and is often unavailable for a subset of samples.

Heterogeneity across layers (different noise levels, coverage, missing samples) is a key challenge for multi-omics integration and for multi-agent systems.

Source: TCGA Network 2012; Li et al., Nat Methods 2013 (TCPA RPPA).
