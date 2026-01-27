HIGH LEVEL SUMMARY
Paper Summary
The paper "SubspaceBank: Federated Transfer Learning For LLMs Using Subspaces" addresses the challenges of Federated Transfer Learning (FTL) for Large Language Models (LLMs) in a heterogeneous, "dataset-pure" regime, where clients correspond to distinct tasks. The authors identify interference during naive aggregation (e.g., FedAvg) as a primary failure mode. They introduce SubspaceBank, a geometry-aware method for LoRA-based adaptation. SubspaceBank decomposes client updates into shared, private, and residual components using a residual-first subspace decomposition, then applies structured aggregation to amplify shared directions and shrink residuals. Experiments on LLaMA2-7B across five benchmarks indicate that SubspaceBank outperforms FedAvg in global robustness and specialization retention.

Key Issues Roadmap
[2. SubspaceBank Methodology]: Analysis of the computational overhead (server-side SVDs) is missing, and the justification for applying subspace decomposition to the inherently low-dimensional right space of LoRA B matrices requires clarification.
[3. Experimental Evaluation and Analysis]: The evaluation lacks reporting of statistical significance (e.g., variance across multiple runs), and the Multi-Client Datasets regime configuration is underspecified.
[4. Discussion, Related Work, and Context]: The appendices still omit essential reproducibility details (e.g., hyperparameters in Appendix A) and theoretical analysis (Appendix B) referenced in the text.
DETAILED SEGMENT REVIEWS
[1] SEGMENT: 1. Introduction, Motivation, and Background
PAGES: [[1, 2]]
Summary: The segment (Sections 1 and 2) introduces and provides background for the paper. Section 1 defines Federated Transfer Fine-tuning (FTL) as a regime where clients correspond to distinct tasks or domains (dataset-pure), leading to intrinsic heterogeneity. This is contrasted with standard Federated Learning (FL) assumptions. The authors identify "interference" during aggregation (e.g., FedAvg) as the primary challenge in FTL. They introduce SubspaceBank, a geometry-aware approach for LoRA-based FTL designed to mitigate this interference through subspace decomposition and structured aggregation. Section 2 provides background on the constraints of FL for LLMs, motivating the use of Parameter-Efficient Fine-Tuning (PEFT) like LoRA, and further details the distinction between classical FL heterogeneity and the structural heterogeneity of FTL.

Potential Mistakes and Improvements:

Precision of Novelty Claim (Section 1): On lines 051-052, the authors state: "there are no prior works that directly study federated transfer learning for LLMs." While the authors provide a specific definition of FTL (L048-051) and further differentiate their work from related areas like Federated Multi-Task Learning (FMTL) in Section 6, this initial claim in the introduction is strong. To prevent potential misinterpretation regarding the existing FMTL literature, it may be beneficial to qualify this statement earlier, perhaps by emphasizing the specific focus on the dataset-pure regime for LLMs or incorporating the nuance mentioned later in L097-098 ("remains underexplored for LLM fine-tuning").
Minor Corrections and Typos:

L045: "around a a single global objective" -> "around a single global objective"
[2] SEGMENT: 2. SubspaceBank Methodology
PAGES: [[3, 5]]
1. Summary Section 3 describes SubspaceBank, a methodology for Federated Transfer Learning (FTL) applied to LoRA adapters. The method operates independently on the matrix-valued updates (deltas) of LoRA A and B matrices. It consists of a residual-first decomposition with structured aggregation at the server.

The server mechanism (Sections 3.4-3.6) employs a "residual-first" approach to decompose client updates. A shared basis is identified via SVD of stacked client deltas. A private basis is then extracted from the residual after removing the shared component, resulting in a shared/private/residual decomposition. Structured aggregation (Eq. 9) recombines these by amplifying the shared component and shrinking the residual component.

2. Potential Mistakes and Improvements

Design Rationale: Justification for Analyzing the Right Subspace of LoRA B Matrices (Section 3.2). The methodology specifies using the right singular vectors (input space [TeX source not found]) for all banked matrices (L119-121). For LoRA A matrices, [TeX source not found] is large. However, for LoRA B matrices, the input dimension [TeX source not found] is the LoRA rank [TeX source not found], which is typically very small (e.g., 8 or 16). The paper should clarify the justification for performing subspace decomposition (identifying shared [TeX source not found] and private [TeX source not found] components) within this already highly constrained, low-dimensional space [TeX source not found].

Operational Properties: Lack of Computational Overhead Analysis (Sections 3.4, 3.5). SubspaceBank introduces significant server-side computation compared to FedAvg. Each round requires, for every banked key, a truncated SVD on the large stacked matrix [TeX source not found] (dimensions [TeX source not found]) and [TeX source not found] SVDs on the residuals [TeX source not found]. The methodology section should include a discussion of this computational complexity to assess the method's practical scalability.

Clarity and Reproducibility: Specification of Subspace Ranks (Sections 3.4, 3.5). The method relies on the shared rank [TeX source not found] (Eq. 3) and the private rank [TeX source not found] (L176). Section 3 does not specify how these ranks are determined. For reproducibility, it should be clarified whether these are fixed hyperparameters or if they are determined adaptively (e.g., based on singular value thresholds).

Clarity: Redundancy in Orthogonalization Step (Section 3.5, Eq. 6). The candidate private basis [TeX source not found] is defined by the right singular vectors of the residual $R^{t,(0)}{k,i}[TeX source not found]R^{t,(0)}{k,i}[TeX source not found]S^t_k[TeX source not found](I - \Pi_{S^t_k})\hat{P}^t_{k,i}$ in Eq. 6 appears mathematically redundant. If this step is included to ensure orthogonality despite potential numerical precision issues, this motivation should be clarified.
[3] SEGMENT: 3. Experimental Evaluation and Analysis
PAGES: [[5, 7]]
1. Summary Section 4 presents the experimental evaluation of SubspaceBank for Federated Transfer Learning (FTL). The authors establish an evaluation protocol using LLaMA2-7B fine-tuned with LoRA across five diverse benchmarks (GSM8k, HellaSwag, XSum, HotPotQA, MBPP). They define two "dataset-pure" regimes: Single-Client Datasets and Multi-Client Datasets. SubspaceBank is compared against FedAvg and centralized training baselines. The evaluation assesses global robustness (Tables 1 and 2) and specialization retention (Table 3). The results indicate that SubspaceBank outperforms FedAvg on both metrics across both regimes. Section 4.6 provides geometry diagnostics analyzing the energy fractions of the update components (Figure 1).

2. Potential Mistakes and Improvements

Missing Hyperparameters and Reproducibility Details (Clarity/Reproducibility): The paper lacks crucial details necessary for reproducibility and assessing the fairness of comparisons. L250-251 state that hyperparameters are in the Appendix, but Appendix A (Page 11) consists of placeholders. Key missing information includes:
SubspaceBank-specific hyperparameters ([TeX source not found]).
Standard training configurations (learning rate, optimizer details, number of local epochs [TeX source not found]).
The LoRA rank ([TeX source not found]), which is needed to verify the claim of "comparable LoRA parameterization" (L031, L063).
Lack of Statistical Significance Reporting (Validity): Tables 1, 2, and 3 report only point estimates of performance. Given the potential variance in LLM fine-tuning and the instability of federated learning under heterogeneity, reporting measures of variance (e.g., standard deviation across multiple runs) is necessary to establish the statistical significance of the improvements over FedAvg and to substantiate claims of "improved stability" (L062).
Underspecified Multi-Client Regime Configuration (Clarity): The configuration for the Multi-Client Datasets regime (Table 2) is underspecified. Essential details such as the total number of clients (N), the data sharding strategy, and the specific client participation rate are missing. The caption for Table 2 (L284) also notes "(placeholders to be filled)," making the experimental setting ambiguous.
Unquantified Overhead (Efficiency/Trade-offs): SubspaceBank introduces overhead compared to FedAvg, including computational costs at the server (SVD computations). The experimental section does not quantify this overhead, which is necessary for evaluating the practical trade-offs of the method, particularly since communication is cited as a major bottleneck in FL for LLMs (L107).
Incomplete and Potentially Inconsistent Geometry Diagnostics (Clarity/Validity):
The analysis in Section 4.6 states that the "residual fraction is effectively zero throughout" (L383, Figure 1). This observation seems inconsistent with the motivation that suppressing residuals (via [TeX source not found], Section 3.6) is important for stability (L202-204). If the residual component is negligible, the mechanism of shrinking it would have minimal impact. This requires clarification.
3. Minor Corrections and Typos

L284: The caption for Table 2 contains the text "(placeholders to be filled)".
[4] SEGMENT: 4. Discussion, Related Work, and Context
PAGES: [[7, 11]]
Summary: The reviewed segment (Pages 7-11) includes Section 5 (Discussion and Limitations), Section 6 (Related Work), Section 7 (Conclusion), the Impact Statement, References, and the Appendix (A-C). Section 5 interprets the empirical results, suggesting that SubspaceBank successfully mitigates interference in Federated Transfer Learning (FTL) by separating shared and private directions. It also discusses limitations, such as the method's sensitivity to partial participation during basis estimation. Section 6 contextualizes the work relative to existing literature on heterogeneous Federated Learning (FL), FL for LLMs using PEFT, and centralized interference control methods. The Appendix still includes placeholders for reproducibility details and proofs.

Potential Mistakes and Improvements:

Completeness (Reproducibility Details): The paper states in Section 4.2 (L252-253) that "Detailed hyperparameters, hardware, and preprocessing are located in the Appendix." However, Appendix A (Page 11) contains only placeholders (L550-562). The absence of key hyperparameters for SubspaceBank ([TeX source not found]) and the baseline training setup (e.g., LoRA rank [TeX source not found], learning rates, optimizer details, local epochs [TeX source not found]) prevents the reproduction and verification of the experimental results presented in Section 4.
Completeness (Theoretical Details): Appendix B lists placeholders for "Proofs and Extended Theory Details" (L564), including operator interpretation and assumptions/limits under partial participation and finite-sample basis estimation. The absence of this analysis leaves the theoretical grounding of the method incomplete.
Clarity (Limitations on LoRA B): Section 5 (L390-391) identifies a potential limitation in regimes with "extremely small right dimension for key parameters (notably LoRA B)". Given that SubspaceBank operates on the right space (Section 3.2), and the right dimension of LoRA B is the rank [TeX source not found] (which is typically small by design), the specific circumstances under which this becomes a limitation are unclear. Clarification on how this small dimensionality negatively impacts the subspace estimation would strengthen the discussion.
Minor Corrections and Typos:

L419: The citation formatting "T Dinh et al. (2020)" appears incorrect.



