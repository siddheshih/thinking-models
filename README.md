# Analyzing Thinking Models

## Overview
This work explores the role of thinking traces in reasoning models, with experiments designed to understand how faithfully these traces represent the model's internal reasoning process. The ultimate goal is to optimize thinking budgets without compromising performance.

## Experiments
Our experiments use the AIME 2024 mathematical olympiad dataset (100-200 examples) and analyze the relationship between thinking traces and model performance:

### 1. Pseudo-thinking Traces
Replaced meaningful thinking with asterisks (*****) to test dependence on content.
- **Finding**: Models maintain accuracy with shorter pseudo-thinking but collapse when traces become excessively long.
- **Implication**: Model reliance on thinking traces appears somewhat superficial rather than genuinely faithful.

### 2. Prompt Injection
Injected random segments of the model's own thinking process back into input prompts.
- **Finding**: Accuracy remained stable while thinking length decreased with more injected content.
- **Implication**: Reasoning process is sensitive to injected hints even while maintaining performance.

### 3. Forced Thinking Termination
Forced models to stop thinking by inserting the `</think>` token after varying numbers of steps.
- **Finding**: Early termination drops performance; longer thinking correlates with higher accuracy.
- **Implication**: Chain-of-thought is actively used for reasoning, establishing evidence for trace faithfulness.

### 4. Thinking Trace Corruption
Corrupted thinking traces by replacing keywords with random tokens or modifying numbers.
- **Finding**: Both corruptions reduced accuracy as replacement fractions increased, with number corruption showing more direct decline.
- **Implication**: Models attempt to compensate for corrupted information by adjusting thinking patterns.

## Conclusions
These experiments demonstrate that while thinking traces are functionally important for reasoning, they exist in a complex relationship with internal model processes. Sufficient chain-of-thought tokens are crucial for optimal performance, and models show adaptive behavior when facing corrupted or injected information.

## Future Directions
- Exploring internal representations using Sparse Autoencoders (SAEs)
- Developing dynamic thinking budget allocation algorithms
- Implementing calibrated confidence in thinking traces
- Designing more robust thinking mechanisms against adversarial perturbations

## Contact
[Siddhesh Pawar](mailto:siddheshpawar1999@gmail.com)
