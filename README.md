<p align="center">
    <img src="Big Five Framework.jpg" height="300">
</p>

## Overview
This project proposes a novel multi-modal personality analysis framework that addresses these challenges by synchronizing and integrating features from multiple modalities and enhancing model generalization through domain adaptation. We introduce a timestamp-based modality alignment mechanism that synchronizes data based on spoken word timestamps, ensuring accurate correspondence across modalities and facilitating effective feature integration. To capture temporal dependencies and inter-modal interactions, we employ Bidirectional Long Short-Term Memory networks and self-attention mechanisms, allowing the model to focus on the most informative features for personality prediction. Furthermore, we develop a gradient-based domain adaptation method that transfers knowledge from multiple source domains to improve performance in target domains with scarce labeled data.
![Image]()

### Key Contributions
Our work contributes to the following aspects:
1. We propose an effective multi-modal personality analysis framework that effectively integrates facial expressions, audio signals, textual content, and background information from short videos for personality prediction.
2. We introduce a semantic unit modality alignment mechanism that synchronizes multi-modal data based on spoken word timestamps, ensuring accurate correspondence across modalities and enhancing feature representation.
3. We develop a gradient-based domain adaptation method that transfers knowledge from multiple source domains to target domains with limited labeled data, enhancing model generalization and performance in few-shot learning scenarios.
4.  We validate the effectiveness of our proposed framework through extensive experiments on real-world datasets, demonstrating significant improvements over existing methods in personality prediction tasks.

## [Datasets]

First Impressions dataset created by Biel and Gatica-Perez from the 2016 ChaLearn competition is used in this research.

## Settings

To run the code, simply clone the repository and install the required packages:

```bash
git clone https://github.com/ZhiyaoShu/LLM-HGNN-MBTI.git
cd LLM-HGNN-MBTI
pip install -r requirements.txt
```

```python

## [Training]

## You can Cite Our Work:
