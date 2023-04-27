# ML for Moonboard

    Work in Progress

## Introduction

This project aims at classifying the difficulty grade of Moonboard climbing routes by using machine learning techniques. Similar projects, such as [moonGen](https://github.com/gestalt-howard/moonGen/) or [Recurrent Neural Network for MoonBoard Climbing Route Classification and Generation](https://arxiv.org/abs/2102.01788), proposed that inspiration can be drawn from the NLP field.

Following this idea, I applied different ideas from NLP to Moonboard Classification. Among other things, I applied the ideas behind Word2Vec to build vector embeddings for each one of the holds on the moonboard. These hold embeddings can be combined into a route embeddings using techniques such as weighted-sum pooling.

### Data
The Datasets used comes from [andrew-houghton/moon-board-climbing](https://github.com/andrew-houghton/moon-board-climbing).


### Current State
 
 **Finished:**
- Basic Data Exploration and Visualization
- Basic Route Embeddings, such as BagOfWords (similar to a Bitmap) and tf-idf weighted BagOfWords. 
- "Hold2Vec" embeddings for the climbing holds.
- Pooled Route Emebeddings from Hold embeddings, such as Sum-Pooling, Average-Pooling, and Weighted Sum-Pooling.

**future items:**

- Web Application
- Baseline Models
- Improving Embeddings:
  - Sif Embeddings ([Paper](https://openreview.net/pdf?id=SyK00v5xx), [Code](https://github.com/PrincetonML/SIF/tree/master/examples))
  - Doc2Vec
  - Embeddings trained on Co-occurence Matrix, such as GloVe
- Improving Classifier:
  - Classifier Models such as RNNs (LSTM, GRU) or Transformers
  - (Convolutional) Graph Classifier
  - Balance out individual weaknesses of equally well performing models using [Voting Classifiers](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier) or [Stacked Generalization](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization).
- Route Generation
- Route Adaptation: making routes harder or easier by removing a hold.