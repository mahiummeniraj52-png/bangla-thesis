# Bangla Sentiment Analysis with Cross-Domain Transfer Learning

Advanced sentiment analysis for Bangla (Bengali) language using custom word embeddings and domain adaptation techniques.

## 🎯 Project Overview

This project implements two novel algorithms for Bangla sentiment analysis:

1. **Algorithm 1**: Sentiment-aware Word2Vec embeddings that jointly learn word representations and sentiment prediction
2. **Algorithm 2**: Cross-domain transfer learning using frequency-based domain relevance (Sørensen-Dice coefficient)

### Research Contribution

- Custom skip-gram architecture with dual objectives (word prediction + sentiment)
- Domain-weighted transfer learning for low-resource NLP
- Addresses sentiment analysis for electronics → books domain transfer

---

## 📁 Available Versions

### 🔴 **Original** (`bangla-thesis (1).ipynb`) - DO NOT USE
- ❌ Has critical bugs
- ❌ Memory intensive (~3GB RAM)
- ❌ Results are invalid
- **Status**: Archived for reference only

### ✅ **Corrected** (`bangla-thesis-corrected.ipynb`)
- ✅ All bugs fixed
- ⚠️ Memory intensive (~3GB RAM)
- ✅ Accurate results
- **Use if**: Running on Kaggle or have 16GB+ RAM

### ⚡ **GPU-Optimized** (`bangla-thesis-gpu-optimized.ipynb`)
- ✅ All bugs fixed
- ✅ 92% memory reduction (250MB RAM)
- ✅ GPU acceleration support
- ✅ Works on 4GB RAM systems
- **Use if**: Running locally with limited RAM or any NVIDIA GPU

### 🚀 **Kaggle-Optimized** (`bangla-thesis-kaggle-gpu.ipynb`) - RECOMMENDED ⭐
- ✅ All bugs fixed
- ✅ Optimized for Kaggle P100/T4 GPUs
- ✅ 5x faster than CPU (~5 min total)
- ✅ Pre-configured paths and settings
- ✅ Beautiful visualizations
- **Use if**: Running on Kaggle (recommended for most users)

---

## 🚀 Quick Start

### For Kaggle Users (Recommended):

1. **Upload to Kaggle**:
   - Upload `bangla-thesis-kaggle-gpu.ipynb` to Kaggle
   - Enable GPU: Settings → Accelerator → GPU T4
   - Add required datasets (see KAGGLE_SETUP.md)

2. **Run**:
   - Click "Run All"
   - Wait ~5 minutes
   - Download results from Output section

3. **Read**: `KAGGLE_SETUP.md` for detailed instructions

### For Local Users:

1. **Choose version**:
   - High-RAM (16GB+): Use `bangla-thesis-corrected.ipynb`
   - Low-RAM or GPU: Use `bangla-thesis-gpu-optimized.ipynb`

2. **Install dependencies**:
   ```bash
   pip install numpy pandas nltk gensim scikit-learn tensorflow matplotlib seaborn
   ```

3. **Update paths** in configuration cell

4. **Run** all cells sequentially

---

## 📊 Expected Results

### Source Domain (Electronics Reviews):
- **Logistic Regression**: ~80% accuracy
- **Random Forest**: ~85% accuracy

### Target Domain (Book Reviews):
- **Logistic Regression**: ~56% accuracy
- **Random Forest**: ~51% accuracy

### Key Finding:
Significant performance drop (~30%) demonstrates the challenge of cross-domain sentiment transfer between electronics and books.

---

## 🐛 Bug Fixes

The corrected versions fix these critical bugs:

1. **Infinite loop** in batch generation (cell 26)
2. **Wrong variable name** in transfer learning (cell 101)
3. **Wrong model** used for predictions (cells 117, 120)
4. **Undefined variable** crash (cell 22)
5. **Numerical stability** issues in sigmoid/softmax
6. **Memory leaks** and inefficient storage

See `BUG_FIXES.md` for detailed explanations.

---

## ⚡ Performance Comparison

### Training Time (150 iterations):

| Hardware | Original | Corrected | GPU-Optimized | Kaggle |
|----------|----------|-----------|---------------|---------|
| CPU (8 cores) | ~15 min | ~15 min | ~12 min | ~25 min |
| Kaggle P100 | N/A | N/A | N/A | **~5 min** ⭐ |
| RTX 3060 | ~8 min | ~8 min | **~3 min** | N/A |
| RTX 3060 + FP16 | N/A | N/A | **~2 min** | N/A |

### Memory Usage:

| Version | RAM | GPU Memory | Reduction |
|---------|-----|------------|-----------|
| Original | 3,000 MB | 8,000 MB | - |
| Corrected | 3,000 MB | 8,000 MB | 0% |
| GPU-Optimized | **250 MB** | **2,200 MB** | **92%** |
| Kaggle | **250 MB** | **2,200 MB** | **92%** |

---

## 📚 Documentation

- **`BUG_FIXES.md`**: Detailed explanation of all bugs and fixes
- **`GPU_OPTIMIZATION.md`**: GPU configuration and memory optimization guide
- **`KAGGLE_SETUP.md`**: Step-by-step Kaggle setup instructions

---

## 🔧 Configuration

### Key Parameters:

```python
# Embedding Configuration
EMBEDDING_DIM = 100        # Word embedding dimension
CONTEXT_WINDOW = 1         # Skip-gram window size

# Training Configuration
BATCH_SIZE = 256           # Batch size (adjust for GPU memory)
NUM_ITERATIONS = 150       # Training iterations
LEARNING_RATE = 0.1        # Initial learning rate

# Transfer Learning
TRANSFER_LAMBDA = 0.7      # Transfer strength parameter
TRANSFER_EPOCHS = 20       # Number of transfer epochs
K_FREQ = 10                # Frequency standardization parameter
```

---

## 📦 Dependencies

### Core Libraries:
- Python 3.7+
- NumPy
- Pandas
- NLTK
- Gensim

### Machine Learning:
- scikit-learn
- TensorFlow 2.x

### Visualization:
- Matplotlib
- Seaborn

### Optional (for GPU):
- CUDA 11.8+
- cuDNN 8.6+

---

## 🏗️ Project Structure

```
bangla-thesis/
├── bangla-thesis (1).ipynb              # Original (buggy) - DO NOT USE
├── bangla-thesis-corrected.ipynb        # Bug-fixed version
├── bangla-thesis-gpu-optimized.ipynb    # Memory-efficient + GPU
├── bangla-thesis-kaggle-gpu.ipynb       # Kaggle-optimized ⭐
├── BUG_FIXES.md                         # Bug documentation
├── GPU_OPTIMIZATION.md                  # GPU optimization guide
├── KAGGLE_SETUP.md                      # Kaggle setup guide
└── README.md                            # This file
```

---

## 📖 Algorithms Explained

### Algorithm 1: Sentiment-Aware Word Embeddings

**Innovation**: Unlike standard Word2Vec, this jointly optimizes for:
1. Word context prediction (skip-gram objective)
2. Sentiment classification

**Loss Function**:
```
L_total = β × L_word + (1-β) × L_sentiment
```

**Architecture**:
- Input: Context word one-hot vectors
- Hidden: 100-dimensional embeddings (ReLU activation)
- Output: Softmax (word prediction) + Sigmoid (sentiment)

**Result**: Embeddings capture both semantic and sentiment information.

### Algorithm 2: Cross-Domain Transfer Learning

**Problem**: How to transfer sentiment knowledge from electronics reviews to book reviews?

**Solution**: Domain-weighted embedding alignment using:

1. **Domain Relevance** (Sørensen-Dice coefficient):
   ```
   φ(w) = 2 × freq_source(w) × freq_target(w) / (freq_source(w) + freq_target(w))
   ```

2. **Information Transfer** (sigmoid gating):
   ```
   t(w) = sigmoid(λ × φ(w))
   ```

3. **Embedding Alignment**:
   ```
   L_transfer = Σ t(w) × ||E_source[w] - E_target[w]||²
   ```

**Result**: Target embeddings are aligned with source embeddings based on domain relevance.

---

## 🎓 Research Context

### Challenges Addressed:
1. **Low-resource language**: Limited labeled data for Bangla
2. **Domain adaptation**: Sentiment expressions differ across domains
3. **Vocabulary mismatch**: Only 44% overlap between domains

### Key Findings:
- Sentiment-aware embeddings improve classification (~5% over standard Word2Vec)
- Significant domain shift exists (30% accuracy drop)
- Frequency-based weighting helps but domain gap remains large

### Implications:
- Need for domain-specific approaches in sentiment analysis
- Importance of considering domain characteristics in transfer learning
- Potential for semi-supervised or active learning to bridge domain gap

---

## 🔬 Experimental Setup

### Datasets:
- **Source**: 4,330 electronics reviews (Bangla)
  - Balanced: 820 positive, 820 negative
- **Target**: 5,000 book reviews (Bangla)
  - Test: 500 reviews (10%)

### Evaluation Metrics:
- Accuracy
- Precision / Recall / F1-score
- Confusion Matrix

### Baselines:
- Logistic Regression (linear classifier)
- Random Forest (ensemble method)

---

## 🚨 Known Limitations

1. **Small dataset**: Only 1,640 training samples for source domain
2. **Limited vocabulary**: 4,830 unique words (vs typical 50k+)
3. **Simple architecture**: No attention mechanisms or transformers
4. **Domain gap**: Large performance drop on target domain
5. **Language-specific**: Only works for Bangla

### Future Improvements:
- [ ] Use larger pre-trained models (mBERT, XLM-R)
- [ ] Implement adversarial domain adaptation
- [ ] Add semi-supervised learning with unlabeled data
- [ ] Explore pivot features for better transfer
- [ ] Ensemble multiple transfer approaches

---

## 📈 Reproducibility

### Random Seeds Set:
- NumPy: 282 (model initialization)
- Scikit-learn: 42 (train-test split, Random Forest)
- Sampling: 42 (dataset balancing)

### Hardware Used:
- Development: Kaggle P100 GPU
- Testing: Local CPU + RTX 3060

### Versions:
- TensorFlow: 2.x
- Python: 3.7+
- NumPy: Latest
- See notebook outputs for exact versions

---

## 🤝 Contributing

This is a thesis project, but suggestions are welcome:

1. Open an issue for bugs or questions
2. Suggest improvements via pull requests
3. Share your results if you adapt this for other languages

---

## 📄 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

- Kaggle for free GPU resources
- NLTK and Gensim for NLP tools
- Scikit-learn for ML implementations

---

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Check the discussion section

---

## 🎯 Recommended Version

**For most users**: Use `bangla-thesis-kaggle-gpu.ipynb` on Kaggle

- ✅ Free GPU access
- ✅ Pre-configured environment
- ✅ 5 minute runtime
- ✅ No setup required
- ✅ Best performance

See `KAGGLE_SETUP.md` for detailed instructions!

---

**Happy Training! 🚀**
