# Kaggle GPU Setup Guide

## 🚀 Quick Start (5 Minutes)

### Step 1: Create New Notebook
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. You'll see a blank notebook

### Step 2: Enable GPU
1. Click **Settings** (gear icon on right sidebar)
2. Under **Accelerator**, select:
   - **GPU T4 x2** (recommended - free tier) or
   - **GPU P100** (if available)
3. Click **Save**

### Step 3: Add Datasets
1. Click **Add Data** button (right sidebar)
2. Search for and add these datasets:
   - `bangla-electronics-lemmatized-final-1-csv`
   - `bangla-book-lemmatized-18002-csv`
3. If datasets don't exist, upload your CSV files:
   - Click **Upload**
   - Drag and drop your CSV files
   - Make sure they're named correctly

### Step 4: Upload Notebook
1. Click **File** → **Upload Notebook**
2. Select `bangla-thesis-kaggle-gpu.ipynb`
3. Wait for upload to complete

### Step 5: Run!
1. Click **Run All** button
2. Sit back and relax ☕
3. Should complete in ~3-5 minutes

---

## 📋 Expected Runtime

### With Kaggle GPU (T4/P100):
- **Algorithm 1 Training**: ~2-3 minutes
- **Algorithm 2 Transfer**: ~30 seconds
- **Classification**: ~1 minute
- **Total**: ~5 minutes ⚡

### Without GPU (CPU only):
- **Algorithm 1 Training**: ~15 minutes
- **Algorithm 2 Transfer**: ~5 minutes
- **Classification**: ~2 minutes
- **Total**: ~25 minutes 🐌

**👉 Always use GPU!** It's 5x faster and free on Kaggle.

---

## 🎮 Kaggle GPU Specifications

### GPU T4 (Free Tier):
- **VRAM**: 16GB
- **CUDA Cores**: 2,560
- **Tensor Cores**: 320
- **FP16 Performance**: Excellent
- **Availability**: Always available
- **Best for**: This notebook ✅

### GPU P100 (Sometimes Available):
- **VRAM**: 16GB
- **CUDA Cores**: 3,584
- **FP16 Performance**: Very Good
- **Availability**: Limited
- **Best for**: Large-scale training

**For this notebook**: Both T4 and P100 work great. T4 is recommended because it's always available.

---

## 📁 File Paths on Kaggle

Kaggle automatically mounts datasets to `/kaggle/input/`. The notebook is pre-configured with these paths:

```python
ELECTRONICS_FILE = "/kaggle/input/bangla-electronics-lemmatized-final-1-csv/bangla_electronics_lemmatized_final.csv"
BOOKS_FILE = "/kaggle/input/bangla-book-lemmatized-18002-csv/bangla_book_lemmatized_18002.csv"
```

**Important**: The folder name is based on your dataset name. If you uploaded datasets with different names, update these paths in the first config cell.

### How to Find Your Paths:
1. Look at the **Data** section in right sidebar
2. Click on dataset name
3. Copy the path shown
4. Paste into notebook config cell

---

## 🔧 Configuration Options

The notebook is pre-configured with optimal settings for Kaggle:

```python
# Current Settings (Optimized for Kaggle)
BATCH_SIZE = 256              # Large batches for GPU
TRANSFER_BATCH_SIZE = 512     # Fast transfer learning
USE_MIXED_PRECISION = True    # FP16 for 2x speedup
ENABLE_XLA = True             # Extra optimization
```

### If You Get Memory Errors:

**Reduce batch sizes:**
```python
BATCH_SIZE = 128              # Instead of 256
TRANSFER_BATCH_SIZE = 256     # Instead of 512
```

**Disable mixed precision:**
```python
USE_MIXED_PRECISION = False   # Use FP32 instead
```

---

## 💾 Saving Results

### Output Files Created:
1. `loss_algo1_kaggle.png` - Training loss plot
2. `loss_algo2_kaggle.png` - Transfer learning loss
3. `confusion_matrix_source_kaggle.png` - Source domain results
4. `confusion_matrix_target_kaggle.png` - Target domain results

### How to Download:
1. After notebook finishes, scroll to **Output** section (right sidebar)
2. Click on each PNG file
3. Click **Download**

**Or download all at once:**
- Click **Save Version** (top right)
- Select **Save & Run All**
- After completion, go to **Versions** tab
- Click on your version → **Output** → Download all files

---

## 🐛 Troubleshooting

### Issue 1: "No GPU Detected"

**Error message:**
```
⚠️ WARNING: No GPU detected!
Make sure to enable GPU in Kaggle:
Settings → Accelerator → GPU T4 or P100
```

**Solution:**
1. Click **Settings** (right sidebar)
2. Under **Accelerator**, select **GPU T4 x2**
3. Click **Save**
4. Restart notebook: **Run** → **Restart & Run All**

---

### Issue 2: "File Not Found"

**Error message:**
```
FileNotFoundError: /kaggle/input/.../bangla_electronics_lemmatized_final.csv
```

**Solution:**
1. Check if datasets are added (look at **Data** section)
2. If not, click **Add Data** and add required datasets
3. Verify file paths match your dataset names
4. Update paths in config cell if needed

---

### Issue 3: "Out of Memory"

**Error message:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solution (rare on Kaggle with 16GB GPU, but just in case):**
```python
# Reduce batch sizes in config cell
BATCH_SIZE = 64
TRANSFER_BATCH_SIZE = 128
USE_MIXED_PRECISION = False
```

Then restart: **Run** → **Restart & Run All**

---

### Issue 4: Notebook Runs Slow

**Check:**
1. ✅ GPU is enabled (Settings → Accelerator)
2. ✅ You see "GPU configured" message in output
3. ✅ Mixed precision is enabled

**If still slow:**
- Check GPU utilization: Kaggle shows GPU usage in bottom bar
- Make sure you're not running multiple notebooks simultaneously
- Try switching to a different GPU (P100 ↔ T4)

---

## 📊 Expected Results

### Source Domain (Electronics):
- **Logistic Regression**: ~79-80% accuracy
- **Random Forest**: ~84-85% accuracy ⭐

### Target Domain (Books):
- **Logistic Regression**: ~56-60% accuracy
- **Random Forest**: ~50-55% accuracy

### Transfer Learning Drop:
- **Expected drop**: 25-35%
- **Reason**: Domain shift between electronics and books
- **This is normal!** Shows need for domain adaptation

---

## 🎯 Performance Tips

### 1. Use Internet (Optional)
If you need to install additional packages:
- Settings → Internet → **On**

### 2. Save Versions Regularly
- Click **Save Version** every 30 minutes
- Kaggle auto-saves, but manual saves are safer

### 3. Monitor GPU Usage
- Check bottom status bar for GPU utilization
- Should show 80-100% during training
- If <50%, GPU might not be utilized properly

### 4. Check Logs
- Click on each cell output to expand
- Look for "✅ GPU Configuration Complete"
- Verify "Mixed precision enabled" message

---

## 📈 Comparison: Kaggle vs Local

| Feature | Kaggle GPU | Local (No GPU) | Local (4GB GPU) |
|---------|------------|----------------|-----------------|
| **Cost** | FREE ✅ | FREE ✅ | Hardware cost |
| **Setup** | 2 minutes | 5 minutes | 30 minutes |
| **Runtime** | ~5 min ⚡ | ~25 min 🐌 | ~8 min |
| **Memory** | 16GB VRAM | N/A | 4GB (limited) |
| **Reliability** | High | Medium | Depends |
| **Convenience** | Very High | Medium | Medium |

**Verdict**: For this project, **Kaggle is the best option** unless you have a powerful local GPU (8GB+ VRAM).

---

## 🔄 Iterating and Experimenting

### To Change Parameters:

1. **Modify config cell** (first code cell):
   ```python
   EMBEDDING_DIM = 200  # Try different sizes
   BATCH_SIZE = 512     # Experiment with batches
   TRANSFER_EPOCHS = 40 # More epochs
   ```

2. **Run** → **Restart & Run All**

3. **Compare results** with previous version

### Parameters to Experiment With:

| Parameter | Default | Try | Impact |
|-----------|---------|-----|--------|
| `EMBEDDING_DIM` | 100 | 50, 200, 300 | Quality vs speed |
| `BATCH_SIZE` | 256 | 128, 512 | Speed (bigger = faster) |
| `LEARNING_RATE` | 0.1 | 0.05, 0.2 | Convergence |
| `TRANSFER_EPOCHS` | 20 | 10, 40 | Transfer quality |
| `TRANSFER_LAMBDA` | 0.7 | 0.3, 1.0 | Transfer strength |

---

## 📝 Sharing Your Results

### Make Notebook Public:
1. Click **Share** (top right)
2. Select **Public**
3. Add description and tags
4. Click **Share**

### Add to Portfolio:
- Copy the public notebook URL
- Share on LinkedIn, GitHub, etc.
- Include confusion matrices in presentations

---

## ⚡ Pro Tips

1. **Enable Auto-Save**: Kaggle auto-saves, but check Settings → Auto-save
2. **Use Comments**: Add markdown cells to document your findings
3. **Version Control**: Save versions before major changes
4. **GPU Quota**: Kaggle gives ~30 hours/week GPU time - plenty for this project
5. **Schedule Runs**: You can schedule notebooks to run automatically
6. **Compare Versions**: Use Kaggle's version comparison feature

---

## 🆘 Getting Help

### If You're Stuck:

1. **Check cell output**: Error messages are usually helpful
2. **Read troubleshooting section** above
3. **Check Kaggle forums**: https://www.kaggle.com/discussions
4. **Ask in comments**: Enable discussions on your notebook

### Common Questions:

**Q: Can I run this without GPU?**
A: Yes, but it will take 5x longer (~25 minutes vs 5 minutes).

**Q: How much does Kaggle GPU cost?**
A: FREE! Kaggle gives free GPU access (30 hours/week).

**Q: Can I use multiple GPUs?**
A: T4 x2 means 2 GPUs, but this notebook uses only 1. Multi-GPU support requires code changes.

**Q: What if I run out of GPU quota?**
A: Wait until next week, or run on CPU (slower but works).

**Q: Can I download the trained model?**
A: Yes, add this code at the end:
```python
np.save('W1.npy', W1)
np.save('W2.npy', W2)
```
Then download from Output section.

---

## ✅ Checklist

Before running, make sure:

- [ ] GPU is enabled (Settings → Accelerator → GPU)
- [ ] Datasets are added (Data section shows 2 datasets)
- [ ] File paths are correct (check config cell)
- [ ] Internet is on (if you need extra packages)
- [ ] You have GPU quota remaining (check account settings)

Then: **Run → Run All** and wait ~5 minutes!

---

## 🎉 Success Indicators

You'll know it worked when you see:

✅ "GPU Configuration Complete" with GPU model name
✅ "Mixed precision enabled (FP16)"
✅ Training completes in ~2-3 minutes
✅ Four PNG files in output
✅ Final results summary showing ~85% source accuracy

**Happy training!** 🚀
