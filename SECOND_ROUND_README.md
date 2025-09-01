# 🎵 Second Round Hyperparameter Experiments

## Overview

This is a comprehensive second round of hyperparameter tuning for your Spotify profiler item2vec model. Based on your first round results, we've designed a strategic approach to find the optimal configuration.

## 🎯 What This Will Do

**Phase 1: Fine-tune Winners (High Impact, Low Risk)**
- Learning Rate: Test 6 values around your optimal 0.01
- Negative Samples: Test 5 values around your optimal 20
- Hidden Dimensions: Test 5 values to find optimal architecture

**Phase 2: Architecture Refinement (Medium Impact, Medium Risk)**
- Context Windows: Test different sequence lengths
- Weight Decay: Test regularization values
- Batch Sizes: Find speed-performance sweet spot

**Phase 3: Advanced Techniques (High Risk, High Reward)**
- Learning Rate Schedulers: Test different scheduling strategies
- Optimizer Variants: Test AdamW, RAdam

## 🚀 Quick Start

### 1. Check Everything is Ready
```bash
python run_second_round.py --check
```

### 2. See What Will Run (Dry Run)
```bash
python run_second_round.py --dry-run
```

### 3. Start All Experiments
```bash
python run_second_round.py
```

### 4. Resume if Interrupted
```bash
python run_second_round.py --resume
```

## 📊 Expected Results

- **Total Experiments**: 35
- **Estimated Time**: 5-7 hours
- **Expected Improvement**: 1-3% over your current best (0.6936)
- **Target**: Beat 0.6936 validation loss

## 🔧 Features

### ✅ **Progress Tracking**
- Real-time progress updates
- Detailed logging with timestamps
- Memory and time monitoring

### ✅ **Checkpoint & Resume**
- Automatic progress saving
- Resume from any interruption
- No lost work

### ✅ **Optimized Execution**
- Phased approach (high impact first)
- Early stopping for efficiency
- K-fold cross-validation

### ✅ **Comprehensive Analysis**
- Automatic best configuration selection
- Performance comparison with first round
- Visualization of results

## 📁 Output Files

Each experiment creates:
- `embeddings_{experiment_name}.npy` - Trained embeddings
- `experiment_results_{experiment_name}.json` - Detailed results

Final outputs:
- `best_second_round_config.json` - Best configuration found
- `second_round_results.png` - Performance visualization
- `second_round_checkpoint.json` - Progress tracking

## 🎮 Usage Examples

### Run Everything Today
```bash
# Start all experiments
python run_second_round.py

# Let it run overnight - it will save progress automatically
# If interrupted, resume with:
python run_second_round.py --resume
```

### Run Just Phase 1 (High Impact)
```bash
python run_second_round.py --phase 1
```

### Check Progress
```bash
# See what's completed
ls experiment_results_*.json

# Check checkpoint
cat second_round_checkpoint.json
```

## 🚨 Important Notes

1. **Database Required**: Ensure `data/MPD/mpd_database.db` exists
2. **Dependencies**: PyTorch, NumPy, Matplotlib, scikit-learn, tqdm
3. **Interruption Safe**: Use Ctrl+C to pause, resume with `--resume`
4. **Storage**: Each experiment uses ~100K pairs (faster than first round)
5. **Memory**: ~2GB RAM per experiment

## 🔍 Monitoring Progress

### Real-time Progress
The script shows:
- Current experiment and phase
- Progress percentage
- Elapsed time
- Memory usage

### Checkpoint Files
- `second_round_checkpoint.json` - Tracks completed experiments
- Resume automatically from any interruption

### Log Files
- Console output with timestamps
- Each experiment saves detailed results

## 🎯 Success Criteria

**Primary Goal**: Beat your current best validation loss of **0.6936**

**Secondary Goals**:
- Find more stable configurations (lower std across folds)
- Identify faster training configurations
- Understand hyperparameter interactions

## 🚀 After Completion

1. **Best Configuration**: Automatically saved to `best_second_round_config.json`
2. **Production Training**: Use the best config for full dataset training
3. **Analysis**: Review `second_round_results.png` for insights
4. **Next Steps**: Plan production training with optimal hyperparameters

## 🆘 Troubleshooting

### Common Issues

**Database Not Found**
```bash
# Ensure you've built the MPD database first
python pipeline/build_co_occurrence.py
```

**Out of Memory**
- Reduce `max_samples` in the script (currently 100K)
- Use smaller batch sizes

**Interrupted Experiments**
```bash
# Always resume with checkpoint
python run_second_round.py --resume
```

**Missing Dependencies**
```bash
pip install torch numpy matplotlib scikit-learn tqdm
```

## 📈 Expected Timeline

**Phase 1 (High Impact)**: 2-3 hours
- Learning Rate: ~1 hour
- Negative Samples: ~1 hour  
- Hidden Dimensions: ~1 hour

**Phase 2 (Medium Impact)**: 2-3 hours
- Context Windows: ~1 hour
- Weight Decay: ~1 hour
- Batch Sizes: ~1 hour

**Phase 3 (Advanced)**: 1-2 hours
- Schedulers & Optimizers: ~1-2 hours

**Total**: 5-8 hours (can run overnight)

## 🎉 Ready to Start?

```bash
# Check everything is ready
python run_second_round.py --check

# Start the experiments!
python run_second_round.py
```

Your model will be significantly improved by the end of this! 🚀
