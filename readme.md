
# RC_Detector

## Detecting the Root Cause Code Lines in Bug-Fixing Commits by Heterogeneous Graph Learning

Python library dependencies:
+ transformers -v :  4.39.3
+ torch -v : 2.4.0
+ torch-geometric -v : 2.5.3
+ torch-scatter -v : 2.1.2
+ torch-sparse -v : 0.6.18
+ torch-spline-conv -v : 1.2.2
+ pandas -v : 2.0.3 

---
## File Organization
- **`rawData/`**: the raw data.
- **`buildGraph/`**: Part of the graph construction.
- **`crossfold_result/`**: the result for crossfold experiment.
- **`microsoft/`**: Storing the pretrain model.
- **`processRawData/`**: process the raw data.
- **`trainData/`**: the processed data.
- **`dataset1.json`, `dataset2.json`, `dataset3.json`**: Decide what data to feed into the model.

There are six folders. A folder represents an experiment.

### FOR RQ1:
- **DL_apporches fold**: the code of DL approaches.
- **ML_apporches fold**: the code of ML approaches.

### FOR RQ2: 
Run `python train.py --project` to run the experimental:
- `--project`: the individual components RC_Detector-g and RC_Detector-h
- `choices=[key1,key2]`

### FOR RQ3:    
Run `python train.py --rnn_type` to run the experimental:
- `--rnn_type`: the rnn type for RC_Detector
- `choices=["gru","lstm","transformer","gru1", "gru2", "gru3"]`

### FOR RQ4:    
Run `python train.py --attention` to run the experimental:
- `--attention`: the attention mechanisms for RC_Detector
- `choices=["Additive Attention","Cosine Similarity Attention","GAT", "Gaussian Kernel Function"]`

### FOR RQ5:    
Run `python train.py --lr --heads` to run the experimental:
- `--lr`: learning rate for RC_Detector, default: 0.000005
- `--heads`: the number of heads for RC_Detector, default: 8

### FOR RQ6:    
Run `python train.py --model_type` to run the experimental:
- `--model_type`: the embedding layer for RC_Detector
- `choices=["graphcodebert", "codet5", "unixcoder", "codebert"]`

For these pre-trained models, you can download them from Hugging Face. For example, the CodeBERT model can be downloaded from [this URL](https://huggingface.co/microsoft/codebert-base/tree/main) and then placed into the `microsoft/codebert-base` directory.
