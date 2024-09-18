
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
- **`rawData/`**: The raw data.
- **`buildGraph/`**: Part of the graph construction.
- **`crossfold_result/`**: The result for crossfold experiment.
- **`microsoft/`**: Store the pretrain model.
- **`processRawData/`**: Process the raw data.
- **`trainData/`**: The processed data.
- **`dataset1.json`, `dataset2.json`, `dataset3.json`**: Decide what data to feed into the model.

There are six folders, each representing an experiment.
They share some common files with the same functionality, but details may vary depending on the specific experiment:

- **`train.py`**: The main training script, which configures the experiment type via command-line parameters.
- **`model.py`**: Contains the model definition and initialization.
- **`genMiniGraphs.py`**: Generates and processes graph data.
- **`genBatch.py`**: Generates batches of data for model training.
- **`genPairs.py`**: Creates sample pairs for ranking models.
- **`genPyG.py`**: Converts data into a format supported by PyTorch Geometric.
- **`eval.py`**: Evaluates the model's performance.


### FOR RQ1:
- **DL_apporches fold**: the code of DL approaches,including Bi-LSTM.

Run `python dl_approches.py ` to run the experiment.
- **ML_apporches fold**: the code of ML approaches,including RF,LR,SVM,XGB,KNN.

Run `python ml_approches.py ` to run the experiment.
- **Neural SZZ**: the code of Neural SZZ.

Run `python train.py ` to run the experiment.

### FOR RQ2: 
- **key1.py**: The model uses RC_Detector-g.
- **key2.py**: The model uses RC_Detector-h.
- **RC_Detector-g.py**: Implementation of RC_Detector-g.

Run `python train.py --project` to run the experiment:
- `--project`: The individual components RC_Detector-g and RC_Detector-h.
- `choices=[key1, key2]`

### FOR RQ3:    
- **AllRnn.py**: Implementation of all RNN types.
- **Transformer_implementation.py**: Implementation of Transformer.

Run `python train.py --rnn_type` to run the experiment:
- `--rnn_type`: The RNN type for RC_Detector.
- `choices=["gru", "lstm", "transformer", "gru1", "gru2", "gru3"]`

### FOR RQ4:    
- **Additive Attention.py**: Implementation of RC_Detector with Additive Attention.
- **Cosine Similarity Attention.py**: Implementation of RC_Detector with Cosine Similarity Attention.
- **Gaussian Kernel Function.py**: Implementation of RC_Detector with Gaussian Kernel Function.
- **GAT.py**: Implementation of RC_Detector with GAT.

Run `python train.py --attention` to run the experiment:
- `--attention`: The attention mechanisms for RC_Detector.
- `choices=["Additive Attention", "Cosine Similarity Attention", "GAT", "Gaussian Kernel Function"]`

### FOR RQ5:    
Run `python train.py --lr --heads` to run the experiment:
- `--lr`: Learning rate for RC_Detector, default: 0.000005.
- `--heads`: The number of heads for RC_Detector, default: 8.

### FOR RQ6:    
Run `python train.py --model_type` to run the experiment:
- `--model_type`: The embedding layer for RC_Detector.
- `choices=["graphcodebert", "codet5", "unixcoder", "codebert"]`

For these pre-trained models, you can download them from Hugging Face. For example, the CodeBERT model can be downloaded from [this URL](https://huggingface.co/microsoft/codebert-base/tree/main) and then placed into the `microsoft/codebert-base` directory.
