<p align="center">
  <a href="https://github.com/SuperBruceJia/EEG-DL"> <img width="500px" src="https://github.com/SuperBruceJia/EEG-DL/raw/master/Logo.png"></a> 
  <br />
  <br />
  <a href="https://gitter.im/EEG-DL/community"><img alt="Chat on Gitter" src="https://img.shields.io/gitter/room/nwjs/nw.js.svg" /></a>
  <a href="https://www.anaconda.com/"><img alt="Python Version" src="https://img.shields.io/badge/Python-3.x-green.svg" /></a>
  <a href="https://www.tensorflow.org/install"><img alt="TensorFlow Version" src="https://img.shields.io/badge/TensorFlow-1.13.1-red.svg" /></a>
  <a href="https://github.com/SuperBruceJia/EEG-DL/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
</p>

<!-- <div align="center">
    <a href="https://github.com/SuperBruceJia/EEG-DL"> <img width="500px" src="https://github.com/SuperBruceJia/EEG-DL/raw/master/Logo.png"></a> 
</div> -->

--------------------------------------------------------------------------------

# Welcome to EEG Deep Learning Library

**EEG-DL** is a Deep Learning (DL) library written by [TensorFlow](https://www.tensorflow.org) for EEG Tasks (Signals) Classification. It provides the latest DL algorithms and keeps updated. 

<!-- [![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/EEG-DL/community)
[![Python 3](https://img.shields.io/badge/Python-3.x-green.svg)](https://www.anaconda.com/)
[![TensorFlow 1.13.1](https://img.shields.io/badge/TensorFlow-1.13.1-red.svg)](https://www.tensorflow.org/install)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/SuperBruceJia/EEG-DL/blob/master/LICENSE) -->

## Table of Contents
<ul>
<li><a href="#Documentation">Documentation</a></li>
<li><a href="#Usage-Demo">Usage Demo</a></li>
<li><a href="#Conclusion">Conclusion</a></li>
<li><a href="#Structure-of-the-Code">Structure of the Code</a></li>
<li><a href="#Citation">Citation</a></li>
</ul>

## Documentation
**The supported models** include

| No.   | Model                                                  | Codes           |
| :----:| :----:                                                 | :----:          |
| 1     | Deep Neural Networks                                   | [DNN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/DNN.py) |
| 2     | Convolutional Neural Networks [[Paper]](https://iopscience.iop.org/article/10.1088/1741-2552/ab4af6/meta) [[Tutorial]](https://github.com/SuperBruceJia/EEG-Motor-Imagery-Classification-CNNs-TensorFlow)| [CNN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/CNN.py) |
| 3     | Deep Residual Convolutional Neural Networks [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) | [ResNet](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/ResCNN.py) |
| 4     | Thin Residual Convolutional Neural Networks [[Paper]](https://arxiv.org/abs/1902.10107) | [Thin ResNet](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/Thin_ResNet.py) |
| 5     | Densely Connected Convolutional Neural Networks [[Paper]](https://arxiv.org/abs/1608.06993) | [DenseNet](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/DenseCNN.py) |
| 6     | Fully Convolutional Neural Networks [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) | [FCN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/Fully_Conv_CNN.py) |
| 7     | One Shot Learning with Siamese Networks (CNNs Backbone) <br> [[Paper]](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) [[Tutorial]](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d) | [Siamese Networks](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/Siamese_Network.py) |
| 8     | Graph Convolutional Neural Networks <br> [[Paper]](https://ieeexplore.ieee.org/document/9889159) [[Presentation]](https://shuyuej.com/files/Presentation/A_Summary_Three_Projects.pdf) | [GCN / Graph CNN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/lib_for_GCN/GCN_Model.py) |
| 9    | Deep Residual Graph Convolutional Neural Networks [[Paper]](https://arxiv.org/abs/2007.13484) | [ResGCN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/lib_for_GCN/ResGCN_Model.py) | 
| 10    | Densely Connected Graph Convolutional Neural Networks  | [DenseGCN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/lib_for_GCN/DenseGCN_Model.py) |
| 11    | Bayesian Convolutional Neural Network <br> via Variational Inference [[Paper]](https://arxiv.org/abs/1901.02731) | [Bayesian CNNs](https://github.com/SuperBruceJia/EEG-BayesianCNN) |
| 12    | Recurrent Neural Networks [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [RNN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/RNN.py) |
| 13    | Attention-based Recurrent Neural Networks [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [RNN with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/RNN_with_Attention.py) |
| 14    | Bidirectional Recurrent Neural Networks [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [BiRNN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/BiRNN.py) |
| 15    | Attention-based Bidirectional Recurrent Neural Networks [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [BiRNN with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/BiRNN_with_Attention.py) |
| 16    | Long-short Term Memory [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [LSTM](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/LSTM.py) |
| 17    | Attention-based Long-short Term Memory [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [LSTM with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/LSTM_with_Attention.py) |
| 18    | Bidirectional Long-short Term Memory [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [BiLSTM](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/BiLSTM.py) |
| 19    | Attention-based Bidirectional Long-short Term Memory [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [BiLSTM with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/BiLSTM_with_Attention.py) |
| 20    | Gated Recurrent Unit [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [GRU](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/GRU.py) |
| 21    | Attention-based Gated Recurrent Unit [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [GRU with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/GRU_with_Attention.py) |
| 22    | Bidirectional Gated Recurrent Unit [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [BiGRU](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/BiGRU.py) |
| 23    | Attention-based Bidirectional Gated Recurrent Unit [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [BiGRU with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/BiGRU_with_Attention.py) |
| 24    | Attention-based BiLSTM + GCN [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [Attention-based BiLSTM](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/BiLSTM_with_Attention.py) <br> [GCN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/lib_for_GCN/GCN_Model.py) |
| 25    | Transformer [[Paper]](https://arxiv.org/abs/1706.03762) [[Paper]](https://arxiv.org/abs/2010.11929) | [Transformer](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/main-Transformer.py) |
| 26    | Transfer Learning with Transformer <br> (**This code is only for reference!**) <br> (**You can modify the codes to fit your data.**) | Stage 1: [Pre-training](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/main-pretrain_model.py) <br> Stage 2: [Fine Tuning](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/main-finetuning_model.py) |


## Usage Demo

1. ***(Under Any Python Environment)*** Download the [EEG Motor Movement/Imagery Dataset](https://archive.physionet.org/pn4/eegmmidb/) via [this script](https://github.com/SuperBruceJia/EEG-DL/blob/master/Download_Raw_EEG_Data/MIND_Get_EDF.py).

  ```text
    $ python MIND_Get_EDF.py
  ```

2. ***(Under Python 2.7 Environment)*** Read the .edf files (One of the raw EEG signals formats) and save them into Matlab .m files via [this script](https://github.com/Razvan03/EEG-DL/blob/master/Download_Raw_EEG_Data/Extract-Raw-Data-Into-Matlab-Files.py). FYI, this script must be executed under the **Python 2 environment (Python 2.7 is recommended)** due to some Python 2 syntax. If using Python 3 environment to run the file, there might be no error, but the labels of EEG tasks would be totally messed up.
	I used a conda environment with Python 2.7 using ```text $ conda create --name EEG2.7 python=2.7 ``` in cmd.
	
Then I ran the python script using the line below and it created a .mat dataset of 10 subjects for every 64 channels. I have applied a Notch Filter and Butterworth Band-pass filter in this process.
	
   ```text
    $ python Extract-Raw-Data-Into-Matlab-Files.py
   ```

3. Preprocessed the Dataset via the Matlab and save the data into the Excel files (training_set, training_label, test_set, and test_label) via [Preprocess_EEG_Dataset.m](https://github.com/Razvan03/EEG-DL/tree/master/Preprocess_EEG_Data/For-CNN-based-Models) with regards to different models. FYI, every lines of the Excel file is a sample, and the columns can be regarded as features, e.g., 4096 columns mean 64 channels X 64 time points. Later, the models will reshape 4096 columns into a Matrix with the shape 64 channels X 64 time points. You should can change the number of columns to fit your own needs, e.g., the real dimension of your own Dataset.
	Because the matlab script was running out of memory while trying to save the large dataset as an Excel file I modified the script above with a [function](https://github.com/Razvan03/EEG-DL/blob/master/Preprocess_EEG_Data/For-CNN-based-Models/save_data_in_chunks.m) to save the excels in chunks:

Then I used a [python script](https://github.com/Razvan03/EEG-DL/blob/master/Preprocess_EEG_Data/For-CNN-based-Models/concatenate.py) to concatenate the chunks files into 3 sets and 3 labels .csv files.
	
	
4. ***(Prerequsites)*** Train and test deep learning models **under the Python 3.6 Environment (Highly Recommended)** for EEG signals / tasks classification via [the EEG-DL library](https://github.com/SuperBruceJia/EEG-DL/tree/master/Models), which provides multiple SOTA DL models.

	First, I needed to create another conda environment using ```text conda create --name EEG3.6 python=3.6 ``` with TensorFlow GPU version 1.13.1
  ```text
    Python Version: Python 3.6 (Recommended)
    TensorFlow Version: TensorFlow 1.13.1
  ```

   Use the below command to install TensorFlow GPU Version 1.13.1:

 ```python
    $ pip install --upgrade --force-reinstall tensorflow-gpu==1.13.1 --user
  ```
After installing tensorflow-gpu 1.13.1 it came with CUDA 11.2 as default which isn't compatible with my version of tensorflow. So I unninstalled it manually and then nstalled the CUDA Toolkit 10.0 and cuDNN 7.6.x using conda:
```text
conda install -c anaconda cudatoolkit=10.0 cudnn=7.6.5
```
To train the CNN model on my database I ran the [main-CNN.py](https://github.com/Razvan03/EEG-DL/blob/master/main-CNN.py) :
```text
python main-CNN.py
```
	
i)Training number #1 was made on a 20-subjects database which resulted in an OOM(Out of Memory). This error typically occurs when your GPU runs out of memory during the model training or evaluation process.

ii)Training number #2. I reduced the subject to 10 and then it trained for 300 iterations (num_epoch = 300 ). The following output is obtained:
```text
	Iter 0, Testing Accuracy: 0.47142857, Training Accuracy: 0.51
Iter 0, Testing Loss: 0.74734324, Training Loss: 0.744785
Learning rate is  1e-04


Iter 1, Testing Accuracy: 0.48333332, Training Accuracy: 0.54
Iter 1, Testing Loss: 0.5743346, Training Loss: 0.56509113
Learning rate is  1e-04


Iter 2, Testing Accuracy: 0.4845238, Training Accuracy: 0.56
Iter 2, Testing Loss: 0.5004847, Training Loss: 0.49093938
Learning rate is  1e-04


Iter 3, Testing Accuracy: 0.4952381, Training Accuracy: 0.53
Iter 3, Testing Loss: 0.47017473, Training Loss: 0.46240148
Learning rate is  1e-04


Iter 4, Testing Accuracy: 0.4940476, Training Accuracy: 0.65
Iter 4, Testing Loss: 0.45639592, Training Loss: 0.43720686
Learning rate is  1e-04


Iter 5, Testing Accuracy: 0.48214287, Training Accuracy: 0.54
Iter 5, Testing Loss: 0.45120627, Training Loss: 0.432128
Learning rate is  1e-04
......
......
......
Iter 295, Testing Accuracy: 0.51785713, Training Accuracy: 1.0
Iter 295, Testing Loss: 0.21398115, Training Loss: 0.034377642
Learning rate is  3.125e-06


Iter 296, Testing Accuracy: 0.51785713, Training Accuracy: 1.0
Iter 296, Testing Loss: 0.21302587, Training Loss: 0.0339612
Learning rate is  3.125e-06


Iter 297, Testing Accuracy: 0.5190476, Training Accuracy: 1.0
Iter 297, Testing Loss: 0.21510491, Training Loss: 0.035236362
Learning rate is  3.125e-06


Iter 298, Testing Accuracy: 0.52738094, Training Accuracy: 1.0
Iter 298, Testing Loss: 0.21519342, Training Loss: 0.03485203
Learning rate is  3.125e-06


Iter 299, Testing Accuracy: 0.5107143, Training Accuracy: 1.0
Iter 299, Testing Loss: 0.21810618, Training Loss: 0.03480571
Learning rate is  3.125e-06


Iter 300, Testing Accuracy: 0.50714284, Training Accuracy: 1.0
Iter 300, Testing Loss: 0.21795654, Training Loss: 0.03447302
Learning rate is  3.125e-06
```

The trained CNN that resulted from the iterations above is available [here](https://github.com/Razvan03/EEG-DL/tree/master/Saved_Files):

## Conclusion
The output of my main-CNN is showing that the model is learning, but the performance is not ideal. The training accuracy reaches 1.0, which suggests that the model is overfitting the training data. The testing accuracy, on the other hand, is quite low, fluctuating between around 0.47 and 0.52.

For future tries, I am gonna adjust the architecture of the CNN by adding or removing layers, changing the number of filters, or modifying the filter sizes or add regularization methods like L1, L2, or Dropout to prevent overfitting.
## Structure of the Code

At the root of the project, you will see:

```text
├── Download_Raw_EEG_Data
│   ├── Extract-Raw-Data-Into-Matlab-Files.py
│   ├── MIND_Get_EDF.py
│   ├── README.md
│   └── electrode_positions.txt
├── Draw_Photos
│   ├── Draw_Accuracy_Photo.m
│   ├── Draw_Box_Photo.m
│   ├── Draw_Confusion_Matrix.py
│   ├── Draw_Loss_Photo.m
│   ├── Draw_ROC_and_AUC.py
│   └── figure_boxplot.m
├── LICENSE
├── Logo.png
├── MANIFEST.in
├── Models
│   ├── DatasetEEG
│   │   ├── all_data.csv
│   │   ├── all_labels.csv
│   │   ├── test_label.csv
│   │   ├── test_set.csv
│   │   ├── training_label.csv
│   │   └── training_set.csv
│   ├── DatasetAPI
│   │   └── DataLoader.py
│   ├── Evaluation_Metrics
│   │   └── Metrics.py
│   ├── Initialize_Variables
│   │   └── Initialize.py
│   ├── Loss_Function
│   │   └── Loss.py
│   ├── Network
│   │   ├── BiGRU.py
│   │   ├── BiGRU_with_Attention.py
│   │   ├── BiLSTM.py
│   │   ├── BiLSTM_with_Attention.py
│   │   ├── BiRNN.py
│   │   ├── BiRNN_with_Attention.py
│   │   ├── CNN.py
│   │   ├── DNN.py
│   │   ├── DenseCNN.py
│   │   ├── Fully_Conv_CNN.py
│   │   ├── GRU.py
│   │   ├── GRU_with_Attention.py
│   │   ├── LSTM.py
│   │   ├── LSTM_with_Attention.py
│   │   ├── RNN.py
│   │   ├── RNN_with_Attention.py
│   │   ├── ResCNN.py
│   │   ├── Siamese_Network.py
│   │   ├── Thin_ResNet.py
│   │   └── lib_for_GCN
│   │       ├── DenseGCN_Model.py
│   │       ├── GCN_Model.py
│   │       ├── ResGCN_Model.py
│   │       ├── coarsening.py
│   │       └── graph.py
│   ├── __init__.py
│   ├── main-BiGRU-with-Attention.py
│   ├── main-BiGRU.py
│   ├── main-BiLSTM-with-Attention.py
│   ├── main-BiLSTM.py
│   ├── main-BiRNN-with-Attention.py
│   ├── main-BiRNN.py
│   ├── main-CNN.py
│   ├── main-DNN.py
│   ├── main-DenseCNN.py
│   ├── main-DenseGCN.py
│   ├── main-FullyConvCNN.py
│   ├── main-GCN.py
│   ├── main-GRU-with-Attention.py
│   ├── main-GRU.py
│   ├── main-LSTM-with-Attention.py
│   ├── main-LSTM.py
│   ├── main-RNN-with-Attention.py
│   ├── main-RNN.py
│   ├── main-ResCNN.py
│   ├── main-ResGCN.py
│   ├── main-Siamese-Network.py
│   └── main-Thin-ResNet.py
├── NEEPU.png
├── Preprocess_EEG_Data
│   ├── For-CNN-based-Models
│   │   └── make_dataset.m
│   ├── For-DNN-based-Models
│   │   └── make_dataset.m
│   ├── For-GCN-based-Models
│   │   └── make_dataset.m
│   ├── For-RNN-based-Models
│   │   └── make_dataset.m
│   └── For-Siamese-Network-One-Shot-Learning
│       └── make_dataset.m
├── README.md
├── Saved_Files
│   └── README.md
├── requirements.txt
└── setup.py
```

## Citation

If you find our library useful, please considering citing our papers in your publications.
We provide a BibTeX entry below.

```bibtex
@article{hou2022gcn,
	title   = {{GCNs-Net}: A Graph Convolutional Neural Network Approach for Decoding Time-Resolved EEG Motor Imagery Signals},
        author  = {Hou, Yimin and Jia, Shuyue and Lun, Xiangmin and Hao, Ziqian and Shi, Yan and Li, Yang and Zeng, Rui and Lv, Jinglei},
	journal = {IEEE Transactions on Neural Networks and Learning Systems},
	volume  = {},
	number  = {},
	pages   = {1-12},
	year    = {Sept. 2022},
	doi     = {10.1109/TNNLS.2022.3202569}
}
  
@article{hou2020novel,
	title     = {A Novel Approach of Decoding EEG Four-class Motor Imagery Tasks via Scout {ESI} and {CNN}},
	author    = {Hou, Yimin and Zhou, Lu and Jia, Shuyue and Lun, Xiangmin},
	journal   = {Journal of Neural Engineering},
	volume    = {17},
	number    = {1},
	pages     = {016048},
	year      = {Feb. 2020},
	publisher = {IOP Publishing},
	doi       = {10.1088/1741-2552/ab4af6}
	
}

@article{hou2022deep,
	title   = {Deep Feature Mining via the Attention-Based Bidirectional Long Short Term Memory Graph Convolutional Neural Network for Human Motor Imagery Recognition},
	author  = {Hou, Yimin and Jia, Shuyue and Lun, Xiangmin and Zhang, Shu and Chen, Tao and Wang, Fang and Lv, Jinglei},   
	journal = {Frontiers in Bioengineering and Biotechnology},      
	volume  = {9},      
	year    = {Feb. 2022},      
	url     = {https://www.frontiersin.org/article/10.3389/fbioe.2021.706229},       
	doi     = {10.3389/fbioe.2021.706229},      
	ISSN    = {2296-4185}
}

@article{Jia2020AttentionGCN,
	title   = {Attention-based Graph {ResNet} for Motor Intent Detection from Raw EEG signals},
	author  = {Jia, Shuyue and Hou, Yimin and Lun, Xiangmin and Lv, Jinglei},
	journal = {arXiv preprint arXiv:2007.13484},
	year    = {2022}
}
```

Our papers can be downloaded from:
1. [A Novel Approach of Decoding EEG Four-class Motor Imagery Tasks via Scout ESI and CNN](https://iopscience.iop.org/article/10.1088/1741-2552/ab4af6/meta)<br>
*Codes and Tutorials for this work can be found [here](https://github.com/SuperBruceJia/EEG-Motor-Imagery-Classification-CNNs-TensorFlow).*<br>

**Overall Framework**:

<div>
    <div style="text-align:center">
    <img width=100%device-width src="https://user-images.githubusercontent.com/31528604/200832194-ea4198f4-e732-436c-bdec-6e454341c442.png" alt="Project1">
</div>

**Proposed CNNs Architecture**:

<div>
    <div style="text-align:center">
    <img width=60%device-width src="https://user-images.githubusercontent.com/31528604/200834151-647319e6-9f6c-428b-b763-36d8859acab9.png" alt="Project1">
</div>

--------------------------------------------------------------------------------

2. [GCNs-Net: A Graph Convolutional Neural Network Approach for Decoding Time-resolved EEG Motor Imagery Signals](https://ieeexplore.ieee.org/document/9889159)<br> 
***Slides Presentation** for this work can be found [here](https://shuyuej.com/files/EEG/GCNs-Net-Presentation.pdf).*<br>

<div>
    <div style="text-align:center">
    <img width=100%device-width src="https://github.com/SuperBruceJia/SuperBruceJia.github.io/raw/master/imgs/Picture2.png" alt="Project2">
</div>

--------------------------------------------------------------------------------

3. [Deep Feature Mining via Attention-based BiLSTM-GCN for Human Motor Imagery Recognition](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full)<br>
***Slides Presentation** for this work can be found [here](https://shuyuej.com/files/EEG/BiLSTM-GCN-Presentation.pdf).*<br>

<div>
    <div style="text-align:center">
    <img width=100%device-width src="https://user-images.githubusercontent.com/31528604/200833742-1b775246-7bb8-4add-a6f9-210f1c5249a0.JPEG" alt="Project3.1">
</div>

<div>
    <div style="text-align:center">
    <img width=100%device-width src="https://user-images.githubusercontent.com/31528604/200833795-157eba9e-0f1b-4f24-8038-8fb385fedcbd.JPEG" alt="Project4.1">
</div>

--------------------------------------------------------------------------------

4. [Attention-based Graph ResNet for Motor Intent Detection from Raw EEG signals](https://arxiv.org/abs/2007.13484)


## Organizations

The library was created and open-sourced by Shuyue Jia, supervised by Prof. Yimin Hou, at the School of Automation Engineering, Northeast Electric Power University, Jilin, Jilin, China.<br>
<a href="http://www.neepu.edu.cn/"> <img width="500" height="150" src="https://github.com/SuperBruceJia/EEG-DL/raw/master/NEEPU.png"></a>
