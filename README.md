# Alternative approach to Generate Meaningful Assert Statement for JAVA Unit Test Cases using Deep Learning

The project is based on the recent paper titled as "[On Learning Assert Statement for JAVA Unit Test Cases](https://arxiv.org/pdf/2002.05800.pdf) ". The authors of the paper give a novel approach to generate meaningful assert statements for unit test cases called ATLAS (AuTomatic Learning of Assert Statements) which is a Neural Machine Translation (NMT) based approach. Given a test method and a focal method (i.e., the main method under test), ATLAS can predict a meaningful assert statement to assess the correctness of the focal method. Keeping the same end goal in mind, the M.Tech. project aims at giving an alternate approach for doing the similar task. The project aims at building a Transformer Enocder-Decoder architecture to train the available dataset of the previous work making use of [CuBERT](https://arxiv.org/pdf/2001.00059.pdf) encoder by Google-research. Further details are given ahead in the report.

### A short note on CuBERT Encoder: 
Taking inspiration from success of pre-trained contextual embeddings for natural languages, the authors of the paper ”[Learning and evaluating Contextual Embeddings of Source Code](https://arxiv.org/pdf/2001.00059.pdf) ”, have presented the first attempt to apply the underlying technique to source code in the form of CuBERT. BERT produces a bidirectional transformer Encoder by training it to predict values of masked tokens, and whether two sentences follow each other in a natural discourse. The pre-trained model can be fine-tuned for downstream tasks and has been shown to produce state-of-the-art results on a number of natural language understanding benchmarks. So, in the paper mentioned above the authors have derived a contextual embedding of
source code by training a BERT model on source code and called this model CuBERT, short for Code Understanding BERT. Till date CuBERT model has been trained for two languages: Python and JAVA.

Dataset Link: [Click here](https://gitlab.com/cawatson/atlas---deep-learning-assert-statements/-/tree/master/Datasets/Raw_Dataset)

Dataset is called Test-Assert Pairs(TAPs)

Test+Focal Method:
```
shouldcreateRedMana(){
 mage.Mana mana = mage.Mana.RedMana (1);
 "AssertPlaceHolder" ;
} getRed () { return red ;}
```
Assert Statement:
```
org.junit.Assert.assertEqual(2, mana.getRed())
```
Togethere, (Test+Focal Method) and Assert Statement, the dataset is called Test-Assert Pairs(TAPs). Given a TAP, the trained models generate the meaningful assert statement.

Train Size: 150523
Validation Size: 18815
Test Size: 18815

Vocabulary - CuBERT Vocabulary used
Size of Vocabulary: 50297

**The project is divided into two parts:**

 - **1st Part:**  First is the data preparation and processing task.
For the M.Tech project , the CuBERT encoder trained to capture JAVA contextual embeddings is to be used as in the model architecture. The authors have not only released pre-trained model checkpoints for the CuBERT Encoder but have also provided a CuBERT vocabulary file and separate Tokenizer file for JAVA and PYTHON source code in order to tokenize the source code data for specific downstream task. The tokenizer follows specific rules to tokenize a JAVA source code. Now the idea is to use this CuBERT tokenizer file to first tokenize the TAPs data and convert them into vector form such that it is ready to be fed to CuBERT encoder.
 
 - **2nd Part:** The Second part of the project involves Data modeling.
Here we have to build an Encoder-Decoder architecture for training the processed data in the first part. For comparative purpose, three different model architectures are trained.

a. CuBERT Fused Transformer

b. CuBERT Encoder with Transformer Decoder

c. Transformer Encoder-Decoder

### CuBERT Fused Transformer
It is first of the two proposed model in the project. The architecture of the model has been taken from the paper “[Incorporating BERT into Neural Machine Translation](https://openreview.net/attachment?id=Hyl7ygStwB&name=original_pdf)”. The architecture in the Figure below is the architecture used for CuBert Fused model. The paper introduces an architecture where they have fused the output from the last layer of BERT model into the transformer encoder-decoder to obtain better results at Machine translation. The architecture proposed in the project is similar to the one proposed in this paper except for few changes. The authors of the paper have used dropnet technique to regularize the training. In the model proposed, dropnet technique is not used. Instead the output of both CuBERT and transformer encoder and decoder is added and normalized.

![Blank board (1)](https://user-images.githubusercontent.com/58558221/120112382-36e0fd80-c193-11eb-9346-5ad097002ae4.png)

### CuBERT Encoder with Transformer Decoder
This is the second of the two model proposed in the project. It involves use of CUBERT as an encoder and Transformer as a decoder. The input test methods vector batches are fed directly into the CuBERT and the output is fed into the decoder multi- head attention layer.

![Cubert_only (2)](https://user-images.githubusercontent.com/58558221/120112799-0732f500-c195-11eb-9ed7-e9c891c9c9cc.png)

### Transformer Encoder-Decoder
The Transformer Encoder-Decoder serves as benchmark for the other two proposed architecture. It is exactly similar as the one proposed in the original pa-
per  “[Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)”.

![Screenshot from 2021-05-15 23-35-13](https://user-images.githubusercontent.com/58558221/120114395-a9a2a680-c19c-11eb-826a-c4d3e5aec2a3.png)

### Some results and Trainig Loss Plots
Model 1 (CuBERT Fused Transformer)
Model 2 (Cubert Encoder with Transformer Decoder)
Model 3 (Transformer Encoder Decoder

![Screenshot from 2021-05-30 23-21-55](https://user-images.githubusercontent.com/58558221/120114722-25e9b980-c19e-11eb-90ae-a95e347c1ce4.png)

Model 1 Training  Loss Plot

![Training loss](https://user-images.githubusercontent.com/58558221/120114727-27b37d00-c19e-11eb-8189-ad012b1e80e7.png)

Model 2 Training Loss Plot

![Training_loss_Cubert_only](https://user-images.githubusercontent.com/58558221/120114725-27b37d00-c19e-11eb-9a28-53bc5f37fb23.png)

Model 3 Training Loss Plot

![Training_loss_transformer](https://user-images.githubusercontent.com/58558221/120114724-271ae680-c19e-11eb-8c1f-09b78d3aa2a0.png)

### Flow of Project:

 - Step 1 - Create the environment as per the requirment.txt file.
 - Step 2 - Clone the [CuBERT](https://github.com/google-research/google-research/tree/master/cubert) repository. 
 - Step 3 - Download the CuBERT pretrained tensorflow model checkpoints from above repository. Convert them into Pytorch checkpoints using huggingface's [this](https://huggingface.co/transformers/converting_tensorflow_models.html#:~:text=You%20can%20convert%20any%20TensorFlow,three%20files%20starting%20with%20bert_model.) documentation.
 - Step 4 - Convert the Train, Test and Validation data into vectorized form using CuBERT tokenizer and CuBERT vocabulary file. Execute the 'vectorize_data.py' file. We also need to specify the path where the vectorized TAPs and corresponding assert statement are stored.
```
python3 vectorize_data.py Path_to_CuBERT_Vocabulary Path_to_input_file Path_to_output_file
 ```
We need to execute the statement twice. Once when the Path_to_output_file is path where the vectorized TAPs are saved and second when path is for corresponding target assert statements. Further we need to do it for tarin, validation and test set one by one.
 
 - Step 5 - Once the dataset is processed, we need to train the 3 models separately. The train.sh file in each repository train the models of  that specific repository. 'train.sh' script contain command to execute the run.py file for training each model. Make sure that the 'run.py' file has path set for the below locations.
 -->  Path where the vectorized Training and Validation TAPS and assert statement are saved  for processing them into Training and Validation batches.
 -->  Path to 'SummaryWriter' where we want save the log file that saves the training and validation loss plots
 -->  Path where the trained model is to be saved.
 -->  'run.py' file for 'Cubert Fused Transformer' and 'Cubert Encoder with Transformer Decoder model'  has a variable 'PRE_TRAINED_MODEL_CUBERT' which must be set to the location for saved Pytorch CuBERT checkpoints.

**Note:** that in each of the .sh file we need to specify the path where we want to store the trained model and the training plots. Note that each of the model has been trained on GPU cluster with 2 GPUs each of 12 GB using Single Linux Utility Resource Management(SLURM).

 - Step 6 - Once all the models have been trained, the models can be used to make predictions for the TAPs in the test set. To obtain the predictions for test set, execute the file 'generate.py', present for each of the models seperately. 
 ```
 python3 generate.py Path_where_to_save_the_predictions Path_to_Cubert_Vocabulary Path_to_Test_set Path_to_saved_model  
 ```
 **Note:** that the 'generate.py' file for 'Cubert Fused Transformer' and 'Cubert Encoder with Transformer Decoder model' has a variable 'PRE_TRAINED_MODEL_CUBERT' which must be set to the location for saved Pytorch CuBERT checkpoints.

 - Step 7 - Once we obtain the predictions made by different models, these predictions can be used to evalute the model using metric.py script. metric.py file gives the BLEU score, GLUE-4, GLUE-2 scores and also total number of perfect predictions. 
 ```
python3 metric.py Path_to_saved_prediction_file Path_to_actual_assert_statements
 ```

 - Step 8 - Now in order to obtain the predicted assert statement using any of the above models, we can use 'assert.py' script.
```
python3 generate_script.py Path_to_CuBERT_vocabulary Path_Input_TAP_file Path_where_to_saved_model
```

