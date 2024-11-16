Named Entity Recognition with Small Strongly Labeled and Large Weakly
Labeled Data

In this roject we have implemented Named Entity Recognition task of Pubmed data which is created and updated by government every year.At first we have calcualted the F1 score,Precision and Recall on the baseline model such as Bio-Bert,Bert-CRF,PubmedBert and then we have created our model from scratch.

We provide a three stage framework:

Stage I: Domain continual pre-training
In this stage we trianed Bert model using the unlabeled data and saved the model for the furthur stages.

Stage II: Noise-aware weakly supervised pre-training
In this stage firstly we have pre processed the data to convert unlabeled data into weakly labeled indomain data.Then we train the model from previous stage on the strongly labeled data to convert in-domain Bert into Bert-CRF model.Then we din noise aware continual pre-training on the weakly labeled data based on the confidence estimation adn plotted the graph for it.Finally the model is saved for the next satge.

Stage III: Fine-tuning
In the final stage we fine tune the data by combining the strongly labeled data and weakly labeled data and training the model on it.
