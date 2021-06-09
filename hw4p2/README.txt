The structure of the code:
Task 1: pre-processing -> model -> Training -> Testing
Explanation:
1. set up: this part mounts google drive and connect to Kaggle.
2. pre-processing: this part includes several functions
	a)letter2index, index2letter dictionaries
	b)translate function that translate output of model to sentences
	c)pad sequences
	d)dataloader
3. model: the model is composed of several part
	a) pBLSTM: while halving length, a dropout is also applied
	b) encoder: 3 bidirectional LSTM layer without dropout + 3 pblstm layers
	c) Attention
	d) decoder: teacher force and gumbel were implemented. After lstm layers, the output 
	   was then fed to three linear layers(with weight tie to embedding weight) 
	e) Seq2Seq: a pre-training constraint was added to encoder.
	f) levenshtein distance function

4. training: 50 epochs + learning rate 0.003 with + shceduler + adam optimizer + 
	     pretrain:0.95 teacher force decrease 0.02 per epoch + training:0.95 teacher force decrease 0.01 per epoch
5. testing: predict the testing set and submit


