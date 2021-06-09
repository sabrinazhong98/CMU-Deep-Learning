The structure of the code:
Task 1: pre-processing -> model -> Training -> Testing
Explanation:
1. set up: this part mounts google drive and connect to Kaggle.
2. pre-processing: read the dataset with pad sequences
3. model: build the model 
	2 cnn layers + 4 lstm layers(with dropout 0.6) + 3 linear layers
4. training: 30 epochs + learning rate 2e-3 with+ weight decay 5e-6 + ctc loss + shceduler + adam optimizer=
5. testing: predict the testing set and submit
   In this part the ctc beam decoder is implemented with beam width of 20


