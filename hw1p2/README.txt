The structure of the code:
1. set up
2. pre-processing
3. Training
4. Validating
5. Tuning
6. Testing
Explanation:
1. set up: this part mounts google drive and connect to Kaggle.
2. pre-processing: this part reads data to dataloader 
	Class dataset_train: uses X,Y, offset, context
	Dataloader: reads data to dataloader
3. training: this part builds the model
	Train_parameters(model, criterion, iter, lr, data): the model uses a SGD optimizer, a scheduler
	Class MLP: the model uses batch normalization and ReLU in every layer
4. validating: this part assesses the model
	Model_val(model, dataloader_val): uses validation dataset and calculate the accuracy
5.tuning: this part experiments with different hyper-parameters
6.testing: this part calculates the final results for submission
The submitted setting is:
Model: (2040, 900, 900, 450, 225, 71)
linear(in_features=2040, out_features=900), batchnorm1d(900, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(0): Linear(in_features=2040, out_features=900, bias=True), Linear(in_features=900, out_features=900, bias=True),BatchNorm1d(900, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(), Linear(in_features=900, out_features=450, bias=True), BatchNorm1d(450, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(), Linear(in_features=450, out_features=225, bias=True), BatchNorm1d(225, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(), Linear(in_features=225, out_features=71, bias=True)
Context: 25   epoch: 5  learning rate: 0.1 
Criterion: nn.CrossEntropyLoss     Optimizer: SGD   

