The structure of the code:
Task 1: pre-processing -> Training -> Tuning -> Testing
Explanation:
1. set up: this part mounts google drive and connect to Kaggle.
2. pre-processing: read image dataset with transforms(colorjitter, randomhorizontalFlip, totensor)
3. training: build the model (notice: I learned this structure from the source code of resnet online. However, I wrote my own version with different parameters. )
	Simple residual block structure: (1)convolution layer 1 with kernel size 1, stride 1 (2) batch norm (3) relu (4) dropout with p = 0.1 (5) convolution layer 2 with kernel size 3,  self-defined stride, padding 1
(6) batch norm (7) shortcut layer with either identity and batch norm layer
	Resnet structure: (1) inchannel of 32 + convolution layer with kernel size 3, stride 1, padding 1 + batch norm + relu + max pool (2)3 simple residual block layer with out channel of 32, stride 1 (3)4 simple residual block layer with out channel of 48, stride 1 (4)6 simple residual block layer with out channel of 96, stride 1 (5)3 simple residual block layer with out channel of 192, stride 2 (6) batch norm layer +avg pool + flatten + linear layer (7)option to return embedding or output
	Training: train the model and validate data every epoch
4.tuning: this part experiments with different hyper-parameters
       The submitted setting is:
       Model: resnet34 epoch: 25  learning rate: 0.1 Criterion: nn.CrossEntropyLoss    Optimizer: SGD   
       In_features: 3 weight decay: 5e-5   scheduler: step size of 10 with gamma 0.2    feature dimension: 1200
5. testing: predict the testing set and submit

Task 2: pre-processing -> cosine similarity -> testing
1. pre-processing:  read images
2. cosine similarity: compute the validation set with nn.CosineSimilarity and calculate auc score. The model is from task 1
3. testing: predict the testing set and submit
