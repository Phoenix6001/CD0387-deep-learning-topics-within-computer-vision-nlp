# Image Classification using AWS SageMaker

In this project, we utilize AWS Sagemaker to train an already pre-trained Resnet50 model capable of image classification on the given dataset for dog breed classification. Additionally, we incorporate Sagemaker profiling, debugger, hyperparameter tuning, and other commendable practices in machine learning engineering.


## Project Set Up and Installation
This project was developed and tested AWS SageMaker. It was created from a starter file provided by udacity [here](https://github.com/udacity/CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter).

### AWS Execution Role:
The AWS execution role used for the project should have the following access:

- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`

## Dataset
The dataset provided is the dog breed classification dataset, accessible through this [link](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). However, the project has been formulated in a manner that is not reliant on a specific dataset.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data.
```sh
!wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
!unzip dogImages.zip
unzip dogImages.zip
aws s3 sync dogImages/  s3://<default_s3_bucket>/data/ 
```

## Hyperparameter Tuning
We utilize the pre-trained ResNet50 model from PyTorch, a convolutional neural network, to enable transfer learning. We then fine-tune this model using transfer learning techniques to classify dog breeds in images.

In this project, we focus on tuning two key parameters - the learning_rate and batch_size - as these impact both the model's accuracy and speed of conversion. The learning_rate falls within `0.001` to `0.1`, while the batch_size can take on one of five values (`32, 64, 128, 256, or 512`).

To achieve the best results, we execute a hyperparameter tuning job that selects parameters from the search space, runs a training job, and then makes predictions.
The primary objective of this process is to improve the Test Loss metric.

Ultimately, the optimal training hyperparameters will minimize the Test Loss metric.

> Completed Hyperparameter Tuning Job :point_down:

![Hyperparameter Tuning Job](./screenshots/hpo_tuning.png?raw=true "Completed Hyperparameter Tuning Job")

> Summery of the best training job :point_down:

![best training job](./screenshots/best_training_job_summary.png?raw=true "best training job")

##### Best hyperparameters :point_down:

- `batch_size: 32`
- `learning_rate: 0.0034`

## Debugging and Profiling
We employed the SMDebug client library from Amazon SageMaker to facilitate model debugging and profiling. The Sagemaker debugger allows us to monitor our machine learning model's training performance, record training and evaluation metrics, and plot learning curves. Additionally, it can detect potential problems such as overfitting, overtraining, poor weight initialization, and vanishing gradients.

Underfitting occurs when the validation score does not improve over time, suggesting that the model is not learning enough from the data. On the other hand, overfitting refers to a situation in which training curve keeps improving while the validation curve is getting worse. Both of these issues can be addressed by tuning the hyperparameters, or by collecting more data samples.

> Debugger Output :point_down:

![Debugger output](./screenshots/debugging_output.png?raw=true "Debugger output")

To enhance the algorithm's performance, providing additional training time with an extended choice of hyperparameters would be beneficial.

### Results
Increasing the training time can improve the algorithm's performance. Profiling the model revealed that the GPU on the compute instance was underutilized, likely due to the use of small batch size (32). We may want to consider either switching to a smaller instance type or increasing the batch size. This issue may have arisen because we used different tuning and model training instances.


profiler results can be found [here](https://github.com/Phoenix6001/Image-classification/tree/main/ProfilerReport/profiler-output)


## Model Deployment
To enable inference, the model was deployed to a Sagemaker endpoint using an `ml.m5.large` instance and The inference script is designed to accept an image URL as input.

![Endpoint](./screenshots/endpoint.png?raw=true "Endpoint")


The deployed endpoint can be queried using the predict function implemented in the notebook.

![Endpoint](./screenshots/predict.png?raw=true "Endpoint")
