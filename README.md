# Foodlifier

Welcome to Foodlifier, the AI-powered food classifier that can identify your favorite dishes with just a glance! This PyTorch project trains a convolutional neural network (CNN) to classify images of food into different categories using the Food-101 dataset. The training process is visualized using TensorBoard, and the accuracy of the trained network is evaluated on the test data.

## Installation

To run Foodlifier, you will need to install the following dependencies:

- PyTorch
- TorchVision
- NumPy
- Matplotlib

You can install these dependencies using `pip`:

```python
pip install torch torchvision numpy matplotlib
```

You will also need to download the Food-101 dataset and specify the path to the training and test data in the Python code.

## Usage

To use Foodlifier, run the following command:

```python
python train.py
```

This will start the training process, which will take several minutes to complete depending on your hardware.

Once the training is complete, the accuracy of the trained network on the test data will be printed out.

## Visualization

To visualize the training process using TensorBoard, run the following command:

```python
tensorboard --logdir=logs
```

This will start TensorBoard on your local machine, which you can access by navigating to `http://localhost:6006` in your web browser. You should see a dashboard showing the training loss and test accuracy over time.

## License

This project is licensed under the BSD 3-Clause License - see the `LICENSE` file for details.

## Acknowledgments

This project was inspired by the PyTorch tutorial on [Image Classification with PyTorch](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).

&copy; Shafik Quoraishee 2023. All Rights Reserved. 
