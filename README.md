# Project Documentation

## Running the Code

Firstly, ensure that Python is installed on your local machine. Then, install all necessary Python packages using pip with the following command:

```bash
pip install numpy tensorflow matplotlib
```

Once all packages are installed, download the `CNN.py` file containing my implementation. Finally, run the script using the command:

```bash
python CNN.py
```


## Network Design Overview

One critical aspect that required trial and error was the choice of optimization algorithm. Ultimately, I settled on Adam optimization after conducting multiple tests with other optimizers, including SGD, Adadelta, Adafactor, and Adagrad.

The results of my experiments revealed that Adam yielded substantially higher accuracy while only taking a few more seconds than the next fastest optimizer, which was SGD. For example, after the first epoch, Adam completed each step in **298s** with an accuracy of **0.3093**, whereas SGD achieved an accuracy of **0.2048** with **252m** per step.

### Gradescope Limitations

Another important factor influencing my network design was the limitations set by Gradescope. Although the instructions in `hw4.pdf` indicate running through 10 epochs, they also specify that this should take no more than one hour. However, Gradescope has a timeout limit of only 30 minutes, which meant that my final code submission did not include 10 epochs. Despite this, my Gradescope submission consistently achieved around **0.55** accuracy with only 1 epoch. When I ran the code on my local machine for 10 epochs, I obtained a training accuracy of **0.9612** and a test accuracy of **0.6885**.

## Graphical Analysis

The graphs below showcase the time per epoch and the training and testing errors from running my code on my local machine. Upon analyzing the graphs, it can be noted that there was a significant decrease in testing errors, while training errors decreased at a slower rate. This discrepancy can be attributed, in part, to the computational constraints limiting the batch size to **64**. With a more capable machine and a larger batch size, we could observe a more substantial decrease in testing errors over time.

![Figure_1](https://github.com/user-attachments/assets/9de9cf51-80bb-49c4-b987-b6f173a01a65)

![Figure_2](https://github.com/user-attachments/assets/57531b65-6e5c-4039-8b7f-fc4e9df025ab)
