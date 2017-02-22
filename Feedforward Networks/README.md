
In the Deep Learning study group I am a part of - see more [HERE](http://dianapfeil.com/machine%20learning/2017/02/19/deep-learning-reading-group/). 
The past two weeks we focused on Deep Feedfoward Networks with a reading assigment and progarmming assigment.

**Reading:** [Chapter 6 Deep learning book](http://www.deeplearningbook.org/contents/mlp.html)

**Implementation:** Feedfoward Network with MNIST dataset

### Summary of Feedfoward Networks

Summary of Chapter 6 (Provided by Diana Pfiel and some minor aditions of mine):

Feedforward neural networks can be thought of as a way to learn general nonlinear functions. 
They refer to these as MLPs (multi-layer perceptrons). There's a theorem that a 1-layer MLP can represent an arbitrary function!!

There are a few considerations when designing a MLP:

**Cost function (aka loss)** - Think of output y as p(y|x; theta) and then use principle of maximum liklihood to create the loss.
This means we maximize negative log-likelihood (same as min cross entropy between training data and model distribution).
                                               
**Output units** - Use linear activation function for regression (𝑦 = 𝑤𝑇ℎ + 𝑏), sigmoid for binary classification (𝑦 = 𝜎(𝑤𝑇ℎ + 𝑏)),
softmax for multi-class classification(𝑦 = softmax 𝑧 where 𝑧 = 𝑊𝑇ℎ + 𝑏).
The reason is that these functions don’t saturate with cross entropy as a loss because the log undoes the exp.
By saturate, we mean that the derivative does not become 0 in any region, which makes it hard to train.
Non-differentiable points are okay in the activation function.

**Hidden units** <- Just use relu, because it does not saturate. Has biological basis! 
There are tons of other activation functions, but they don't make too much difference, so start with relu.

Typical  activation functions: Threshold, Sigmoid, Tanh
Problem is Saturation because gradient is too small --> Solution ReLU (rectified linear unit)

Summary slides from Princeton [here](https://www.cs.princeton.edu/courses/archive/spring16/cos495/slides/DL_lecture1_feedforward.pdf)

                                                      
                                                      
