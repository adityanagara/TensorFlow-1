Tutorial on simple MLP MNIST implementation.
TensorFlow feedfoward neural network for classification of the MNIST data set.

Two files:
[mnist.py](https://github.com/adrifloresm/TensorFlow/blob/master/Feedforward%20Networks/mnist.py)
[fully_connected_feed.py](https://github.com/adrifloresm/TensorFlow/blob/master/Feedforward%20Networks/fully_connected_feed.py)

These files and tutorial are an extension on [TensorFlow Mechanic 101](https://www.tensorflow.org/versions/r0.10/tutorials/mnist/tf/)

Requirements:
Tensorflow 1.0

Run command:
`python fully_connected_feed.py`

Optional arguments
--learning_rate (Optimzer learning rate)
--hidden1 256 (Number of units in hidden layer 1)
--hidden2 128 (Number of units in hidden layer 2)
--max_steps (Number steps for training)    
--batch_size (Batch size.  Must divide evenly into the dataset sizes.)
--input_data_dir (Directory to put the input data.)
--log_dir (Directory to put the log data.)
--fake_data (If true, uses fake data for unit testing.)
--debug (Use debugger to track down bad values during training.)

## mnist.py
This file builds the graph in 3 steps.

### 1. inference() 
  Builds the graph as far as needed to return the tensor that would contain the output predictions..
  It takes the images placeholder as input and builds on top of it two fully connected layers with **ReLU** activation followed by a ten node **linear** softmax layer specifying the output logits.

  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  
```
def inference(images, hidden1_units, hidden2_units):
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units], stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))), name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units], stddev=1.0 / math.sqrt(float(hidden1_units))),  name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(  tf.truncated_normal([hidden2_units, NUM_CLASSES], stddev=1.0 / math.sqrt(float(hidden2_units))),  name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
    logits = tf.matmul(hidden2, weights) + biases
  return logits
```
### 2. loss() (aka. Cost)
 Calculates the loss from the logits (inference) and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.

```
 def loss(logits, labels):
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')
 ```

First, cross entropy op (tf.nn.sparse_softmax_cross_entropy_with_logits) computes sparse softmax cross enotropy between logits and labels, i.e. measures probability error in discrete classification (classes are mutually exclusive).

Second, tf.reduce_mean averages the cross entropy values across the batch dimension (the first dimension) as the total loss.

### 3. training() 
Sets up the training ops required to compute and apply gradients.
Creates a summarizer to track the loss over time in TensorBoard.
Creates an optimizer and applies the gradients to all trainable variables.

The Op returned by this function is what must be passed to the `sess.run()` call to train the model.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
    
```
def training(loss, learning_rate):
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  #optimizer = tf.train.RMSPropOptimizer(learning_rate) 

  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op
 ```
   
## fully_connected_feed.py
  Train the Model.
  
def run_training():
Train MNIST for a number of steps.

Read Data
data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
 
Tell TensorFlow that the model will be built into the default Graph. (` with tf.Graph().as_default(): `)
  
Generate placeholders for the images and labels in accordance to Batch size

Call all the functions to build the graph from mnist.py.
Build a Graph that computes predictions from the inference model.
Add to the Graph the Ops for loss calculation.
Add to the Graph the Ops that calculate and apply gradients.
Add the Op to compare the logits to the labels during evaluation.

For TensorBoard, build the summary Tensor based on the TF collection of Summaries. (`summary = tf.summary.merge_all()`)

Initialize variables (`init = tf.global_variables_initializer()`)

Create a saver for writing training checkpoints. (`saver = tf.train.Saver()`)

Create a session for running Ops on the Graph. (`sess = tf.Session()`)

Instantiate a SummaryWriter to output summaries and the Graph. (`summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)`)

And then after everything is built, run the Op to initialize the variables. (`sess.run(init)`)
  
HERE TO BE CONTINUED
 
 ---
 def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  
  def fill_feed_dict(data_set, images_pl, labels_pl):
  
  def placeholder_inputs(batch_size):
  
  def main(_):
  The main function reads input arguments - sets parameters
  tf.app.run
