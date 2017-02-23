##Tutorial on simple MLP MNIST implementation

TensorFlow feedforward neural network for classification of the MNIST data set.

Two files:
- [mnist.py](https://github.com/adrifloresm/TensorFlow/blob/master/Feedforward%20Networks/mnist.py)
- [fully_connected_feed.py](https://github.com/adrifloresm/TensorFlow/blob/master/Feedforward%20Networks/fully_connected_feed.py)

These files and tutorial are an extension on [TensorFlow Mechanics 101](https://www.tensorflow.org/versions/r0.10/tutorials/mnist/tf/)

Requirements:
Tensorflow 1.0

Run command:
`python fully_connected_feed.py`

For TensorBoard run following command and open in your browser provided address:
`tensorboard --logdir=/tmp/tensorflow/mnist/logs/fully_connected_feed`

For best results:
`python fully_connected_feed.py --learning_rate 0.2 --hidden1 256 --hidden2 128`  

Optional arguments:
- --learning_rate (Optimzer learning rate)
- --hidden1 (Number of units in hidden layer 1)
- --hidden2 (Number of units in hidden layer 2)
- --max_steps (Number steps for training)    
- --batch_size (Batch size.  Must divide evenly into the dataset sizes.)
- --input_data_dir (Directory to put the input data.)
- --log_dir (Directory to put the log data.)
- --fake_data (If true, uses fake data for unit testing.)
- --debug (Use debugger to track down bad values during training.)

## mnist.py
This file has the building components for the graph in 3 main functions.

### 1. inference() 
  Builds the graph as far as needed to return the tensor that would contain the output predictions.
  
  It takes the images placeholder as input and builds on top of it two fully connected layers with **ReLU** activation followed by a ten node **linear** softmax layer specifying the output logits.
 
```python
def inference(images, hidden1_units, hidden2_units):
  """
  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
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
  
```python
 def loss(logits, labels):
 """
 Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')
 ```

First, cross entropy op (`tf.nn.sparse_softmax_cross_entropy_with_logits`) computes sparse softmax cross entropy between logits and labels, i.e. measures probability error in discrete classification (classes are mutually exclusive).

Second, `tf.reduce_mean` averages the cross entropy values across the batch dimension (the first dimension) as the total loss.

### 3. training() 
Sets up the training ops required to compute and apply gradients.
Creates a summarizer to track the loss over time in TensorBoard.
Creates an optimizer and applies the gradients to all trainable variables.

The Op returned by this function is what must be passed to the `sess.run()` call to train the model.
    
```python
def training(loss, learning_rate):
"""
  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
"""
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
In this file, the magic happens... we train the model.

### def run_training():

Train MNIST for a number of steps.

- Unpack Data:`data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)`
- Tell TensorFlow that the model will be built into the default Graph. ` with tf.Graph().as_default(): `
- Generate placeholders for the images and labels in accordance to batch size. ` images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)`
- Call all the functions to build the graph from mnist.py.
   Build a Graph that computes predictions from the inference model. `logits = mnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)`
   - Add to the Graph the Ops for loss calculation. `loss = mnist.loss(logits, labels_placeholder)`
   - Add to the Graph the Ops that calculate and apply gradients. `train_op = mnist.training(loss, FLAGS.learning_rate)`
   - Add the Op to compare the logits to the labels during evaluation. `eval_correct = mnist.evaluation(logits, labels_placeholder)`
- For TensorBoard, build the summary Tensor based on the TF collection of Summaries. `summary = tf.summary.merge_all()`
- Initialize variables Op (`init = tf.global_variables_initializer()`)
- Create a saver for writing training checkpoints. (`saver = tf.train.Saver()`)
- Create a session for running Ops on the Graph. (`sess = tf.Session()`)

To enable debugging (when the --debug flag is provided), I use [TensorFlow Debugger (tfdbg)](https://www.tensorflow.org/versions/master/how_tos/debugger/), wrap the Session object with a debugger code below.
``` python
if FLAGS.debug:
    	sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    	sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan) 
 ```

- Instantiate a SummaryWriter to output summaries and the Graph. (`summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)`)

After everything is built, run the Op to initialize the variables. (`sess.run(init)`)
  
### Training Loop
The graph is run here.
- Start training loop
``` python
  for step in xrange(FLAGS.max_steps):
      start_time = time.time()
 ```
 
-  Fill a feed dictionary with the actual set of images and labels for this particular training step.
```python
 feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)
```
      
- Run one step of the model.
   - Note: "sess.run() returns a tuple with two items. Each Tensor in the list of values to fetch corresponds to a numpy array in the returned tuple, filled with the value of that tensor during this step of training. Since train_op is an Operation with no output value, the corresponding element in the returned tuple is None and, thus, discarded. However, the value of the loss tensor may become NaN if the model diverges during training, so we capture this value for logging."
```python
      _, loss_value = sess.run([train_op, loss],feed_dict=feed_dict)`
      `duration = time.time() - start_time
```

- Write summaries and print overviews every 100 steps. Also perform necessary updates to events files to use TensorBoard visualization.
```python
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file. To visualize 
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
```
- Save checkpoint to be able to restore model (`saver.restore(sess, FLAGS.train_dir)`) for further evaluation.        
```python   
      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
 ```
- Model evaluation every 1000 steps with the `do_eval()` function, called three times for training, validation and test datasets.
```python
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.train)
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.validation)
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.test)
```
 
#### Other functions:
 
- **do_eval()** runs one evaluation against the full epoch of data.
 
```python
def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
 """
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))
  ```
- **fill_feed_dict()** Fills the feed_dict for training of the given step. A python dictionary object is then generated with the placeholders as keys and the representative feed tensors as values.
```python
def fill_feed_dict(data_set, images_pl, labels_pl):
  """
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                 FLAGS.fake_data)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict
```
- **placeholder_inputs()**. Generates placeholder variables to represent the input tensors.
```python
 def placeholder_inputs(batch_size):
 """
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         mnist.IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder
  ```
  
 - **The main function** reads input arguments - sets parameters and runs the app.  `def main(_):  tf.app.run`
