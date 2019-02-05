# Flask TensorFlow

This simple web app serves a [TensorFlow](https://www.tensorflow.org/) protobuf file as a flask web app. A docker image can be found here:

```
https://hub.docker.com/r/chrisparsonsdev/flask_tensorflow
```

The app is intentionally simple, and merely serves as a guide for those looking to take their TensorFlow models to production/integrate them into an existing web app.

Using this as an example? The model here classifies chocolates. (I thought cats were a bit overdone). The whole dataset is [here](https://github.com/chrisparsonsdev/paiv_datasets) and is a great place to start if you're looking for some test images to check everything works.

If you have any questions, or if you just need some help, feel free to raise them as issues against this GitHub repo or tweet me [@chrisparsonsdev](https://twitter.com/chrisparsonsdev).

## Usage

The app is designed to work with [Image Classification](https://www.tensorflow.org/tutorials/keras/basic_classification) workloads. Input images are assigned a category based on training data.

By default there are two endpoints, one which presents this readme as a homepage and `/classification` that will classify user-supplied images against the trained model.

To classify an image submit the following URL in your browser:

```
IP:PORT/classification?file_name=/path/to/image
```

## Running The Application

There are a couple of ways to run the application, either via docker (see instructions on building your own docker container later in the readme) or via command line.

* **Docker**

1. Pull the container to your machine

```
docker pull chrisparsonsdev/flask_tensorflow
```

2. Run the container

```
$ docker run it -p 5000:5000 flask_tensorflow
```

You can deploy with Kubernetes and stuff, but I'm not providing instructions on that here.

* **Command Line**

1. Clone this repository

```
git clone git@github.com:ChrisParsonsDev/flask_tensorflow.git
```

2. Navigate to the source

```
cd flask_tensorflow
```

3. Run the Application

```
python app.py
```

Either approach will let you view the app in your browser at:

```
IP:PORT:5000/
```

## The Code

As you know **all** code is self documenting so there really isn't any need for this section.

Seriously though, if rooting through error messages for weeks doesn't light your fire, here's a description of what's going on in `app.py` that I hope will get you off the ground really quickly.

### Endpoints

These endpoints define where your app is listening for http requests. The preconfigured routes are described below. Of course you're welcome to replace these and do your own thing.

* `/` (default)

The default route just serves this readme as an html page. A last ditch effort to get my documentation in front of people.

* `/classification`

This route takes the input file and performs classification against the TensorFlow model.

### TensorFlow

The TensorFlow code for this app is relatively simple. We load the stored graph from a protobuf file, configure a session and perform relevant requests.

### Utilities

* `def readLabels()`

Reads in each of the labels in the `labels.txt` file line by line, adding them to a list we can use when we return results from the API.

* `def apiResponseCreator(labels, classifications)`

This function concatenates two lists and returns the result as a dictionary. This object can be `jsonify-ed` and sent back to the client. How you then handle that is up to you!

list 1 - the labels
list 2 - the classification results from the TensorFlow model.

* `def printTensors(model_file)`

If you built the model yourself you probably wont ever need to use this function. Otherwise it'll print out all the layer names for the graph in your protobuf. You'll need that to identify the input and output tensor names so you can read output/classify data for your custom model.

It'll load the model file, import the graph definition and iterate over that definition printing the `operation.name` at each layer.

## Modifying the app

The comments in the code, combined with this overview, should make modifying the app to suit your needs as accessible as possible.

### Exposing different ports

Networking and Docker seems like a lot of fun, so rather than going through it all in detail here I thought I'd just simply explain how to expose different ports.

Line `24` of the `Dockerfile` configures the exposed port/networking for the container itself. I've set it to 5000 _the default for flask_ but modifying this will allow you to integrate the container into your current environment.

### TensorFlow Session

There are lots of reasons you might want to change the TensorFlow session. By default we're not using GPUs or any CPU related acceleration. To customise your session modify the `sess_config` [tf.ConfigProto](https://www.tensorflow.org/api_docs/python/tf/ConfigProto) object.

### Using Custom Models

I've done as much as possible to keep the app as modular as possible. Adding a custom model should be as easy as changing the `labels.txt` and `model.pb` files in the `/models` directory.

You'll then need to modify the `app.py` script to account for these changes. Here are a few of the **gotchas** you'll need to check / update:

* **Input Operation Name**

This is the [tf.Operation](https://www.tensorflow.org/api_docs/python/tf/Operation) for the input layer of the Graph you've uploaded in the protobuf file. It needs to be located by name so that we can pass new images/data to this layer of the network.

```
input_op = graph.get_operation_by_name('LAYERNAME')
```

* **Output Operation Name**

As above but the output layer of the network.  

```
output_op = graph.get_operation_by_name('LAYERNAME')
```

If you're not sure what the layer names are (and why would you be) you can use the helper utility `printTensors` to print the full names to the console.

```
printTensors('PATH_TO_MODEL')
```
### Classification

These code described here can all be found in the `classification()` function called when requests are made to the `/classification` route of the application.

This is where you'll have to make the bulk of the changes to make the code work for your model.

Two common modifications will be:

1. Modify input data to match `input` tensor from the graph.
2. Modify the code to handle image uploads as opposed to file paths

The code here looks for the `file_path` parameter in the request body which is commonly passed as part of the URL string (see usage).

We load in the image file using the `imread` function and resize it to `224*224` to match the input file size of the network (mobilenet trained on chocolates in my example).

I also use the [np.expand_dims](https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html) function to prepend the batch size to the input object and convert the tensor from (224,224,3) to (1,224,224,3).

```
x_input = np.expand_dims(image, axis=0)
```

Yes you can do this with `tf.expand_dims` but then you end up having to manage graph contexts and session scopes.

We then pass this np array to the TensorFlow session to obtain results:

```  
tf_results = sess.run(output_tensor, {input_tensor : x_input})
```

Finally the classification is converted to JSON and returned to the client.

```
return jsonify(apiResponseCreator(classification_labels, predictions.tolist()))
```

## Docker Build Instructions

Rebuilding this docker container for your own use is fairly easy. The instructions below provide a brief guide, but are no means exhaustive, please see the [Docker documentation](https://docs.docker.com/) for further details and more advanced tutorials.

1. Clone this repository

```
git clone git@github.com:ChrisParsonsDev/flask_tensorflow.git
```

2. Naviage to the source code repository

```
cd flask_tensorflow
```

3. Modify the code/app as you wish!

4. Build new docker container

```
docker build -t flask_tensorflow:latest .
```

5. Run the container

To run the container locally, with an interactive terminal session, enter the following command:

```
$ docker run it -p 5000:5000 flask_tensorflow
```
