# our base image
FROM ubuntu:16.04

# upgrade pip
RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

# install Python modules needed by the Python app
COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r /usr/src/app/requirements.txt

# copy files required for the app to run

COPY app.py /usr/src/app/
# TensorFlow graph/labels
COPY model/graph.pb /usr/src/app/model/
COPY model/labels.txt /usr/src/app/model/


# port number to expose
EXPOSE 5000

# run the application
CMD ["python", "/usr/src/app/app.py"]
