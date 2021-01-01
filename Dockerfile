FROM tensorflow/tensorflow:2.4.0

# Upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean

RUN pip3 install --upgrade pip

WORKDIR .

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt
RUN pip3 uninstall opencv-python -y
RUN pip3 install opencv-python-headless

COPY . .

EXPOSE 5000

CMD ["python3", "flaskapp.py"]