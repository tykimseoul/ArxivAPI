FROM public.ecr.aws/lambda/python:3.6

RUN pip3 install --upgrade pip

WORKDIR .

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt
RUN pip3 uninstall opencv-python -y
RUN pip3 install opencv-python-headless

COPY . ${LAMBDA_TASK_ROOT}

EXPOSE 5000

CMD [ "flaskapp.handler" ]