FROM python:3.7.6-buster

RUN mkdir /basketball_project
COPY ./requirements.txt /basketball_project/

RUN pip install --upgrade pip
RUN pip3 install -r /basketball_project/requirements.txt

WORKDIR /basketball_project
COPY . /basketball_project

CMD "pytest"
ENV PYTHONDONTWRITEBYTECODE=true