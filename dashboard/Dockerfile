FROM python:3.8-slim-buster

ARG port

COPY requirements.txt app/requirements.txt

ENV PORT=$port

WORKDIR /app

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

COPY . /app

EXPOSE 5000

#Run the command
CMD ["python3", "wgsi.py"]
