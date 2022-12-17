FROM python:3.8

COPY ./app /app

COPY ./requirements.txt /app/requirements.txt

WORKDIR app

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt

RUN wget -c 'https://pjreddie.com/media/files/yolov3.weights' --header 'Referer: pjreddie.com'

RUN mv yolov3.weights ./Models/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
