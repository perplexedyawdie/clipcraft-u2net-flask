FROM python:3.8.16-slim-bullseye

WORKDIR /app

RUN apt-get update -y
RUN apt-get install -y ffmpeg
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY app.py .
CMD ["flask", "--app", "app", "run", "--debug", "--host=0.0.0.0", "--port=5000"]