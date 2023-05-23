FROM python:3.8.16-slim-bullseye

WORKDIR /app

RUN apt-get update -y
# RUN apt-get install -y ffmpeg
# RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN apt-get install libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 libsndfile1-dev -y
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY app.py .
COPY u2net.onnx .
CMD ["flask", "--app", "app", "run", "--debug", "--host=0.0.0.0", "--port=5000"]

