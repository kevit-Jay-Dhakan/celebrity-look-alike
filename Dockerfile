FROM python:3.12.1

WORKDIR /

RUN apt-get update && apt-get install -y libgl1-mesa-dev && apt install -y python3-h5py && apt install -y python3-dev libhdf5-dev

COPY requirements.txt .

RUN pip install --no-binary=h5py h5py && pip install -r requirements.txt

COPY . .

EXPOSE 29492

CMD ["./run_apps.sh"]
