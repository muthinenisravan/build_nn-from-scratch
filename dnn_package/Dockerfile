FROM python:3.9-slim-bullseye
WORKDIR /app
RUN pip install --upgrade pip
COPY . .
RUN pip install -r requirments.txt
ENTRYPOINT ["python"]
RUN pip install .
RUN chmod +x app_4layer_dnn.py
CMD ["app_4layer_dnn.py"]
