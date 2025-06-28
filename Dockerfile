FROM python:3.11-slim

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sample_comm_primitive.py .
COPY sample_train.py .
COPY data_parallel/ data_parallel/
COPY tensor_parallel/ tensor_parallel/
COPY pipeline_parallel/ pipeline_parallel/
COPY entrypoint.sh .

RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
# CMD ["sleep", "infinity"]
