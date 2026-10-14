FROM python:3.12-slim
# AWS Lambda Web Adapter: lets the same image run on Lambda; harmless elsewhere
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter
WORKDIR /srv
COPY requirements.serve.txt .
RUN pip install --no-cache-dir -r requirements.serve.txt
COPY app app
COPY models models
ENV PORT=8000 LOG_PATH=/tmp/predictions.jsonl
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
