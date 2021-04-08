FROM python:alpine
WORKDIR /app
COPY myproxy.py .
COPY requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "myproxy:app"]