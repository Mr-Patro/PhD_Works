FROM python:3.10-slim

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 git && rm -rf /var/lib/apt/lists/*

COPY --chown=user:user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY --chown=user:user . /app
RUN mkdir -p /app/models_deploy && chown -R user:user /app

USER user
EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
