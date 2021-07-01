FROM python:3.9.4-slim-buster

ARG GEMFURY_TOKEN
ARG POSTGRES_PORT
ARG POSTGRESQL_DB_NAME
ARG POSTGRESQL_DB_USER
ARG POSTGRESQL_DB_PASSWORD
ARG POSTGRESQL_DB_HOST

ENV TOKEN=$GEMFURY_TOKEN
ENV POSTGRES_PORT=$POSTGRES_PORT
ENV POSTGRESQL_DB_NAME=$POSTGRESQL_DB_NAME
ENV POSTGRESQL_DB_USER=$POSTGRESQL_DB_USER
ENV POSTGRESQL_DB_PASSWORD=$POSTGRESQL_DB_PASSWORD
ENV POSTGRESQL_DB_HOST=$POSTGRESQL_DB_HOST

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=allisgood
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install \
        gcc \
        g++ \
        jq \
        curl \
        apt-transport-https \
        ca-certificates gnupg -y

WORKDIR /opt/


COPY service service
COPY configs configs
COPY setup.py setup.py
# COPY models models
# COPY generated generated

RUN pip install -e . --extra-index-url https://${TOKEN}@pypi.fury.io/cytora/
RUN python service/handler.py

EXPOSE 3000

CMD ["run-service"]
