FROM python:3.9.4-slim

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /opt/

COPY run-component-tests.sh run-component-tests.sh
COPY tests tests
COPY requirements.txt requirements.txt

RUN chmod +x run-component-tests.sh
RUN pip install -r requirements.txt

CMD [ "/opt/run-component-tests.sh" ]
