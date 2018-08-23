FROM python:3.6
MAINTAINER Kosuke Futamata <matasuke.f@gmail.com>

ARG project_dir=/title2hist/

RUN apt-get update && apt-get install -y \
        git \
        wget \
        sudo \
        vim

COPY requirements.txt $project_dir
COPY inference.py $project_dir
COPY snapshot $project_dir
COPY source_vocab.ja $project_dir
COPY target_vocab.en $project_dir
COPY flask_app.py $project_dir

RUN mkdir templates
COPY templates/index.html $project_dir/templates/

WORKDIR $project_dir
RUN pip install -r requirements.txt

CMD ["python", "flask_app.py", "source_vocab.ja", "target_vocab.en", "snapshot"]
