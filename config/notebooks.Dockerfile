FROM leandatascience/configcredential

RUN pip install --upgrade pip
RUN apt-get update && \
    apt-get install -y git

RUN pip install numpy
RUN pip install scipy
RUN pip install sklearn
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install pandas
RUN pip install pivottablejs
RUN pip install nltk
RUN pip install pycorenlp
#RUN pip install stanza
#RUN pip install google
#RUN pip install protobuf
#RUN pip install stanfordcorenlp
#RUN pip install PyStanfordDependencies

RUN pip install plotly==3.4.0
RUN pip install "notebook>=5.3" "ipywidgets>=7.2"
RUN pip install jupyterlab==0.35

RUN export NODE_OPTIONS=--max-old-space-size=4096 # this takes a long time, installs yarn, webpack etc ...
#RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager@0.38
RUN jupyter labextension install plotlywidget@0.5.0
RUN jupyter labextension install @jupyterlab/plotly-extension@0.18
RUN jupyter labextension install @jupyterlab/latex

ENV MAIN_PATH=/usr/local/bin/notebooks
ENV LIBS_PATH=${MAIN_PATH}/libs
ENV CONFIG_PATH=${MAIN_PATH}/config
ENV NOTEBOOK_PATH=${MAIN_PATH}/notebooks

EXPOSE 8888

CMD cd ${MAIN_PATH} && sh config/run_jupyter.sh
