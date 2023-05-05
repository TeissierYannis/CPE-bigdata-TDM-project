# Utilisez une image de base Jupyter Lab avec Python 3.8
FROM jupyter/base-notebook:python-3.8.8

# Copiez le fichier index.ipynb dans le répertoire de travail
COPY index.ipynb /home/jovyan/work/

# Installez les bibliothèques nécessaires
COPY requirements.txt /tmp/
RUN pip install --trusted-host pypi.python.org -r /tmp/requirements.txt

# Exposez le port 8888 pour le serveur Jupyter Lab
EXPOSE 8888

# Démarrez le serveur Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
