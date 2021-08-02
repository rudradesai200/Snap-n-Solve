FROM tensorflow/tensorflow:2.1.1

# install nginx
RUN apt-get update && apt-get install nginx python3-dev python3-pip python3-venv ffmpeg libsm6 libxext6 -y --no-install-recommends 
COPY nginx.default /etc/nginx/sites-available/default
RUN ln -sf /dev/stdout /var/log/nginx/access.log \
    && ln -sf /dev/stderr /var/log/nginx/error.log

# copy source and install dependencies
RUN mkdir -p /opt/app
COPY requirements.txt start_server.sh /opt/app/
COPY mldeployed /opt/app/mldeployed

WORKDIR /opt/app
RUN pip install -r requirements.txt
RUN chown -R www-data:www-data /opt/app

# start server
EXPOSE 8020
STOPSIGNAL SIGTERM
CMD ["/opt/app/start_server.sh"]