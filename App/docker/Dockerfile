FROM python:3.10

COPY Shared /common/Shared
COPY setup.py /common/
# create directory in the container for adding your files
WORKDIR /user/src

# copy over the requirements file and run pip install to install the packages into your container at the directory defined above
COPY App ./App
WORKDIR /user/src/App
RUN pip install --no-cache-dir -r requirements.txt --user 
RUN pip install -e /common/
# enter entry point parameters executing the container
EXPOSE 5057
# CMD [ "python","app.py" ]