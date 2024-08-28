# Python runtime as a image
FROM python:3.10
# install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
#Mounts the application code to the image andexpose 8post 8000
COPY . code
WORKDIR /code
EXPOSE 8000
# runs the production server
ENTRYPOINT ["python", "manage.py"]
CMD ["runserver", "0.0.0.0:8000"]