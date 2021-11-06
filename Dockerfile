FROM nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Create app directory and move into it
WORKDIR /usr/src/app

# Copy all files into container
COPY . .

# Install python modules
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Expose port
EXPOSE 80

# Run the app
CMD uvicorn main:app --host 0.0.0.0 --port 80
