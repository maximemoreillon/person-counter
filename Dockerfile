FROM dustynv/jetson-inference:r32.6.1

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1


# Download model
WORKDIR /jetson-inference/data/networks
RUN wget -q -c  https://nvidia.box.com/shared/static/jcdewxep8vamzm71zajcovza938lygre.gz -O - | tar -xz

# Application code
WORKDIR /usr/src/app
COPY . .

RUN pip3 install -r requirements.txt

# Initialize the network using TensorRT
RUN python3 init.py

EXPOSE 80

CMD uvicorn main:app --host 0.0.0.0 --port 80
