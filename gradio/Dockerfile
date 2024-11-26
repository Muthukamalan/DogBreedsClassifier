FROM python:3.9-slim-buster AS build
 
RUN apt-get update -y && apt install -y --no-install-recommends git build-essential\
    && pip install --no-cache-dir -U pip

COPY requirements.txt .
# Create the virtual environment.
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH

RUN pip install -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim-buster 
RUN apt-get update -y && apt install -y --no-install-recommends build-essential
COPY --from=build /venv /venv
ENV PATH=/venv/bin:$PATH

WORKDIR /code

COPY . .

EXPOSE 7860

CMD [ "python","app.py" ]