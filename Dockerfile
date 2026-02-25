FROM python:3.12-slim

WORKDIR /work


COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

RUN printf '%s\n' '#!/usr/bin/env sh' 'exec python -m clingo "$@"' > /usr/local/bin/clingo \
  && chmod +x /usr/local/bin/clingo

COPY . .

ENTRYPOINT ["/bin/bash"]