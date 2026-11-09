FROM ghcr.io/astral-sh/uv:0.9.26 AS uv
FROM python:3.14.2-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
