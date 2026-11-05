# The CPU build contains no CUDA backend. Digest also freezes native libraries.
FROM vllm/vllm-openai-cpu:v0.27.0@sha256:16981b40bb538e9127ad82e74a9f5386fddb4071c141725309d7ca4e2ea5872e

WORKDIR /opt/ml
COPY requirements.txt tools/check_lock.py ./
# Dependencies are already present in this digest-pinned CPU image. Check ALL
# distributions, including transitive/build dependencies; never resolve at build.
RUN python check_lock.py requirements.txt && python -m pip check \
    && groupadd --gid 10001 inference \
    && useradd --uid 10001 --gid 10001 --no-create-home inference \
    && mkdir -p /models && chown 10001:10001 /models
COPY app ./app
ENV PATH="/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    PYTHONPATH=/opt/ml PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp HF_HOME=/tmp/huggingface VLLM_CACHE_ROOT=/tmp/vllm \
    XDG_CACHE_HOME=/tmp/cache TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
    VLLM_CPU_KVCACHE_SPACE=1 VLLM_CPU_OMP_THREADS_BIND=nobind OMP_NUM_THREADS=4
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["python", "/opt/ml/app/serve.py"]
