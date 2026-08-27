FROM node:22-bookworm-slim@sha256:d649c27dae7ba0137b3cef5dd75baa422c08dc3d9e3fc0c23dfb172dc3cc6436

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

COPY deploy/server/calendar-mcp/package.json deploy/server/calendar-mcp/package-lock.json /app/calendar-mcp/
RUN cd /app/calendar-mcp && npm ci --omit=dev --no-audit --no-fund

COPY vendor/telegram-mcp /app/vendor/telegram-mcp
RUN /opt/venv/bin/pip install --no-cache-dir /app/vendor/telegram-mcp

COPY pyproject.toml uv.lock /app/
RUN python3 -m venv /opt/uv-bootstrap \
    && /opt/uv-bootstrap/bin/pip install --no-cache-dir uv==0.11.29 \
    && /opt/uv-bootstrap/bin/uv export \
        --locked \
        --no-dev \
        --no-emit-project \
        --format requirements.txt \
        --output-file /tmp/requirements.txt \
    && /opt/venv/bin/pip install --no-cache-dir --require-hashes \
        --requirement /tmp/requirements.txt \
    && rm -rf /opt/uv-bootstrap /tmp/requirements.txt

# RapidOCR otherwise downloads non-default language weights on first use. Bake
# the exact ONNX PP-OCRv5 detector, orientation classifier, and both Russian
# recognition variants into the image so the read-only runtime needs no model
# network or writable HOME.
RUN mkdir -p /opt/rapidocr-models \
    && /opt/venv/bin/python -c 'from rapidocr import EngineType, LangCls, LangDet, LangRec, ModelType, OCRVersion, RapidOCR; common = {"Global.model_root_dir": "/opt/rapidocr-models", "Global.log_level": "critical", "Global.use_cls": True, "Det.engine_type": EngineType.ONNXRUNTIME, "Det.lang_type": LangDet.CH, "Det.model_type": ModelType.MOBILE, "Det.ocr_version": OCRVersion.PPOCRV5, "Cls.engine_type": EngineType.ONNXRUNTIME, "Cls.lang_type": LangCls.CH, "Cls.model_type": ModelType.MOBILE, "Cls.ocr_version": OCRVersion.PPOCRV5, "Rec.engine_type": EngineType.ONNXRUNTIME, "Rec.model_type": ModelType.MOBILE, "Rec.ocr_version": OCRVersion.PPOCRV5}; [RapidOCR(params={**common, "Rec.lang_type": language}) for language in (LangRec.ESLAV, LangRec.CYRILLIC)]' \
    && test -s /opt/rapidocr-models/ch_PP-OCRv5_det_mobile.onnx \
    && test -s /opt/rapidocr-models/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx \
    && test -s /opt/rapidocr-models/eslav_PP-OCRv5_rec_mobile.onnx \
    && test -s /opt/rapidocr-models/cyrillic_PP-OCRv5_rec_mobile.onnx \
    && chmod -R a-w /opt/rapidocr-models

COPY README.md /app/
COPY src /app/src
RUN /opt/venv/bin/pip install --no-cache-dir --no-deps /app

COPY deploy/server/telegram_gateway.py deploy/server/telegram-gateway /app/deploy/server/
RUN chmod 0555 /app/deploy/server/telegram-gateway \
    && mkdir -p /app/.runtime /tmp/telegram-gateway-files \
    && chown -R 1000:1000 /app/.runtime /tmp/telegram-gateway-files

USER 1000:1000
EXPOSE 8080

ENTRYPOINT ["/opt/venv/bin/tg-voice-transcriber-bot"]
