# ═══════════════════════════════════════════════════════════════════
# Dockerfile
# ═══════════════════════════════════════════════════════════════════
# IMP: python:3.10-slim — 3.12 nahi
#      TensorFlow 2.15 ka 3.12 pe wheel nahi milta
#      Car Price mein bhi same issue tha — 3.10 fix tha
# ═══════════════════════════════════════════════════════════════════

FROM python:3.10-slim

# working directory
WORKDIR /app

# ── STEP 1: system deps ───────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
# gcc/g++ → TensorFlow C extensions compile karne ke liye

# ── STEP 2: requirements install ─────────────────────────────────
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt
# IMP: requirements-prod.txt → pinned versions
#      local requirements.txt se alag — Docker mein strict versions

# ── STEP 3: code copy ─────────────────────────────────────────────
COPY . .
# .dockerignore file decide karta hai kya copy nahi hoga

# ── STEP 4: port expose ───────────────────────────────────────────
EXPOSE 8080

# ── STEP 5: run ───────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
# IMP: --host 0.0.0.0 → container ke bahar accessible
#      localhost nahi → container ke andar hi rehta