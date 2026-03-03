#!/usr/bin/env bash
# ============================================================
# deploy.sh — Build & deploy the Banana project in Minikube
# ============================================================
set -euo pipefail

# ── Resolve absolute path of this script's directory ────────
# BASH_SOURCE[0] is reliable even when called as ./deploy.sh,
# bash deploy.sh, or via a symlink.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colour helpers ───────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}▶ $*${NC}"; }
warn()  { echo -e "${YELLOW}⚠ $*${NC}"; }
error() { echo -e "${RED}✗ $*${NC}"; exit 1; }

# ── 0. Prerequisites ─────────────────────────────────────────
info "Checking prerequisites..."
command -v minikube >/dev/null 2>&1 || error "minikube not found"
command -v kubectl  >/dev/null 2>&1 || error "kubectl not found"
command -v docker   >/dev/null 2>&1 || error "docker not found"

info "Working directory: $SCRIPT_DIR"

# ── Pre-flight: verify Dockerfiles are real ──────────────────
# macOS tar extraction creates ghost '._Dockerfile' AppleDouble files.
# If the real Dockerfile is missing or corrupt (< 50 bytes), abort early.
check_dockerfile() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        error "Dockerfile not found: $path"
    fi
    local size
    size=$(wc -c < "$path" | tr -d ' ')
    if [[ "$size" -lt 50 ]]; then
        error "Dockerfile looks corrupt (only ${size} bytes): $path
       Likely a macOS AppleDouble '._' metadata file.
       Fix: run  find . -name '._*' -delete  then re-extract the archive."
    fi
    info "  OK: $path (${size} bytes)"
}

info "Checking Dockerfiles..."
check_dockerfile "$SCRIPT_DIR/Dockerfile"
check_dockerfile "$SCRIPT_DIR/mcp_servers/Dockerfile.market"
check_dockerfile "$SCRIPT_DIR/mcp_servers/Dockerfile.sec"
check_dockerfile "$SCRIPT_DIR/mcp_servers/Dockerfile.social"

# ── 1. Ensure Minikube is running ────────────────────────────
MINIKUBE_STATUS=$(minikube status --format='{{.Host}}' 2>/dev/null || echo "Stopped")
if [[ "$MINIKUBE_STATUS" != "Running" ]]; then
    warn "Minikube not running (status: ${MINIKUBE_STATUS}). Starting..."
    minikube start --cpus=4 --memory=6144 --disk-size=20g
else
    info "Minikube already running."
fi

# ── 2. Point Docker at Minikube's daemon ─────────────────────
info "Configuring Docker to use Minikube's daemon..."
DOCKER_ENV=$(minikube docker-env) \
    || error "Failed to get minikube docker-env — run: minikube status"
eval "$DOCKER_ENV"
info "Docker host: ${DOCKER_HOST:-}"

# ── 3. Build images (all use repo root as context) ───────────
# Absolute -f paths remove any ambiguity about the working directory.

info "Building banana-api:v2 ..."
docker build -f "$SCRIPT_DIR/Dockerfile" \
             -t banana-api:v2 \
             "$SCRIPT_DIR" \
    || error "Failed to build banana-api:v2"

info "Building banana-market:v1 ..."
docker build -f "$SCRIPT_DIR/mcp_servers/Dockerfile.market" \
             -t banana-market:v1 \
             "$SCRIPT_DIR" \
    || error "Failed to build banana-market:v1"

info "Building banana-sec:v1 ..."
docker build -f "$SCRIPT_DIR/mcp_servers/Dockerfile.sec" \
             -t banana-sec:v1 \
             "$SCRIPT_DIR" \
    || error "Failed to build banana-sec:v1"

info "Building banana-social-mcp:v1 ..."
docker build -f "$SCRIPT_DIR/mcp_servers/Dockerfile.social" \
             -t banana-social-mcp:v1 \
             "$SCRIPT_DIR" \
    || error "Failed to build banana-social-mcp:v1"

info "Images in Minikube's Docker registry:"
docker images | grep -E "banana|REPOSITORY" || true

# ── 4. Create Alpha Vantage secret ───────────────────────────
if [[ -z "${ALPHA_VANTAGE_API_KEY:-}" ]]; then
    warn "ALPHA_VANTAGE_API_KEY is not set — market data will fail."
    warn "Set it with: export ALPHA_VANTAGE_API_KEY=your_key_here"
    ALPHA_VANTAGE_API_KEY="REPLACE_ME"
fi
info "Creating alpha-vantage-secret..."
kubectl create secret generic alpha-vantage-secret \
    --from-literal=ALPHA_VANTAGE_API_KEY="${ALPHA_VANTAGE_API_KEY}" \
    --dry-run=client -o yaml | kubectl apply -f -

# ── 5. Apply manifests in dependency order ───────────────────
info "Applying ConfigMap..."
kubectl apply -f "$SCRIPT_DIR/banana-configmap.yaml"

info "Applying MCP manifests..."
kubectl apply -f "$SCRIPT_DIR/banana-market-deployment.yaml"
kubectl apply -f "$SCRIPT_DIR/banana-market-service.yaml"
kubectl apply -f "$SCRIPT_DIR/banana-sec-deployment.yaml"
kubectl apply -f "$SCRIPT_DIR/banana-sec-service.yaml"
kubectl apply -f "$SCRIPT_DIR/banana-social-deployment.yaml"
kubectl apply -f "$SCRIPT_DIR/banana-social-service.yaml"

info "Applying API manifests..."
kubectl apply -f "$SCRIPT_DIR/banana-api-deployment.yaml"
kubectl apply -f "$SCRIPT_DIR/banana-api-service.yaml"

# ── 6. Wait for rollouts ─────────────────────────────────────
info "Waiting for MCP pods..."
kubectl rollout status deployment/banana-market --timeout=120s \
    || { warn "banana-market not ready:"; kubectl logs -l app=banana-market --tail=20; }
kubectl rollout status deployment/banana-sec    --timeout=120s \
    || { warn "banana-sec not ready:";    kubectl logs -l app=banana-sec    --tail=20; }
kubectl rollout status deployment/banana-social --timeout=120s \
    || { warn "banana-social not ready:"; kubectl logs -l app=banana-social --tail=20; }

info "Waiting for API pod (first run downloads ML model — allow a few minutes)..."
kubectl rollout status deployment/banana-api --timeout=300s \
    || { warn "banana-api logs:"; kubectl logs -l app=banana-api --tail=40; error "API pod failed to start"; }

# ── 7. Show status ───────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ Deployment complete!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
kubectl get pods -o wide
echo ""
kubectl get services
echo ""
SERVICE_URL=$(minikube service banana-api-service --url 2>/dev/null || echo "unavailable")
echo -e "${GREEN}🔗 API: ${SERVICE_URL}/analyze${NC}"
echo ""
echo "Test:"
echo "  curl -s -X POST ${SERVICE_URL}/analyze \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"query\": \"What is AAPL stock price?\"}' | python3 -m json.tool"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
