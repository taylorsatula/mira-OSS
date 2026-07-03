# MIRA Application Image
# Thin layer on top of mira-base with just the application code.
# Rebuilds in seconds after code changes.
#
# Prerequisites: docker build -t mira-base:latest -f deploy/docker/Dockerfile.base .
# Build:         docker build -t mira:latest -f deploy/docker/Dockerfile .
# Run:           docker run -it -v mira-data:/opt/vault -p 1993:1993 mira:latest

ARG BASE_IMAGE=mira-base:latest
FROM ${BASE_IMAGE}

# Labels
LABEL org.opencontainers.image.title="MIRA"
LABEL org.opencontainers.image.description="MIRA - Memory-Integrated Reasoning Assistant"
LABEL org.opencontainers.image.source="https://github.com/taylorsatula/mira-OSS"

# Copy application code
COPY --chown=mira:mira . /opt/mira/app/

# Copy s6 service definitions
COPY deploy/docker/s6-rc.d/ /etc/s6-overlay/s6-rc.d/

# Copy configuration files
COPY deploy/docker/config/vault.hcl /opt/vault/config/vault.hcl
COPY deploy/docker/config/valkey.conf /etc/valkey/valkey.conf

# Copy scripts
COPY deploy/docker/scripts/init-mira.sh /init-mira.sh
COPY deploy/docker/scripts/container-setup.sh /opt/mira/container-setup.sh
COPY deploy/docker/scripts/healthcheck.sh /opt/mira/healthcheck.sh
COPY deploy/docker/scripts/vault-unseal.sh /opt/mira/app/deploy/docker/scripts/vault-unseal.sh
RUN chmod +x /init-mira.sh /opt/mira/container-setup.sh /opt/mira/healthcheck.sh \
    /opt/mira/app/deploy/docker/scripts/vault-unseal.sh \
    && chmod +x /etc/s6-overlay/s6-rc.d/mira/run \
                /etc/s6-overlay/s6-rc.d/vault/run \
                /etc/s6-overlay/s6-rc.d/valkey/run \
                /etc/s6-overlay/s6-rc.d/postgresql/run \
                /etc/s6-overlay/s6-rc.d/vault-init/up

# Expose ports
EXPOSE 1993

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD /opt/mira/healthcheck.sh

# Volumes for persistent data
VOLUME ["/var/lib/postgresql/17/main", "/opt/vault", "/var/lib/valkey", "/home/mira/.cache/huggingface"]

# Use our custom entrypoint that handles first-boot, then hands off to s6
ENTRYPOINT ["/init-mira.sh"]
