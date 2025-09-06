FROM nvidia/cuda:12.1.0-cudnn8-devel-ubi8

# Set working directory
WORKDIR /workspace

# Copy startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Set entrypoint
ENTRYPOINT ["/start.sh"]