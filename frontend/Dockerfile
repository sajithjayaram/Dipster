# Frontend Dockerfile (multi-stage: build with Node, serve with Nginx)
FROM node:20-alpine AS builder
WORKDIR /app

# Install deps
COPY frontend/package.json frontend/yarn.lock frontend/craco.config.js* /app/
# copy config files if exist (tailwind/postcss)
COPY frontend/tailwind.config.js frontend/postcss.config.js /app/
RUN yarn install --frozen-lockfile

# Copy source
COPY frontend/ /app/

# Build-time backend URL (passed via build args)
ARG REACT_APP_BACKEND_URL
ENV REACT_APP_BACKEND_URL=${REACT_APP_BACKEND_URL}

# Build static assets
RUN yarn build

# Serve with nginx
FROM nginx:1.27-alpine AS frontend

# Copy build
COPY --from=builder /app/build /usr/share/nginx/html

# Use hardened nginx config
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Optional: small nginx config improvements
RUN sed -i 's/sendfile\s\s*on;/sendfile on;\n    gzip on;\n    gzip_types text\/css application\/javascript application\/json text\/javascript text\/plain;/' /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
