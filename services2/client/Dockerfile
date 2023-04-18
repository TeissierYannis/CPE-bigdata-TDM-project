# Install dependencies only when needed
FROM node:16-alpine AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app
COPY package.json ./
RUN yarn install

# Rebuild the source code only when needed
FROM node:16-alpine AS builder

WORKDIR /app

#COPY --from=deps /app/node_modules ./node_modules

COPY . .

# Remove .next
RUN rm -rf .next
# remove .vscode
RUN rm -rf .vscode
# remove node_modules
RUN rm -rf node_modules

RUN yarn build

# Production image, copy all the files and run next
FROM node:16-alpine AS runner
WORKDIR /app

ENV NODE_ENV production

RUN addgroup --system --gid 1001 nextgroup
RUN adduser --system --uid 1001 nextuser

COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json

# Automatically leverage output traces to reduce image size
# https://nextjs.org/docs/advanced-features/output-file-tracing
COPY --from=builder --chown=nextuser:nextgroup /app/.next/standalone ./
COPY --from=builder --chown=nextuser:nextgroup /app/.next/static ./.next/static

USER user

CMD ["node", "server.js"]