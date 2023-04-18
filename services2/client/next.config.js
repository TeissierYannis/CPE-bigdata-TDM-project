/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  images: {
    domains: ['127.0.0.1:81'],
    remotePatterns: [
      {
        protocol: 'http',
        hostname: '127.0.0.1',
        port: "81",
        pathname: '/gather/show/**',
      }
    ],
  }
}

module.exports = nextConfig
