import './globals.css'

export const metadata = {
  title: 'Recommender system',
  description: 'Recommender system',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
