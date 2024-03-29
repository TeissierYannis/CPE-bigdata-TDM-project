import './globals.css'
import Navbar from '../app/components/navBar'

export const metadata = {
    title: 'Recommender system',
    description: 'Recommender system',
}

export default function RootLayout({children}: {
    children: React.ReactNode
}) {
    return (
        <html lang="en">
        <body>
        <Navbar/>
        <main className="p-8 flex flex-col items-center">{children}</main>
        </body>
        </html>
    )
}
