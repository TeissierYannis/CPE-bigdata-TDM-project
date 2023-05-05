// components/Navbar.tsx
import React from 'react';
import Link from 'next/link';

const Navbar: React.FC = () => {

    const ENDPOINTS = [
        {name: 'Recommend', address: '/recommend'},
        {name: 'Status', address: '/status'},
        {name: 'Graphs', address: '/graphs'},
        {name: 'Upload', address: '/upload'},
    ];

    return (
        <nav className="bg-black p-4">
            <ul className="flex space-x-4 justify-center">
                {ENDPOINTS.map((endpoint) => (
                    <li key={endpoint.name}>
                        <Link href={endpoint.address}>
                            <span
                                className="text-white hover:bg-gray-700 px-3 py-2 rounded-md text-sm font-medium cursor-pointer">
                                {endpoint.name}
                            </span>
                        </Link>
                    </li>
                ))}
            </ul>
        </nav>
    );
};

export default Navbar;
