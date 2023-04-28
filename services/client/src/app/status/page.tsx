"use client";
import React, {useEffect, useState} from 'react';

interface Endpoint {
    name: string;
    url: string;
}

const endpoints: Endpoint[] = [
    {name: 'Recommend', url: 'http://127.0.0.1:81/recommend/api/v1/health'},
    {name: 'Visualize', url: 'http://127.0.0.1:81/visualize/api/v1/health'},
    {name: 'Gather', url: 'http://127.0.0.1:81/gather/api/v1/health'},
    {name: 'CDN', url: 'http://127.0.0.1:81/cdn/api/v1/health'},
];

const fetchStatus = async (url: string) => {
    try {
        const response = await fetch(url);
        const data = await response.json();
        return data.status;
    } catch (error) {
        return 'error';
    }
};

const StatusPage: React.FC = () => {
    const [statuses, setStatuses] = useState<{ [key: string]: string }>({});

    useEffect(() => {
        const getStatuses = async () => {
            const newStatuses: { [key: string]: string } = {};
            for (const endpoint of endpoints) {
                newStatuses[endpoint.name] = await fetchStatus(endpoint.url);
            }
            setStatuses(newStatuses);
        };

        getStatuses();
    }, []);

    return (
        <div className="min-h-screen flex flex-col items-center justify-start py-8">
            <h1 className="sm:text-3xl md:text-5xl lg:text-5xl font-bold mb-10">Endpoints status</h1>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {endpoints.map((endpoint) => (
                    <div
                        key={endpoint.name}
                        className="bg-white rounded-lg sm:p-6 md:p-8 lg:p-10 shadow-md"
                    >
                        <h2 className="text-lg font-bold text-center mb-4 text-black">
                            {endpoint.name}
                        </h2>
                        <div className="flex justify-center">
                            {statuses[endpoint.name] === undefined && (
                                <svg className="animate-spin h-5 w-5 text-yellow-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.292A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.646z"></path>
                                </svg>
                            )}
                            {statuses[endpoint.name] === 'ok' && (
                                <img
                                    src="/icons/check.svg"
                                    alt="Check Icon"
                                    className="text-green-500 h-6 w-6"
                                />
                            )}
                            {statuses[endpoint.name] === 'error' && (
                                <img
                                    src="/icons/cross.svg"
                                    alt="Cross Icon"
                                    className="text-red-500 h-6 w-6"
                                />
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default StatusPage;
