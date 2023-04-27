"use client";
import React, { useEffect, useState } from 'react';
import axios from 'axios';

const DockerStatus = () => {
    const [statuses, setStatuses] = useState([]);

    useEffect(() => {
        const fetchStatuses = async () => {
            const routes = [
                'http://127.0.0.1:81/recommend/api/v1/health',
                'http://127.0.0.1:81/visualize/api/v1/health',
                'http://127.0.0.1:81/gather/api/v1/health',
                'http://127.0.0.1:81/cdn/api/v1/health',
            ];

            const statusResponses = await Promise.all(routes.map(route => axios.get(route)));

            const newStatuses = statusResponses.map(response => ({
                route: response.config.url,
                status: response.data.status === 'ok' ? 'OK' : 'Error',
            }));

            // setStatuses(newStatuses);
        };

        fetchStatuses();
    }, []);

    return (
        <div>
            <h1>Docker Status</h1>
            {statuses.map((status, index) => (
                <div key={index}>
                    <h2>Route: </h2>
                    <p>Status: </p>
                </div>
            ))}
        </div>
    );
};

export default DockerStatus;
