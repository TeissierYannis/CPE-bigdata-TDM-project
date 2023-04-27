// pages/DockerStatus.tsx
"use client";
import React from 'react';
import axios from 'axios';

const fetchStatus = async (url: string) => {
    const {data} = await axios.get(url);
    return data;
};

const DockerStatus: React.FC = () => {
    const urls = [
        '/recommend/api/v1/health',
        '/visualize/api/v1/health',
        '/gather/api/v1/health',
        '/cdn/api/v1/health',
    ];

    return (
        <div>
            <h1>Statut des services Docker</h1>
        </div>
    );
};

export default DockerStatus;
