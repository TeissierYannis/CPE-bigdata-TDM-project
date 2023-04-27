"use client";
import React from 'react';

const Page = ({}: {}) => {

    return (
        <div style={{ width: '100%', height: '100%' }}>
            <iframe
                title="JupyterLab"
                src="http://localhost:5005/notebook"
                width="100%"
                height="100%"
                style={{ border: 'none' }}
            />
        </div>
    );
}

export default Page;