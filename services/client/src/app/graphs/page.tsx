"use client";
import React, {useEffect, useState} from 'react';
import axios from 'axios';
import Head from 'next/head';

const IMAGE_ROUTES = [
    '/visualize/graph/size/static',
    '/visualize/graph/size/dynamic',
    '/visualize/graph/year',
    '/visualize/graph/brand',
    '/visualize/graph/countries',
    '/visualize/graph/altitude',
    '/visualize/graph/dominant_color',
    '/visualize/graph/tags/dendrogram',
    '/visualize/graph/tags/top'
];
// TODO : get the map as an image
// const HTML_ROUTE = '/visualize/graph/map';

const TITLES = [
    'Static size graph',
    'Dynamic size graph',
    'Year graph',
    'Brand graph',
    'Countries graph',
    'Altitude graph',
    'Dominant colors graph',
    'Tags dendrogram',
    'Tags top'
];

const Graphs: React.FC = () => {
    const [images, setImages] = useState<string[]>([]);
    const [isLoading, setIsLoading] = useState<boolean>(true);
    const [htmlContent, setHtmlContent] = useState<string | null>(null);

    useEffect(() => {
        const fetchImages = async () => {
            const imagePromises = IMAGE_ROUTES.map((route) =>
                axios.get(`http://127.0.0.1:81${route}`, {responseType: 'arraybuffer'})
            );

            const responses = await Promise.all(imagePromises);
            const imageBuffers = responses.map((response) =>
                `data:image/png;base64,${Buffer.from(response.data, 'binary').toString('base64')}`
            );

            setImages(imageBuffers);
            setHtmlContent(responses[responses.length - 1].data);
            setIsLoading(false);
        };
        fetchImages();
    }, []);

    if (isLoading) {
        return (
            <div className="flex justify-center items-center h-screen">
                <Head>
                    <title>Visualisation des graphiques</title>
                </Head>
                <svg className="animate-spin -ml-1 mr-3 h-10 w-10 text-gray-900" xmlns="http://www.w3.org/2000/svg"
                     fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor"
                            strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor"
                          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <div>
                    Loading all graphs...
                </div>
            </div>
        );
    }

    return (
        <div className="container bg-black min-h-screen">
            <Head>
                <title>Images graphs</title>
            </Head>

            <main className="p-8">
                {images.map((src, index) => (
                    <div key={index} className="image-container mb-6 bg-white p-6 rounded-2xl shadow-md">
                        <h2 className="text-2xl sm:text-3xl md:text-4xl font-semibold mb-4 text-center text-black">{TITLES[index]}</h2>
                        <img src={src} alt={`Graph ${index + 1}`} className="w-4/5 mx-auto"/>
                    </div>
                ))}
            </main>
        </div>
    );
};


export default Graphs;