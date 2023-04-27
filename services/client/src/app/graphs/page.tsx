"use client";
import React, {useEffect, useState} from 'react';
import axios from 'axios';
import Loading from 'react-loading';
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

const HTML_ROUTE = '/visualize/graph/map';

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
    const [htmlContent, setHtmlContent] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(true);

    useEffect(() => {
        const fetchImages = async () => {
            const imagePromises = IMAGE_ROUTES.map((route) =>
                axios.get(`http://127.0.0.1:81${route}`, {responseType: 'arraybuffer'})
            );
            const htmlPromise = axios.get(`http://127.0.0.1:81${HTML_ROUTE}`, {responseType: 'text'});

            const responses = await Promise.all([...imagePromises, htmlPromise]);
            const imageBuffers = responses.slice(0, -1).map((response) =>
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
            <div className="flex justify-center items-center h-screen mr-4">
                <Head>
                    <title>Visualisation des graphiques</title>
                </Head>
                <Loading type="spin" color="#333"/>
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
                        <h2 className="text-2xl sm:text-3xl md:text-4xl font-semibold mb-4 text-center">{TITLES[index]}</h2>
                        <img src={src} alt={`Graph ${index + 1}`} className="w-4/5 mx-auto"/>
                    </div>
                ))}
                {htmlContent && (
                    <div className="map-container mb-6 bg-white p-6 rounded-2xl shadow-md">
                        <div className="map-content" dangerouslySetInnerHTML={{__html: htmlContent}}/>
                    </div>
                )}
            </main>
        </div>
    );
};

export default Graphs;