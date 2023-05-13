"use client";
import React, {useEffect, useState} from 'react';
import axios from 'axios';
import Head from 'next/head';

const IMAGE = [
    {route: '/visualize/graph/size/static', title: 'Static size graph',},
    {route: '/visualize/graph/size/dynamic', title: 'Dynamic size graph',},
    {route: '/visualize/graph/year', title: 'Year graph',},
    {route: '/visualize/graph/brand', title: 'Brand graph',},
    {route: '/visualize/graph/countries', title: 'Countries graph',},
    {route: '/visualize/graph/altitude', title: 'Altitude graph',},
    {route: '/visualize/graph/dominant_color', title: 'Dominant colors graph',},
    {route: '/visualize/graph/tags/dendrogram', title: 'Tags dendrogram',},
    {route: '/visualize/graph/tags/top', title: 'Tags top'},
];

const Graphs: React.FC = () => {
    const [images, setImages] = useState<string[]>(Array(IMAGE.length).fill(null));
    const [isLoading, setIsLoading] = useState<boolean[]>(Array(IMAGE.length).fill(true));

    const resetMetadata = async () => {
        try {
            await axios.get('http://127.0.0.1:81/visualize/reset');
            alert('Les métadonnées ont été réinitialisées avec succès');
        } catch (error) {
            alert('Une erreur s\'est produite lors de la réinitialisation des métadonnées');
        }
    };

    useEffect(() => {
        const fetchImages = async () => {
            const imagePromises = IMAGE.map(async (image, index) => {
                const response = await axios.get(`http://127.0.0.1:81${image.route}`, {responseType: 'arraybuffer'});
                const imageBuffer = `data:image/png;base64,${Buffer.from(response.data, 'binary').toString('base64')}`;

                setImages(prevImages => {
                    const newImages = [...prevImages];
                    newImages[index] = imageBuffer;
                    return newImages;
                });

                setIsLoading(prevLoading => {
                    const newLoading = [...prevLoading];
                    newLoading[index] = false;
                    return newLoading;
                });
            });

            await Promise.all(imagePromises);
        };
        fetchImages();
    }, []);

    return (
        <div className="container bg-none min-h-screen justify-center">
            <Head>
                <title>Images graphs</title>
            </Head>

            <main className="p-8 justify-center">
                <div className={'flex justify-center items-center py-2 px-4 mb-6 bg-white text-black font-semibold rounded'}>
                    <button
                        onClick={resetMetadata}
                    >
                        Reset metadata
                    </button>
                    <img
                        src="/icons/refresh-cw.svg"
                        alt="Refresh"
                        className="h-4 w-4 ml-3"
                    />
                </div>

                {IMAGE.map((image, index) => (
                    <div key={index} className="image-container mb-6 bg-white p-6 rounded-2xl shadow-md">
                        <h2 className="text-2xl sm:text-3xl md:text-4xl font-semibold mb-4 text-center text-black">{image.title}</h2>
                        {isLoading[index] ? (
                            <div className="flex justify-center items-center">
                                <svg className="animate-spin -ml-1 mr-3 h-10 w-10 text-black"
                                     xmlns="http://www.w3.org/2000/svg"
                                     fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor"
                                            strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor"
                                          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                            </div>
                        ) : (
                            <img src={images[index]} alt={`Graph ${index + 1}`} className="w-4/5 mx-auto"/>
                        )}
                    </div>
                ))}
                <div className="image-container mb-6 bg-white p-6 rounded-2xl shadow-md">
                    <h2 className="text-2xl sm:text-3xl md:text-4xl font-semibold mb-4 text-center text-black">Images
                        map</h2>
                    <div className="flex justify-center items-center">
                        <iframe
                            title="JupyterLab"
                            src="http://127.0.0.1:81/visualize/graph/map"
                            className="w-full h-80"
                        />
                    </div>
                </div>
            </main>
        </div>
    );
};
export default Graphs;