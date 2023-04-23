// Recommendations.js
import React from 'react';
import Image from 'next/image';

const Recommendations = ({ recommendations, setRecommendations }: { recommendations: string[], setRecommendations: React.Dispatch<React.SetStateAction<string[]>> }) => {
    return (
        <div className="bg-white shadow-md p-6 rounded-md">
            <h2 className="text-2xl font-bold mb-4 text-black">Your Recommendations</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                {recommendations.map((recommendation, index) => (
                    <div key={index} className="relative aspect-w-1 aspect-h-1">
                        <img
                            src={`http://127.0.0.1:4006/show/${recommendation}`}
                            alt={`Recommendation ${index + 1}`}
                            className="rounded-md object-cover inset-0 w-full h-full"
                            width={300}
                            height={300}
                        />
                    </div>
                ))}
            </div>
            <button className={`mt-4 w-full bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline`} onClick={() => setRecommendations([])}>
                Clear Recommendations
            </button>
        </div>
    );
};


export default Recommendations;
