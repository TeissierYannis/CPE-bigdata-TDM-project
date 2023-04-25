"use client";

import React, {useState} from 'react';
import Loader from "@/app/components/loader";
import Recommendations from "@/app/components/recommendations";

const axios = require('axios');
export default function Page() {
    const [step, setStep] = useState(1);
    const [isLoading, setIsLoading] = useState(false);
    const [recommendations, setRecommendations] = useState<string[]>([]);

    const [name, setName] = useState('');
    const [hexColor, setHexColor] = useState('');
    const [words, setWords] = useState([]);
    const [currentWord, setCurrentWord] = useState('');
    const [width, setWidth] = useState(0);
    const [height, setHeight] = useState(0);
    const [make, setMake] = useState('');

    const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        console.log(name, hexColor, words, width, height, make);

        // TODO
        const prefOrientation = 0;

        setIsLoading(true);
        const preferences = {
            dominant_color: hexColor,
            imagewidth: width,
            imageheight: height,
            orientation: prefOrientation,
            tags: words,
            make: make,
        };

        axios.post('http://127.0.0.1:81/recommend/recommend', {preferences})
            .then(function (response: any) {
                console.log(response.data);
                // store only values
                setRecommendations(Object.values(response.data));
            })
            .catch(function (error: any) {
                console.error(error);
            })
            .finally(function () {
                setIsLoading(false);
            });
    };

    const onNext = () => {
        setStep(step + 1);
    };

    const onPrevious = () => {
        setStep(step - 1);
    };

    // Update the handleWords function
    const handleWords = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            // if there is more than 5 words, don't add more
            if (words.length >= 5) return;
            // @ts-ignore
            setWords([...words, currentWord]);
            setCurrentWord('');
        }
    };

    return (
        <main className="flex min-h-screen flex-col items-center justify-between p-24">
            {isLoading ? (
                <Loader/>
            ) : recommendations.length === 0 ? (

                <div className="flex flex-col items-center justify-center">
                    <h1 className="text-6xl font-bold text-center mb-5">Recommender system</h1>
                    <form onSubmit={handleSubmit} className="flex flex-col space-y-4">
                        {step === 1 && (
                            <div className="flex flex-col space-y-2">
                                <label htmlFor="name">Name:</label>
                                <input
                                    className="text-black w-full p-3 border-2 border-blue-500 rounded-md focus:outline-none focus:border-blue-700 focus:ring-2 focus:ring-blue-300 shadow-md transition duration-200 ease-in-out"
                                    type="text"
                                    id="name"
                                    value={name}
                                    onChange={(e) => setName(e.target.value)}
                                />
                            </div>
                        )}
                        {step === 2 && (
                            <div className="flex space-y-2">
                                <label htmlFor="color" className="mt-auto mr-2">Color:</label>
                                <div className="relative w-16 h-8">
                                    <input
                                        type="color"
                                        id="color"
                                        value={hexColor}
                                        onChange={(e) => setHexColor(e.target.value)}
                                        className="w-full h-full opacity-0 absolute left-0 top-0 cursor-pointer"
                                    />
                                    <div
                                        className="w-full h-full bg-gradient-to-r from-blue-500 to-green-500 rounded-md shadow-md"
                                        style={{background: hexColor}}
                                    ></div>
                                </div>
                            </div>
                        )}
                        {step === 3 && (
                            <div className="flex flex-col space-y-2">
                                <label htmlFor="words">Words:</label>
                                <div className="relative">
                                    <input
                                        className="text-black w-full p-3 border-2 border-blue-500 rounded-md focus:outline-none focus:border-blue-700 focus:ring-2 focus:ring-blue-300 shadow-md transition duration-200 ease-in-out"
                                        type="text"
                                        id="words"
                                        value={currentWord}
                                        onChange={(e) => setCurrentWord(e.target.value)}
                                        onKeyDown={handleWords}
                                    />
                                    <div className="flex flex-wrap mt-2">
                                        {words.map((word, index) => (
                                            <div
                                                key={index}
                                                className="bg-blue-500 text-white px-3 py-1 mr-2 mb-2 rounded-full flex items-center space-x-2"
                                            >
                                                <span>{word}</span>
                                                <button
                                                    className="focus:outline-none"
                                                    onClick={() =>
                                                        setWords(words.filter((_, wordIndex) => wordIndex !== index))
                                                    }
                                                >
                                                    <span className="font-bold">&times;</span>
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}
                        {step === 4 && (
                            <div className="flex flex-col space-y-2">
                                <label htmlFor="width">Width:</label>
                                <input
                                    className="text-black w-full p-3 border-2 border-blue-500 rounded-md focus:outline-none focus:border-blue-700 focus:ring-2 focus:ring-blue-300 shadow-md transition duration-200 ease-in-out"
                                    type="number"
                                    min="0"
                                    id="width"
                                    value={width}
                                    onChange={(e) => setWidth(parseInt(e.target.value))}
                                />
                                <label htmlFor="height">Height:</label>
                                <input
                                    className="text-black w-full p-3 border-2 border-blue-500 rounded-md focus:outline-none focus:border-blue-700 focus:ring-2 focus:ring-blue-300 shadow-md transition duration-200 ease-in-out"
                                    type="number"
                                    min="0"
                                    id="height"
                                    value={height}
                                    onChange={(e) => setHeight(parseInt(e.target.value))}
                                />
                            </div>
                        )}
                        {step === 5 && (
                            <div className="flex flex-col space-y-2">
                                <label htmlFor="make">Make:</label>
                                <input
                                    className="text-black w-full p-3 border-2 border-blue-500 rounded-md focus:outline-none focus:border-blue-700 focus:ring-2 focus:ring-blue-300 shadow-md transition duration-200 ease-in-out"
                                    type="text"
                                    id="make"
                                    value={make}
                                    onChange={(e) => setMake(e.target.value)}
                                />
                            </div>
                        )}
                        <div className="flex space-x-4">
                            {step > 1 && (
                                <button type="button" className="bg-blue-500 text-white px-4 py-2 rounded"
                                        onClick={onPrevious}>
                                    Previous
                                </button>
                            )}
                            {step < 5 && (
                                <button type="button" className="bg-green-500 text-white px-4 py-2 rounded"
                                        onClick={onNext}>
                                    Next
                                </button>
                            )}
                            {step === 5 &&
                                <button type="submit"
                                        className="bg-green-500 text-white px-4 py-2 rounded">Submit</button>}
                        </div>
                    </form>
                </div>

            ) : (
                <>
                    {recommendations.length > 0 && (
                        <div className="w-full max-w-4xl mt-8">
                            <Recommendations recommendations={recommendations} setRecommendations={setRecommendations}/>
                        </div>
                    )}
                </>
            )}
        </main>
    );
}