"use client";

import {Inter} from 'next/font/google'
import React from "react";
import {useState} from "react";

const inter = Inter({subsets: ['latin']})

export default function Home() {

    const [status, setStatus] = useState("");
    const [message, setMessage] = useState("");

    const handleInit = async (event: any) => {
        event.preventDefault();

        try {
            const response = await fetch('http://127.0.0.1:81/gather/download', {
                method: 'GET'
            });

            // Poll the status endpoint every second
            const intervalId = setInterval(async () => {
                const res = await fetch("http://127.0.0.1:81/gather/status");
                const data = await res.json();
                setStatus(data.status);
                setMessage(data.message);

                // If the status is not 'in progress', stop the polling
                if (data.status !== "in progress") {
                    clearInterval(intervalId);
                }
            }, 1000);
        } catch (error) {
            console.error('Error while starting download:', error);
            alert('An error occurred while starting the download. Please try again.');
        }
    }

    const getPercent = () => {
        // from {
        //     "message": "16 of 100 images downloaded",
        //     "status": "in progress"
        // }

        // to progress = 16 and total = 100
        const [progress, total] = message.split(" of ");
        // as int
        const progressInt = parseInt(progress);
        const totalInt = parseInt(total);

        return {
            progress: progressInt,
            total: totalInt,
        }
    }

    return (
        <div className="flex min-h-screen flex-col items-center justify-between p-24">
            <h1 className="text-6xl font-bold text-center mb-5">Welcome to the Recommendation System!</h1>
            <p className="mb-4">This project is realized within the framework of the study of massive data, in 4th year
                of engineering at CPE Lyon. In this project, we aimed to develop a comprehensive image recommendation
                system based on the Unsplash image dataset, taking into account user preferences.</p>
            <p className="mb-4">The objective of this interface is to allow you to simply exploit some features, such as
                :</p>
            <ul className="list-disc list-inside mb-4">
                <li>The launch of the data download and the visualization of the progress.</li>
                <li>The creation of a user profile.</li>
                <li>The recommendation of images.</li>
                <li>The addition of images to our dataset.</li>
                <li>The graphic visualization of the data from the dataset.</li>
                <li>Visualization of the states of our containers.</li>
                <li>At the following link (localhost:8888), the possibility to exploit our database with a Jupyter
                    Lab.
                </li>
            </ul>
            <p className="mb-4 text-red-500">Please note that the download of the dataset can take up to 10 minutes.</p>
            <p className="mb-4">To start, please click on the button below.</p>
            <button
                onClick={handleInit}
                className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
            >
                Init App
            </button>
            {status && (
                // progress bar
                <div className="mt-4">
                    <p className="text-center">{message}</p>
                    <div className="relative pt-1">
                        <div className="flex mb-2 items-center justify-between">
                            <div>
                                <span
                                    className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-blue-600 bg-blue-200">
                                    {status}
                                </span>
                            </div>
                            <div className="text-right mr-1">
                                <span className="text-xs font-semibold inline-block text-blue-600">
                                    {message}
                                </span>
                            </div>
                        </div>
                        <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-blue-200">
                            <div
                                style={{width: `${(getPercent().progress / getPercent().total) * 100}%`}}
                                className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-blue-500"
                            />
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
