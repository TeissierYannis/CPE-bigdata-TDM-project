import React, { useState, useEffect } from 'react';

const Loader = () => {
    const waitingSentences = [
        "Recommender system is analyzing...",
        "Crunching numbers in the matrix...",
        "Training the machine learning model...",
        "AI is processing your request...",
        "Optimizing the neural network...",
        "Analyzing data patterns...",
        "Searching for the perfect recommendation...",
        "Please wait, the algorithm is hard at work...",
        "Estimating the optimal parameters...",
        "Processing the input data...",
        "Refining the recommendation...",
        "Adjusting weights and biases...",
        "Backpropagation in progress...",
        "Evaluating the cost function...",
        "Running gradient descent...",
        "Fine-tuning the hyperparameters...",
        "Feature extraction in progress...",
        "Calculating the similarity scores...",
        "Filtering the most relevant items...",
        "Collaborative filtering underway...",
        "Content-based recommendations loading...",
        "Just a few more moments...",
        "The AI is pondering your preferences...",
        "Machine learning magic happening...",
        "Your personalized recommendations are almost ready...",
    ];


    const [randomWord, setRandomWord] = useState(
        waitingSentences[Math.floor(Math.random() * waitingSentences.length)]
    );

    useEffect(() => {
        const interval = setInterval(() => {
            setRandomWord(waitingSentences[Math.floor(Math.random() * waitingSentences.length)]);
        },Math.floor(Math.random() * 10000) + 2000);

        return () => clearInterval(interval); // Clean up the interval on component unmount
    }, [waitingSentences]);

    return (
        <div className="flex flex-col items-center justify-center space-y-4">
            <style>
                {`
          .loader {
            box-sizing: border-box;
            animation-duration: 1s;
            animation-iteration-count: infinite;
            animation-timing-function: linear;
          }
        `}
            </style>
            <div className="loader w-16 h-16 border-t-4 border-blue-500 border-solid rounded-full animate-spin"></div>
            <h2 className="text-xl font-bold">{randomWord}</h2>
        </div>
    );
};

export default Loader;
