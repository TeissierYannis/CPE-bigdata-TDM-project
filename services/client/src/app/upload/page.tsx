"use client";
// Recommendations.js
import React from 'react';

const Page = ({}: {}) => {

    const [images, setImages] = React.useState<File[]>([]);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        // limit to 10 images
        if (event.target.files && event.target.files.length > 10) {
            alert('You can only upload a maximum of 10 images at once.');
            return;
        }
        if (event.target.files) {
            // filter images types
            const acceptedFileTypes = ['image/jpeg', 'image/png'];
            const filteredFiles = Array.from(event.target.files).filter((file: File) => acceptedFileTypes.includes(file.type));
            setImages([...images, ...filteredFiles]);
        }
    }

    const handleRemove = (index: number) => {
        setImages(images.filter((_, i) => i !== index));
    }

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        const formData = new FormData();

        images.forEach((image) => {
            formData.append('files[]', image);
        });

        try {
            const response = await fetch('http://127.0.0.1:81/gather/uploads', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (data.status === 'success') {
                alert(data.message);
                setImages([]);
            } else {
                alert(data.message);
            }
        } catch (error) {
            console.error('Error while uploading images:', error);
            alert('An error occurred while uploading the images. Please try again.');
        }
    }

    return (
        <>
            <h1>Upload new image for the recommender system</h1>
            <div className="col-span-full">
                <label htmlFor="cover-photo" className="block text-sm font-medium leading-6 text-white-900">Upload new
                    photo</label>
                <div
                    className="mt-2 flex justify-center rounded-lg border border-dashed border-white-900/25 px-6 py-10">
                    <div className="text-center">
                        <svg className="mx-auto h-12 w-12 text-white-300" viewBox="0 0 24 24" fill="currentColor"
                             aria-hidden="true">
                            <path fillRule="evenodd"
                                  d="M1.5 6a2.25 2.25 0 012.25-2.25h16.5A2.25 2.25 0 0122.5 6v12a2.25 2.25 0 01-2.25 2.25H3.75A2.25 2.25 0 011.5 18V6zM3 16.06V18c0 .414.336.75.75.75h16.5A.75.75 0 0021 18v-1.94l-2.69-2.689a1.5 1.5 0 00-2.12 0l-.88.879.97.97a.75.75 0 11-1.06 1.06l-5.16-5.159a1.5 1.5 0 00-2.12 0L3 16.061zm10.125-7.81a1.125 1.125 0 112.25 0 1.125 1.125 0 01-2.25 0z"
                                  clipRule="evenodd"/>
                        </svg>
                        <div className="mt-4 flex text-sm leading-6 text-white-600">
                            <label htmlFor="file-upload"
                                   className="relative cursor-pointer rounded-md bg-white font-semibold text-indigo-600 focus-within:outline-none focus-within:ring-2 focus-within:ring-indigo-600 focus-within:ring-offset-2 hover:text-indigo-500">
                                <span className="mx-1">Upload a file</span>
                                <input id="file-upload" name="file-upload" type="file" className="sr-only" multiple
                                       max={10}
                                       onChange={handleFileChange}/>
                            </label>
                            <p className="pl-1">or drag and drop</p>
                        </div>
                        <p className="text-xs leading-5 text-white-600">PNG, JPG, GIF up to 10MB</p>
                    </div>
                </div>

                <form onSubmit={handleSubmit}>
                    <div className="mt-4">
                        <button
                            className={`w-full bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline`}
                            type="submit">
                            Submit
                        </button>
                    </div>
                </form>

                <div className="mt-4 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                    {images.map((image, index) => (
                        <div key={index} className="relative aspect-w-1 aspect-h-1">
                            <img
                                src={URL.createObjectURL(image)}
                                alt={`Recommendation ${index + 1}`}
                                className="rounded-md object-cover inset-0 w-full h-full"
                                width={300}
                                height={300}
                            />
                            <button
                                className={`absolute top-0 right-0 mt-2 mr-2 bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline`}
                                onClick={() => handleRemove(index)}>
                                Remove
                            </button>
                        </div>
                    ))}
                </div>
            </div>
        </>
    );
};


export default Page;
