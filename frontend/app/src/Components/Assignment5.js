import React from 'react'
import { useEffect, useState } from 'react';
import axios from "axios"
import image1 from "./static/ANNplot.png"
import image2 from "./static/KNNplot.png"

const Assignment5 = () => {

    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);


    
  useEffect(() => {

    axios.get('http://localhost:8000/assignment5')
      .then((response) => {
        setData(response.data);
        setLoading(false);
      })
      .catch((error) => {
        console.error('Axios error:', error); 
        setLoading(false);
      });
  }, []);

  return (
    <div>
      {loading ? (
        <>
        <p>Loading...</p>
        </>
      ) : data ? (
        <div className="container mt-5">
          
          
            <h1><u>File : {data.name}</u></h1>
           
              <h3>ANN</h3>
            <img src={image1} alt="no_image">

            </img>

           <h3>KNN</h3>
            <img src={image2} alt="no_image">

            </img>  
        </div>

      ) : (
        <p>No data available.</p>
      )}
    </div>
  )
}

export default Assignment5