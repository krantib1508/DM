import React from 'react'
import { useEffect, useState } from 'react';
import axios from "axios"

const Assignment4 = () => {

    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);


    
  useEffect(() => {

    axios.get('http://localhost:8000/assignment4')
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
          <h3>Rules:<br></br> {data.rules}</h3>
          <br />
          <h2><u>Accuracy: </u>  {data.accuracy}</h2>
          <h2><u>Coverage: </u>  {data.coverage}</h2>
          <h2><u>Toughness: </u>  {data.toughness}</h2>
          <br />



 

          





      </div>

    ) : (
      <p>No data available.</p>
    )}
  </div>
  )
}

export default Assignment4