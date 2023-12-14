import React from 'react'
import { useEffect, useState } from 'react';
import axios from "axios"


const Assignment2 = () => {


    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);


    
  useEffect(() => {
    axios.get('http://localhost:8000/assignment2')
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
          
          
            <h2><u>File : {data.name}</u></h2>
            <br />
                    
            <h3><u>result:</u> {data.result}</h3>
            <br />
            <h3><u>p:</u> {data.p}</h3>
            <br />
            <h3><u>chi2:</u> {data.chi2}</h3>
            <br />
            <h3><u>dof:</u> {data.dof}</h3>
            <br />




        
        </div>

      ) : (
        <p>No data available.</p>
      )}
    </div>
  )
}

export default Assignment2