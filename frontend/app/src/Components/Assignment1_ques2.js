import React, { useEffect, useState } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './Assignment1_ques2.css';

const Assignment1_ques2 = () => {

    const quantiles = [0.25, 0.50, 0.75]; // Example quantile values
    const [val,setVal] = useState([]);
    const [val2,setVal2] = useState([]);
    const [qqData, setQQData] = useState([]);
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [scatter, setScatterData] = useState([]);


    const quantileData = {
        x: quantiles,
        y: val, // Replace 'values' with your data
        type: 'scatter',
        mode: 'markers',
        marker: {
          size: 8,
          color: 'blue',
        },
        name: 'Quantile Plot',
      };

      const histogramData = [
        {
          x: val,
          type: 'histogram',
          name: 'Frequency',
          marker: {
            color: 'rgba(75, 192, 192, 0.7)',
            line: {
              color: 'rgba(75, 192, 192, 1)',
              width: 2,
            },
          },
        },
      ];
    
      const layout = {
        title: "graph",
        xaxis: {
          title: "xlable",
        },
        yaxis: {
          title: 'Frequency',
        },
      };
    

      const scatterData = {
        x: val,
        y: val2,
        mode: 'markers',
        type: 'scatter',
        marker: { color: 'rgba(75, 192, 192, 0.7)' },
      };

      const layouts = {
        title: 'Scatter Plot',
        xaxis: { title: 'xLabel' },
        yaxis: { title: 'yLabel' },
      };


      const databox = [
        {
          y: val,
          type: 'box',
          boxpoints: 'all',
          jitter: 0.3,
          pointpos: -1.8,
          marker: { color: 'blue' },
        },
      ];

    
      const layout2 = {
        title: 'Box Plot Example',
        showlegend: false,
      };

    
  useEffect(() => {
    axios.get('http://localhost:8000/assignment1_que2/')
      .then((response) => {
        setData(response.data);
        setVal(response.data.values);
        setScatterData([scatterData]);
        setLoading(false);
        console.log(response.data.values2);
        setVal2(response.data.values2);
        setQQData([
            {
              x: quantiles,
              y: response.data.values.slice().sort((a, b) => a - b),
              mode: 'markers',
              type: 'scatter',
              name: 'Q-Q Plot',
            },
            {
              x: [0, 10],
              y: [0, 10],
              mode: 'lines',
              type: 'scatter',
              name: 'Ideal Line',
            },
          ]);
      })
      .catch((error) => {
        console.error('Axios error:', error);
        setLoading(false);
      });

         const quantiles = qqData
        .slice()
        .sort((a, b) => a - b)
        .map((value, index, array) => (index + 1) / array.length);


  }, []);


  return (
    <div>
      {loading ? (
        <p>Loading...</p>
      ) : data ? (
        <div className="container mt-5">
                <p>Range : {data.Range}</p>

      <table className="table table-striped">

        <thead>
          <tr>
            <th>Attribute</th>
            <th>Five</th>
          </tr>
        </thead>
        <tbody>

            
          { Object.keys(data.Five).map((attribute) => (
            <tr key={attribute}>
              <td>{attribute}</td>
              <td>{data.Five[attribute]}</td>
            </tr>
          ))}

          <br/>
          <br/>   

        </tbody>
      </table>


      <h2>Quantile Plot</h2>
      <Plot
        data={[quantileData]}
        layout={{
          title: 'Quantile Plot',
          xaxis: {
            title: 'Quantiles',
          },
          yaxis: {
            title: 'Data Values',
          },
        }}
      />
         <br />
        <h2>Quantile-Quantile (Q-Q) Plot</h2>

        
            <Plot
                data={qqData}
                layout={{
                title: 'Q-Q Plot',
                xaxis: { title: 'Theoretical Quantiles' },
                yaxis: { title: 'Sample Quantiles' },
                }}
            />
            <br />
            <h2>Histogram</h2>
            
                <Plot data={histogramData} layout={layout} />
                <br />
                <h2>Scatter Plot</h2>
                
                      <Plot data={scatterData} layout={layouts} />
                      <br />
                    <h2>Box Plot</h2>
                    
                      <Plot data={databox} layout={layout2} />

                      <br />

                    
        </div>
      ) : (
        <p>No data available.</p>
      )}
    </div>
  )
}

export default Assignment1_ques2