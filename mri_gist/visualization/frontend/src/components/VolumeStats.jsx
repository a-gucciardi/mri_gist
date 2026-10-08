import React, { useMemo, useState, useEffect } from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
);

const VolumeStats = ({ currentFile }) => {
    const [statsData, setStatsData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const pollAnalyticsJob = async (jobId) => {
        return new Promise((resolve, reject) => {
            const checkStatus = async () => {
                try {
                    const response = await fetch(`/api/analytics/${jobId}`);
                    if (!response.ok) throw new Error("Failed to fetch job status");
                    const data = await response.json();

                    if (data.status === 'completed') {
                        resolve(data.results);
                    } else if (data.status === 'failed') {
                        reject(new Error("Analytics job failed"));
                    } else {
                        setTimeout(checkStatus, 1000);
                    }
                } catch (e) {
                    reject(e);
                }
            };
            checkStatus();
        });
    };

    useEffect(() => {
        if (!currentFile) return;

        const fetchStats = async () => {
            setLoading(true);
            setError(null);
            try {
                // Submit Job
                const response = await fetch('/api/analytics', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        input_file: currentFile, // Assuming backend resolves name to path, or we need absolute path logic as discussed
                        analysis_type: 'tissue_distribution',
                        params: { threshold: 50 } // Example param
                    })
                });

                if (!response.ok) throw new Error("Failed to submit analytics job");
                const submitData = await response.json();

                // Poll
                const results = await pollAnalyticsJob(submitData.job_id);
                setStatsData(results);

            } catch (e) {
                console.error("Analytics error:", e);
                setError(e.message);
            } finally {
                setLoading(false);
            }
        };

        fetchStats();
    }, [currentFile]);

    const options = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
            },
            title: {
                display: true,
                text: 'Volume Statistics',
            },
        },
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'Voxel Count'
                }
            }
        }
    };

    const data = useMemo(() => {
        if (!statsData || !statsData.tissue) {
            return {
                labels: ['Background', 'Tissue'],
                datasets: [{
                    label: 'Voxel Count',
                    data: [0, 0],
                    backgroundColor: 'rgba(200, 200, 200, 0.5)'
                }]
            };
        }

        return {
            labels: ['Background', 'Tissue'], // Simplified based on available backend analytics
            datasets: [
                {
                    label: 'Voxel Count',
                    data: [statsData.background.voxel_count, statsData.tissue.voxel_count],
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.5)',
                        'rgba(255, 99, 132, 0.5)',
                    ],
                    borderColor: [
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 99, 132, 1)',
                    ],
                    borderWidth: 1,
                },
            ],
        };
    }, [statsData]);

    if (!currentFile) return <div style={{ color: 'white', padding: '20px' }}>Please select a file to view statistics.</div>;
    if (loading) return <div style={{ color: 'white', padding: '20px' }}>Analyzing volume...</div>;
    if (error) return <div style={{ color: 'red', padding: '20px' }}>Error: {error}</div>;



    return (
        <div style={{
            position: 'absolute',
            bottom: '10px',
            right: '10px',
            width: '300px',
            background: 'rgba(255, 255, 255, 0.8)',
            padding: '10px',
            borderRadius: '5px'
        }}>
            <Bar options={options} data={data} />
        </div>
    );
};

export default VolumeStats;
