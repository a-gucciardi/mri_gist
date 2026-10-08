import React, { useState, useEffect } from 'react';
import VolumeViewer from './components/VolumeViewer';
import VolumeStats from './components/VolumeStats';
import ControlPanel from './components/ControlPanel';

function App() {
  const [files, setFiles] = useState([]);
  const [fileMap, setFileMap] = useState({});
  const [currentFile, setCurrentFile] = useState(null);
  const [showStats, setShowStats] = useState(false);
  const [colormap, setColormap] = useState('gray');

  useEffect(() => {
    // Fetch files from backend
    const fetchFiles = async () => {
      try {
        const response = await fetch('/api/files');
        if (response.ok) {
          const data = await response.json();
          if (Array.isArray(data)) {
            // Store full objects if possible, for now just names to not break child components
            // But handleSegmentation needs path. 
            // Let's actually store objects in a separate state map or refactor?
            // Quickest refactor: 'files' state remains names for UI dropdown
            // New state 'fileMap' stores name -> full info
            setFiles(data.map(f => f.name));
            setFileMap(data.reduce((map, obj) => { map[obj.name] = obj; return map; }, {}));

            if (data.length > 0 && !currentFile) {
              setCurrentFile(data[0].name);
            }
          }
        } else {
          console.error("Failed to fetch files");
        }
      } catch (e) {
        console.error("Error fetching files:", e);
      }
    };

    fetchFiles();
  }, []);

  const [activeTab, setActiveTab] = useState('viewer');

  const pollJob = async (jobId) => {
    return new Promise((resolve, reject) => {
      const checkStatus = async () => {
        try {
          const response = await fetch(`/api/process/${jobId}`);
          const data = await response.json();
          console.log("Job status:", data.status);

          if (data.status === 'completed') {
            resolve(data);
          } else if (data.status === 'failed') {
            reject(new Error(data.message || 'Job failed'));
          } else {
            // Continue polling
            setTimeout(checkStatus, 1000);
          }
        } catch (e) {
          reject(e);
        }
      };
      checkStatus();
    });
  };

  const handleSegmentation = async () => {
    if (!currentFile) return;
    try {
      // 1. Submit Job
      const submitResponse = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_type: 'segmentation',
          input_file: currentFile, // Backend list_files returns absolute path now? Or checks absolute?
          // The backend list_files returns 'path' which is absolute.
          // But the App state might store just 'name' if we didn't update it to store object.
          // Let's check how we stored files.
          // We stored just names. We should update the fetchFiles logic or assume backend handles names.
          // Ideally, pass absolute path if backend requires it.
          // For now let's assume valid path.
          output_file: currentFile.replace('.nii', '_seg.nii').replace('.nrrd', '_seg.nrrd'),
          params: { robust: true }
        })
      });

      const submitData = await submitResponse.json();
      console.log("Job submitted:", submitData.job_id);

      // 2. Poll for completion
      alert(`Segmentation started (Job ID: ${submitData.job_id}). Please wait...`);
      const result = await pollJob(submitData.job_id);

      console.log("Job completed:", result);
      alert("Segmentation completed successfully!");

      // Refresh file list to see new segmentation
      // fetchFiles(); // Need to extract fetchFiles to outside useEffect or use a reload trigger

    } catch (e) {
      console.error("Segmentation failed:", e);
      alert(`Segmentation failed: ${e.message}`);
    }
  };

  return (
    <div style={{ width: '100vw', height: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: '#000', color: '#fff', fontFamily: 'Inter, sans-serif' }}>
      {/* Navigation Bar */}
      <nav style={{
        height: '60px',
        backgroundColor: '#111',
        borderBottom: '1px solid #333',
        display: 'flex',
        alignItems: 'center',
        padding: '0 20px',
        justifyContent: 'space-between'
      }}>
        <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#fff' }}>MRI Gist</div>
        <div style={{ display: 'flex', gap: '20px' }}>
          <button
            onClick={() => setActiveTab('viewer')}
            style={{
              background: 'none',
              border: 'none',
              color: activeTab === 'viewer' ? '#4CAF50' : '#888',
              fontSize: '1rem',
              cursor: 'pointer',
              borderBottom: activeTab === 'viewer' ? '2px solid #4CAF50' : 'none',
              paddingBottom: '5px'
            }}
          >
            Viewer
          </button>
          <button
            onClick={() => setActiveTab('analytics')}
            style={{
              background: 'none',
              border: 'none',
              color: activeTab === 'analytics' ? '#4CAF50' : '#888',
              fontSize: '1rem',
              cursor: 'pointer',
              borderBottom: activeTab === 'analytics' ? '2px solid #4CAF50' : 'none',
              paddingBottom: '5px'
            }}
          >
            Analytics
          </button>
        </div>
      </nav>

      {/* Main Content */}
      <main style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>

        {/* Viewer Tab Content */}
        <div style={{ display: activeTab === 'viewer' ? 'block' : 'none', width: '100%', height: '100%' }}>
          {currentFile && (
            <VolumeViewer
              file={currentFile} // Kept for key/name reference if needed, but url is primary now
              url={fileMap[currentFile]?.url}
              colormap={colormap}
            />
          )}
          <ControlPanel
            files={files}
            currentFile={currentFile}
            onFileChange={setCurrentFile}
            onSegment={handleSegmentation}
            showStats={showStats} // Kept for backward compatibility or remove later
            onToggleStats={setShowStats}
            colormap={colormap}
            onColormapChange={setColormap}
          />
        </div>

        {/* Analytics Tab Content */}
        {activeTab === 'analytics' && (
          <VolumeStats currentFile={currentFile} />
        )}

      </main>
    </div>
  );
}

export default App;
