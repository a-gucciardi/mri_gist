import React from 'react';

const ControlPanel = ({
    files,
    currentFile,
    onFileChange,
    onSegment,
    showStats,
    onToggleStats,
    colormap,
    onColormapChange
}) => {
    return (
        <div style={{
            position: 'absolute',
            top: '0',
            right: '0',
            width: '250px',
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            color: 'white',
            padding: '10px',
            margin: '10px',
            borderRadius: '5px',
            fontFamily: 'sans-serif'
        }}>
            <h3>Controls</h3>

            <div style={{ marginBottom: '10px' }}>
                <label style={{ display: 'block', marginBottom: '5px' }}>Select File:</label>
                <select
                    value={currentFile}
                    onChange={(e) => onFileChange(e.target.value)}
                    style={{ width: '100%', padding: '5px' }}
                >
                    {files.map(file => (
                        <option key={file} value={file}>{file}</option>
                    ))}
                </select>
            </div>

            <div style={{ marginBottom: '10px' }}>
                <label style={{ display: 'block', marginBottom: '5px' }}>Colormap:</label>
                <select
                    value={colormap}
                    onChange={(e) => onColormapChange(e.target.value)}
                    style={{ width: '100%', padding: '5px' }}
                >
                    <option value="gray">Gray</option>
                    <option value="viridis">Viridis</option>
                </select>
            </div>

            <button
                onClick={onSegment}
                style={{
                    width: '100%',
                    padding: '8px',
                    marginBottom: '10px',
                    backgroundColor: '#4CAF50',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer'
                }}
            >
                Run Segmentation
            </button>

            <div style={{ marginTop: '10px' }}>
                <label>
                    <input
                        type="checkbox"
                        checked={showStats}
                        onChange={(e) => onToggleStats(e.target.checked)}
                    /> Show Statistics
                </label>
            </div>
        </div>
    );
};

export default ControlPanel;
