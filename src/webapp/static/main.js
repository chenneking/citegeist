// Constants
const POLLING_RATE_MS = 1000;

// Update slider values live
document.getElementById('slider-breadth').addEventListener('input', (e) => {
    document.getElementById('slider-breadth-value').textContent = e.target.value;
});
document.getElementById('slider-depth').addEventListener('input', (e) => {
    document.getElementById('slider-depth-value').textContent = e.target.value;
});
document.getElementById('slider-diversity').addEventListener('input', (e) => {
    document.getElementById('slider-diversity-value').textContent = e.target.value;
});

// Toggle input type
const inputToggle = document.getElementById('input-toggle');
const textInputSection = document.getElementById('text-input-section');
const pdfInputSection = document.getElementById('pdf-input-section');
const pdfFileInput = document.getElementById('pdf-file-input');
const pdfFileName = document.getElementById('pdf-file-name');

inputToggle.addEventListener('change', () => {
    if (inputToggle.checked) {
        textInputSection.style.display = 'none';
        pdfInputSection.style.display = 'block';
    } else {
        textInputSection.style.display = 'block';
        pdfInputSection.style.display = 'none';
    }
});

// PDF file selection
pdfFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        pdfFileName.textContent = file.name;
    } else {
        pdfFileName.textContent = 'No file selected';
    }
});

// Update progress bar and text
function updateProgress(progress, statusText) {
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const progressPercentage = document.getElementById('progress-percentage');
    
    progressBar.style.width = `${progress}%`;
    progressBar.setAttribute('aria-valuenow', progress);
    progressText.innerText = statusText;
    progressPercentage.innerText = `${progress}%`;
}

// Poll Job status
async function pollJobStatus(jobId, intervalMs = POLLING_RATE_MS) {
    // Show progress section and hide output sections
    document.getElementById('progress-section').style.display = 'block';
    document.getElementById('related-works-section').style.display = 'none';
    document.getElementById('citations-section').style.display = 'none';
    
    let pollInterval;
    
    try {
        pollInterval = setInterval(async () => {
            try {
                const response = await fetch(`/status/${jobId}`);
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const jobStatus = await response.json();
                
                // Update progress bar and text
                updateProgress(jobStatus.progress, jobStatus.status_text);
                
                // Check if job is complete
                if (jobStatus.status === 'completed' && jobStatus.result) {
                    clearInterval(pollInterval);
                    
                    // Show result sections
                    document.getElementById('related-works-section').style.display = 'block';
                    document.getElementById('citations-section').style.display = 'block';
                    
                    // Populate results
                    document.getElementById('related-works-output').innerText =
                        jobStatus.result['related_works'] || 'No related works found.';
                    document.getElementById('citations-output').innerText =
                        jobStatus.result.citations?.join('\n\n') || 'No citations found.';
                        
                    // Re-enable generate button
                    document.getElementById('generate-button').disabled = false;
                }
                
                // Check if job failed
                if (jobStatus.status === 'failed') {
                    clearInterval(pollInterval);
                    throw new Error(jobStatus.error || 'Job failed for unknown reason');
                }
                
            } catch (error) {
                clearInterval(pollInterval);
                console.error('Error polling job status:', error);
                alert(`Error: ${error.message}`);
                document.getElementById('generate-button').disabled = false;
            }
        }, intervalMs);
    } catch (error) {
        if (pollInterval) clearInterval(pollInterval);
        console.error('Error setting up polling:', error);
        alert(`Error: ${error.message}`);
        document.getElementById('generate-button').disabled = false;
    }
}

// Handle the Generate button click
document.getElementById('generate-button').addEventListener('click', async () => {
    // Disable the button
    const button = document.getElementById('generate-button');
    button.disabled = true;

    // Clear previous outputs
    document.getElementById('related-works-output').innerText = '';
    document.getElementById('citations-output').innerText = '';

    // Initialize progress 
    updateProgress(0, 'Starting job...');
    document.getElementById('progress-section').style.display = 'block';

    // Collect input data
    const breadth = document.getElementById('slider-breadth').value;
    const depth = document.getElementById('slider-depth').value;
    const diversity = document.getElementById('slider-diversity').value;

    // Prepare form data
    const formData = new FormData();
    formData.append('breadth', parseInt(breadth));
    formData.append('depth', parseInt(depth));
    formData.append('diversity', parseFloat(diversity));

    // Determine input type and add appropriate data
    if (inputToggle.checked) {
        // PDF upload
        const pdfFile = pdfFileInput.files[0];
        if (!pdfFile) {
            alert('Please select a PDF file');
            button.disabled = false;
            return;
        }
        formData.append('pdf', pdfFile);
    } else {
        // Text input
        const abstract = document.getElementById('abstract-input').value;
        if (!abstract.trim()) {
            alert('Please enter an abstract');
            button.disabled = false;
            return;
        }
        formData.append('abstract', abstract);
    }

    try {
        // Send POST request to /create-job to create a job
        const response = await fetch('/create-job', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        // Get job ID from response
        const result = await response.json();
        const jobId = result.job_id;
        
        if (!jobId) {
            throw new Error('No job ID returned from server');
        }
        
        // Start polling for job status
        pollJobStatus(jobId);
        
    } catch (error) {
        console.error('Error starting job:', error);
        alert(`Error: ${error.message}`);
        button.disabled = false;
        document.getElementById('progress-section').style.display = 'none';
    }
});