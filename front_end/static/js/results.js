document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const loadingState = document.getElementById('loading-state');
    const resultsDisplay = document.getElementById('results-display');
    const errorState = document.getElementById('error-state');
    const processingProgress = document.getElementById('processing-progress');
    const processingStatus = document.getElementById('processing-status');
    const mediaContainer = document.getElementById('media-container');
    const mediaOverlay = document.getElementById('media-overlay');
    const playButton = document.getElementById('play-button');
    const deepfakeProbability = document.getElementById('deepfake-probability');
    const deepfakePercentage = document.getElementById('deepfake-percentage');
    const confidenceLevel = document.getElementById('confidence-level');
    const confidencePercentage = document.getElementById('confidence-percentage');
    const modelUsed = document.getElementById('model-used');
    const processingTime = document.getElementById('processing-time');
    const fileType = document.getElementById('file-type');
    const fileSize = document.getElementById('file-size');
    const detailedAnalysis = document.getElementById('detailed-analysis');
    const recommendations = document.getElementById('recommendations');
    const downloadReport = document.getElementById('download-report');
    const shareResults = document.getElementById('share-results');
    const retryButton = document.getElementById('retry-button');
    const errorMessage = document.getElementById('error-message');

    // Get processing ID from URL
    const processingId = window.location.pathname.split('/').pop();

    // Initialize media player
    let mediaPlayer = null;

    // Fetch results
    async function fetchResults() {
        try {
            const response = await fetch(`/api/v1/processing/${processingId}`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });

            if (!response.ok) {
                throw new Error('Failed to fetch results');
            }

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error fetching results:', error);
            showError('Failed to fetch results. Please try again.');
            return null;
        }
    }

    // Update loading progress
    function updateProgress(progress, status) {
        processingProgress.style.width = `${progress}%`;
        processingStatus.textContent = status;
    }

    // Show error state
    function showError(message) {
        loadingState.classList.add('hidden');
        resultsDisplay.classList.add('hidden');
        errorState.classList.remove('hidden');
        errorMessage.textContent = message;
    }

    // Show results
    function showResults(data) {
        loadingState.classList.add('hidden');
        errorState.classList.add('hidden');
        resultsDisplay.classList.remove('hidden');

        // Update media preview
        if (data.media_type === 'image') {
            mediaContainer.innerHTML = `<img src="${data.media_url}" alt="Analyzed media" class="w-full h-full object-contain">`;
        } else {
            mediaContainer.innerHTML = `
                <video class="w-full h-full object-contain" controls>
                    <source src="${data.media_url}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            `;
            mediaPlayer = mediaContainer.querySelector('video');
            mediaOverlay.classList.remove('hidden');
        }

        // Update detection results
        deepfakeProbability.style.width = `${data.deepfake_probability * 100}%`;
        deepfakePercentage.textContent = `${(data.deepfake_probability * 100).toFixed(1)}%`;
        confidenceLevel.style.width = `${data.confidence_level * 100}%`;
        confidencePercentage.textContent = `${(data.confidence_level * 100).toFixed(1)}%`;

        // Update analysis details
        modelUsed.textContent = data.model_used;
        processingTime.textContent = `${data.processing_time.toFixed(2)} seconds`;
        fileType.textContent = data.media_type.toUpperCase();
        fileSize.textContent = formatFileSize(data.file_size);

        // Update detailed analysis
        detailedAnalysis.innerHTML = data.detailed_analysis.map(item => `
            <div class="flex justify-between items-center">
                <span class="text-gray-600">${item.feature}</span>
                <div class="flex items-center">
                    <div class="w-32 h-2 bg-gray-200 rounded-full mr-2">
                        <div class="h-2 rounded-full ${item.score > 0.7 ? 'bg-red-500' : 'bg-green-500'}" style="width: ${item.score * 100}%"></div>
                    </div>
                    <span class="text-gray-800 font-medium">${(item.score * 100).toFixed(1)}%</span>
                </div>
            </div>
        `).join('');

        // Update recommendations
        recommendations.innerHTML = data.recommendations.map(rec => `
            <div class="flex items-start">
                <div class="flex-shrink-0 mt-1">
                    <svg class="w-5 h-5 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
                    </svg>
                </div>
                <p class="ml-3 text-gray-700">${rec}</p>
            </div>
        `).join('');
    }

    // Format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Event Listeners
    playButton.addEventListener('click', function() {
        if (mediaPlayer) {
            mediaPlayer.play();
            mediaOverlay.classList.add('hidden');
        }
    });

    downloadReport.addEventListener('click', async function() {
        try {
            const response = await fetch(`/api/v1/processing/${processingId}/report`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });

            if (!response.ok) {
                throw new Error('Failed to download report');
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `analysis_report_${processingId}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (error) {
            console.error('Error downloading report:', error);
            showError('Failed to download report. Please try again.');
        }
    });

    shareResults.addEventListener('click', function() {
        // Implement sharing functionality
        alert('Sharing functionality will be implemented soon!');
    });

    retryButton.addEventListener('click', function() {
        window.location.reload();
    });

    // Poll for results
    let pollInterval = setInterval(async function() {
        const data = await fetchResults();
        if (data) {
            if (data.status === 'completed') {
                clearInterval(pollInterval);
                showResults(data);
            } else if (data.status === 'failed') {
                clearInterval(pollInterval);
                showError(data.error || 'Analysis failed. Please try again.');
            } else {
                updateProgress(data.progress, data.status);
            }
        }
    }, 2000);

    // Cleanup on page unload
    window.addEventListener('beforeunload', function() {
        clearInterval(pollInterval);
    });
}); 