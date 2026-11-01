document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const filePreview = document.getElementById('file-preview');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const fileIcon = document.getElementById('file-icon');
    const removeFile = document.getElementById('remove-file');
    const uploadButton = document.getElementById('upload-button');
    const uploadProgress = document.getElementById('upload-progress');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const thresholdInput = document.getElementById('confidence-threshold');
    const thresholdValue = document.getElementById('threshold-value');

    // Constants
    const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB
    const SUPPORTED_TYPES = {
        'image/jpeg': 'image',
        'image/png': 'image',
        'video/mp4': 'video',
        'video/quicktime': 'video'
    };

    // File Validation
    function validateFile(file) {
        if (!SUPPORTED_TYPES[file.type]) {
            showError('Unsupported file type. Please upload JPG, PNG, MP4, or MOV files.');
            return false;
        }

        if (file.size > MAX_FILE_SIZE) {
            showError('File size exceeds 100MB limit.');
            return false;
        }

        return true;
    }

    // Error Handling
    function showError(message) {
        // Create error message element
        const errorDiv = document.createElement('div');
        errorDiv.className = 'fixed top-4 right-4 bg-red-500 text-white px-6 py-3 rounded-lg shadow-lg';
        errorDiv.textContent = message;

        // Add to DOM
        document.body.appendChild(errorDiv);

        // Remove after 5 seconds
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }

    // File Preview
    function showFilePreview(file) {
        const fileType = SUPPORTED_TYPES[file.type];
        
        // Set file icon based on type
        fileIcon.innerHTML = `
            <svg class="w-8 h-8 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                ${fileType === 'image' ? 
                    '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />' :
                    '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />'
                }
            </svg>
        `;

        // Set file name and size
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);

        // Show preview
        filePreview.classList.remove('hidden');
        uploadButton.disabled = false;
    }

    // Format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Drag and Drop Handling
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropArea.classList.add('border-blue-500', 'bg-blue-50');
    }

    function unhighlight() {
        dropArea.classList.remove('border-blue-500', 'bg-blue-50');
    }

    // File Drop Handling
    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    // File Input Handling
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (validateFile(file)) {
                showFilePreview(file);
            }
        }
    }

    // Remove File
    removeFile.addEventListener('click', function() {
        fileInput.value = '';
        filePreview.classList.add('hidden');
        uploadButton.disabled = true;
    });

    // Confidence Threshold
    thresholdInput.addEventListener('input', function() {
        thresholdValue.textContent = `${this.value}%`;
    });

    // Upload Handling
    uploadButton.addEventListener('click', async function() {
        const file = fileInput.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);
        formData.append('model', document.getElementById('model-select').value);
        formData.append('confidence_threshold', thresholdInput.value / 100);
        formData.append('enhanced_analysis', document.getElementById('enhanced-analysis').checked);
        formData.append('save_results', document.getElementById('save-results').checked);

        try {
            uploadProgress.classList.remove('hidden');
            uploadButton.disabled = true;

            const response = await fetch('/api/v1/files/upload', {
                method: 'POST',
                body: formData,
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });

            if (!response.ok) {
                throw new Error('Upload failed');
            }

            const data = await response.json();
            window.location.href = `/results/${data.processing_id}`;
        } catch (error) {
            showError('Upload failed. Please try again.');
            uploadProgress.classList.add('hidden');
            uploadButton.disabled = false;
        }
    });
}); 