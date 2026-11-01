document.addEventListener('DOMContentLoaded', function() {
    // File upload handling
    const fileUpload = document.getElementById('file-upload');
    const uploadArea = document.querySelector('.border-2');
    const uploadButton = document.getElementById('upload-button');
    const progressBar = document.getElementById('upload-progress');
    const progressFill = progressBar.querySelector('.bg-blue-600');
    const thresholdInput = document.getElementById('confidence-threshold');
    const thresholdValue = document.getElementById('threshold-value');

    // Drag and drop handling
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        uploadArea.classList.add('border-blue-500', 'bg-blue-50');
    }

    function unhighlight(e) {
        uploadArea.classList.remove('border-blue-500', 'bg-blue-50');
    }

    uploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    fileUpload.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (isValidFileType(file)) {
                uploadButton.disabled = false;
            } else {
                alert('Please upload a valid image or video file.');
            }
        }
    }

    function isValidFileType(file) {
        const validTypes = ['image/jpeg', 'image/png', 'video/mp4', 'video/quicktime'];
        return validTypes.includes(file.type);
    }

    // Confidence threshold handling
    thresholdInput.addEventListener('input', function() {
        thresholdValue.textContent = `${this.value}%`;
    });

    // Upload button handling
    uploadButton.addEventListener('click', async function() {
        const file = fileUpload.files[0];
        if (!file) {
            alert('Please select a file to upload.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('model', document.getElementById('model-select').value);
        formData.append('confidence_threshold', thresholdInput.value / 100);

        try {
            progressBar.classList.remove('hidden');
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
            alert('Upload failed. Please try again.');
            progressBar.classList.add('hidden');
            uploadButton.disabled = false;
        }
    });

    // Authentication state handling
    const token = localStorage.getItem('token');
    if (token) {
        document.querySelectorAll('.auth-required').forEach(el => {
            el.classList.remove('hidden');
        });
        document.querySelectorAll('.auth-not-required').forEach(el => {
            el.classList.add('hidden');
        });
    } else {
        document.querySelectorAll('.auth-required').forEach(el => {
            el.classList.add('hidden');
        });
        document.querySelectorAll('.auth-not-required').forEach(el => {
            el.classList.remove('hidden');
        });
    }
}); 