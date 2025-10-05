document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select a file.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('result').innerHTML = '<p>Error: ' + data.error + '</p>';
        } else {
            document.getElementById('result').innerHTML = '<p>Recognized Text: ' + data.text + '</p>';
        }
    })
    .catch(error => {
        document.getElementById('result').innerHTML = '<p>Error: ' + error.message + '</p>';
    });
});
