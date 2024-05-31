document.getElementById('image-upload').addEventListener('change', function() {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('selected-image').style.display = "block"; // Display the selected image preview
            document.getElementById('selected-image').src = e.target.result;
        }
        reader.readAsDataURL(file);
    }
});
