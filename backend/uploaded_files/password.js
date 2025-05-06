
document.addEventListener('DOMContentLoaded', function () {
  const button = document.getElementById('submitButton');
  const realPassword = "frankbutt";
  
  button.addEventListener('click', function () {
    const password = document.getElementById('passwordInput').value;
    if(password === realPassword)
    {
      window.location.replace('Transcript.html');
    }
  });
});