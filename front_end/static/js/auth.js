document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');

    // Form Validation
    function validateEmail(email) {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(email);
    }

    function validatePassword(password) {
        return password.length >= 8;
    }

    function showError(element, message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'mt-2 text-sm text-red-600';
        errorDiv.textContent = message;
        element.parentNode.appendChild(errorDiv);
        element.classList.add('border-red-500');
    }

    function clearErrors(form) {
        const errors = form.querySelectorAll('.text-red-600');
        errors.forEach(error => error.remove());
        const inputs = form.querySelectorAll('input');
        inputs.forEach(input => input.classList.remove('border-red-500'));
    }

    // Login Form Handler
    if (loginForm) {
        loginForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            clearErrors(loginForm);

            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            const rememberMe = document.getElementById('remember-me').checked;

            // Validate email
            if (!validateEmail(email)) {
                showError(document.getElementById('email'), 'Please enter a valid email address');
                return;
            }

            // Validate password
            if (!validatePassword(password)) {
                showError(document.getElementById('password'), 'Password must be at least 8 characters long');
                return;
            }

            try {
                const response = await fetch('/api/v1/auth/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        email,
                        password,
                        remember_me: rememberMe
                    })
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.message || 'Login failed');
                }

                // Store token
                localStorage.setItem('token', data.token);
                
                // Redirect to dashboard
                window.location.href = '/dashboard';
            } catch (error) {
                showError(loginForm, error.message);
            }
        });
    }

    // Registration Form Handler
    if (registerForm) {
        registerForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            clearErrors(registerForm);

            const name = document.getElementById('name').value;
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            const confirmPassword = document.getElementById('confirm-password').value;
            const terms = document.getElementById('terms').checked;

            // Validate name
            if (name.length < 2) {
                showError(document.getElementById('name'), 'Name must be at least 2 characters long');
                return;
            }

            // Validate email
            if (!validateEmail(email)) {
                showError(document.getElementById('email'), 'Please enter a valid email address');
                return;
            }

            // Validate password
            if (!validatePassword(password)) {
                showError(document.getElementById('password'), 'Password must be at least 8 characters long');
                return;
            }

            // Validate password match
            if (password !== confirmPassword) {
                showError(document.getElementById('confirm-password'), 'Passwords do not match');
                return;
            }

            // Validate terms
            if (!terms) {
                showError(document.getElementById('terms'), 'You must agree to the terms and conditions');
                return;
            }

            try {
                const response = await fetch('/api/v1/auth/register', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        name,
                        email,
                        password
                    })
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.message || 'Registration failed');
                }

                // Store token
                localStorage.setItem('token', data.token);
                
                // Redirect to dashboard
                window.location.href = '/dashboard';
            } catch (error) {
                showError(registerForm, error.message);
            }
        });
    }

    // Social Login Handlers
    document.querySelectorAll('.social-login').forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const provider = this.dataset.provider;
            window.location.href = `/api/v1/auth/${provider}`;
        });
    });
}); 