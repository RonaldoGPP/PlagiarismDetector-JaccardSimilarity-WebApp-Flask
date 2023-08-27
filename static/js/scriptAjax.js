jQuery(document).ready(function($){
    var $form_modal = $('.user-modal'),
      $form_login = $form_modal.find('#login'),
      $form_signup = $form_modal.find('#signup'),
      $form_forgot_password = $form_modal.find('#reset-password'),
      $form_modal_tab = $('.switcher'),
      $tab_login = $form_modal_tab.children('li').eq(0).children('a'),
      $tab_signup = $form_modal_tab.children('li').eq(1).children('a'),
      $forgot_password_link = $form_login.find('.form-bottom-message a'),
      $back_to_login_link = $form_forgot_password.find('.form-bottom-message a'),
      $main_nav = $('.main-nav');
      $signup = $('.signup');
      $signin = $('.signin');
    $(document).ready(function() {
        // Submit login form
        $form_login.find('input[type="submit"]').on('click', function(e){
        // Prevent form submission

        var email = $('#signin-email').val();
        var password = $('#signin-password').val(); // Serialize form data
        $.ajax({
            type: "POST",
            url: "/login",
            data: {
            'signin-email': email,
            'signin-password': password
        },
            success: function(response) {
            if (response.success) {
                // Login successful, perform desired action (e.g., show success message)
                // You can modify this part according to your needs
                alert("Login successful");
                window.location.href = '/home'
            } else {
                // Login failed, display the error message in the login modal
                $("#login-error").text(response.error).show();
                $form_login.find('input[type="submit"]').toggleClass('has-error').next('span').toggleClass('is-visible');
            }
            },
            error: function() {
            // Error occurred, display a generic error message or handle it accordingly
            
            $("#login-error").text("An error occurred during login.").show();
            $form_login.find('input[type="submit"]').toggleClass('has-error').next('span').toggleClass('is-visible');
            
            }
        });
        e.preventDefault(); 
        });

        // Submit signup form
        $form_signup.find('input[type="submit"]').on('click', function(e){
        // Prevent form submission

        var username = $('#signup-username').val();
        var email = $('#signup-email').val();
        var password = $('#signup-password').val(); // Serialize form data
        $.ajax({
            type: "POST",
            url: "/signup",
            data: {
            'signup-username': username,
            'signup-email': email,
            'signup-password': password
        },
            success: function(response) {
            if (response.success) {
                // Signup successful, perform desired action (e.g., show success message)
                // You can modify this part according to your needs
                alert("Signup successful");
                window.location.href = '/home'
            } else {
                // Signup failed, display the error message in the signup modal
                
                $("#signup-error").text(response.error).show();
                $form_signup.find('input[type="submit"]').toggleClass('has-error').next('span').toggleClass('is-visible');
                
            }
            },
            error: function() {
            // Error occurred, display a generic error message or handle it accordingly
            $("#signup-error").text("An error occurred during signup.");
            $form_signup.find('input[type="submit"]').toggleClass('has-error').next('span').toggleClass('is-visible');
            }
        });
        e.preventDefault(); 
        });
        // Submit upload form
        $("#upload-form").find('input[type="submit"]').on('click', function(e){
    
            var fileInput = $('#fileinput')[0].files[0];

            // Create a new FormData object
            var formData = new FormData();
            formData.append('file', fileInput);
            $.ajax({
                type: "POST",
                url: "/upload",
                data: formData,
                success: function(response) {
                if (response.success) {
                    alert("Upload successful");
                    window.location.href = '/uploadform'
                } else {
                    $("#upload-error").text(response.error).show();
                    $("#upload-form").find('input[type="submit"]').toggleClass('has-error').next('span').toggleClass('is-visible');
                }
                },
                error: function() {
                // Error occurred, display a generic error message or handle it accordingly
                $("#upload-error").text("An error occurred during upload.");
                $("#upload-form").find('input[type="submit"]').toggleClass('has-error').next('span').toggleClass('is-visible');
                }
            });
            e.preventDefault(); 
            });
    });
});