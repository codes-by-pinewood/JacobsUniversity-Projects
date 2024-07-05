const config_temp = {
    post_member: "post_user.py"
  }
    
  
  
  window.onload = function () {


      let form = document.getElementById('create-member');
  
      form.addEventListener('submit', function(event){
        event.preventDefault();
        res = handleSubmission(form);
        res['permission'] = 0;
        console.log(res);
        sendJson(JSON.stringify(res), config_temp.post_member, redirectuser);
    }, false);

  };
  function redirectuser(){
    window.location.href="welcomepage.html"
  }