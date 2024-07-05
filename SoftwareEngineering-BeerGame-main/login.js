const config_temp = {
    authentication_url: "create_user_session.py", // python file
    cookie_name: "se-project" // has also in python file
}

window.onload = () => {
    const form = document.getElementById('userCredentials');
    form.addEventListener('submit', handleLoginSubmission);
    
    
    function handleLoginSubmission(event){
        console.log(event)
        console.log(form.elements);
        event.preventDefault();
        valuepairs = {}
        for (let i = 0;  i < form.elements.length; i++){
            if (form.elements[i].name)
                valuepairs[form.elements[i].name] = form.elements[i].value
        }
        console.log("logging valuepairs in jquery-less approach: " + JSON.stringify(valuepairs));
        verifyUser(valuepairs);

    }
}



function verifyUser(LoginCredentials){
    console.log("LoginCredentials: "+JSON.stringify(LoginCredentials));
    let xhr = new XMLHttpRequest();
    xhr.open("POST", config_temp.authentication_url, true);
    let body = JSON.stringify(LoginCredentials);

    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8")
    console.log('body is: '+ body);

    xhr.send(body);
    xhr.onload = () =>{
        let resp = xhr.responseText;
        console.log("response received! logging response . . . ");
        console.log(resp);
        checkResponse(resp);
    }
}
function checkResponse(resp){
    if (!resp){
        alert("invalid response received");
    }
    try{
        resp = JSON.parse(resp);
        if (resp['status'] >=  200 && resp['status'] <= 300){
            alert("success! you are now authenticated!");
            window.location.href="../index.html"
        }
        else{
            if (resp['description']){
                alert(resp['description'])
            }
            else alert("invalid username/password");
        }
    }
    catch (err){
        alert('an invalid response was received by the server, please try again later');
    }
}
function AddCookieToBrowser(cookie_json){
    if (! (cookie_json['status'] >=  200 && cookie_json['status'] <= 300) ){
        alert('error in user login');
        return;
    }
    /*
    let date = new Date()
    date.setMinutes(date.getMinutes() + 30);
    let cookie_string = config_temp.cookie_name + "="+cookie_json['session_id']+"; expires="+date+";path=/;domain=."+window.location.hostname;
    console.log(cookie_string);
    document.cookie = cookie_string;

    
    */


}