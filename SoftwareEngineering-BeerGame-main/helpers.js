
//creating a list of all the references for a given relationship
function GetSelectElementFromDatabase(html_id, link){
    let req = new XMLHttpRequest();
    req.open('GET', link, true);
    req.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    req.send();
    req.onload = () => {
        let resp = req.responseText;
        let elemDict = JSON.parse(resp);
        updateAvailableSelect(elemDict, html_id);
    }
}
function updateAvailableSelect(clist, html_id){
   let s = document.getElementById(html_id);
   for ( var key in  clist){
       var opt = document.createElement('option');
       opt.text = clist[key];
       opt.id=key;
       opt.value = key;
       s.add(opt);
   }
}

//gets all the input data and creates a json object with it
function handleSubmission(form){
    let valuepairs = {}
    for (let i = 0;  i < form.elements.length; i++){
        if (form.elements[i].name)
            valuepairs[form.elements[i].name] = form.elements[i].value
    }
   return valuepairs;              
}

//sends the json object to the backend/where the 
//necessary checking is done and returns the response
function sendJson(submission, post_route, callback=helperCallBack){
    let xhr = new XMLHttpRequest();
    xhr.open('POST', post_route, 'true');
    xhr.setRequestHeader("Content-Type", "multipart/form-data");
    xhr.send(submission);
    xhr.onload = () => {
        callback(xhr);
    }
    
}   
function helperCallBack(xhr){
    if (! (xhr.status >=200 && xhr.status <= 205) ){
        alert("an error occured: "+ xhr.responseText)
    }
    let resp = xhr.responseText;
    console.log(resp);
    alert(xhr.responseText);
}